#include "typeOperation.h"
#include "StreamlineDistance.h"
#include "helper_cuda.h"

#define BLOCK_WIDTH		1
#define BLOCK_HEIGHT	16
#define GRID_WIDTH		1

__global__ void computeClosestPointDistanceMatrixInSmallPools_d(Streamline* stls, vec3f* points, int num_stls,
																float* ret)
{
	int idx1 = blockDim.x*blockIdx.x+threadIdx.x;
	int idx2 = blockDim.y*blockIdx.y+threadIdx.y;

	if(idx1<num_stls && idx2<num_stls){
		Streamline s1 = stls[idx1];
		Streamline s2 = stls[idx2];
		ret[idx1*num_stls+idx2] = computeClosestPointDistance(&points[s1.start], &points[s2.start], s1.numPoint, s2.numPoint);
	}
}

__global__ void computeClosestPointDistanceFromLineToSmallPools_d(Streamline* stls, vec3f* points, int num_stls,
																vec3f* line_points, int num_line_points,
																float* ret)
{
	int idx = blockDim.x*blockIdx.x+threadIdx.x;

	if(idx<num_stls){
		Streamline s = stls[idx];
		float dist1 = computeClosestPointDistance(&points[s.start], line_points, s.numPoint, num_line_points);
		float dist2 = computeClosestPointDistance(line_points, &points[s.start], num_line_points, s.numPoint);
		ret[idx] = (dist1>dist2)?(dist1):(dist2);
	}
}


__global__ void computeMinMeanLineToGroupDistance_d(Streamline* stl1, Streamline* stl2, 
						vec3f* points1, vec3f* points2,
						int* group_members,
						int num_stl1, int group_size, float* ret)
{
	int idx1 = blockDim.x*blockIdx.x+threadIdx.x;
	int idx2 = blockDim.y*blockIdx.y+threadIdx.y;

	if(idx1<num_stl1 && idx2<group_size){
		Streamline s1 = stl1[idx1];
		Streamline s2 = stl2[group_members[idx2]];
		ret[idx1*group_size+idx2] = computeClosestPointDistance(&points1[s1.start], &points2[s2.start], s1.numPoint, s2.numPoint);
	}
}

__global__ void computeClosestPointDistanceMin_d(Streamline* stl1, Streamline* stl2, 
												 vec3f* points1, vec3f* points2, 
												 int num_stl1, int num_stl2, 
												 int idx1_offset, int idx1_num,
												 float* ret)
{
	int idx1 = blockDim.x*blockIdx.x+threadIdx.x;
	int idx2 = blockDim.y*blockIdx.y+threadIdx.y;

	if(idx1<idx1_num && (idx1+idx1_offset)<num_stl1 && idx2<num_stl2){
		idx1 += idx1_offset;
		Streamline s1 = stl1[idx1];
		Streamline s2 = stl2[idx2];
		ret[idx2*num_stl1+idx1] = computeClosestPointDistanceMin(&points1[s1.start], &points2[s2.start], s1.numPoint, s2.numPoint);
	}
}

__global__ void computeClosestPointDistance_d(Streamline* stls, vec3f* points, int num_stls,
											  int idx1_offset, int idx1_num,
											  float bin_width, int num_bin, 
											  float* avg, float* minv, float* maxv, short* hist)
{
	int idx1 = blockDim.x*blockIdx.x+threadIdx.x;
	int idx2 = blockDim.y*blockIdx.y+threadIdx.y;

	if(idx1<idx1_num && (idx1+idx1_offset)<num_stls && idx2<num_stls){
		idx1 += idx1_offset;
		if (idx1!=idx2) {
			float tmp_avg, tmp_minv, tmp_maxv;
			computeClosestPointDisanceDistribution(&points[stls[idx1].start], &points[stls[idx2].start], 
				stls[idx1].numPoint, stls[idx2].numPoint,
				bin_width, num_bin,
				&hist[(idx1*num_stls+idx2)*num_bin],
				tmp_avg, tmp_minv, tmp_maxv);

			avg[idx1*num_stls+idx2] = tmp_avg;
			minv[idx1*num_stls+idx2] = tmp_minv;
			maxv[idx1*num_stls+idx2] = tmp_maxv;
		} else {
			avg[idx1*num_stls+idx2] = 0.0f;
			minv[idx1*num_stls+idx2] = 0.0f;
			maxv[idx1*num_stls+idx2] = 0.0f;

			for (int i=(idx1*num_stls+idx2)*num_bin, j=0; j<num_bin; ++j, ++i) {
				hist[i] = 0;
			}
		}
	}
}

__global__ void computeMCPOneWay_d(vec3f* from, vec3f* to, int from_num, int to_num, float* ret){
	int idx = blockDim.x*blockIdx.x+threadIdx.x;

	if (idx<from_num) {
		vec3f diff = {from[idx].x-to[0].x, from[idx].y-to[0].y, from[idx].z-to[0].z};
		float minv = diff*diff, dist;
		for (int i=1; i<to_num; ++i) {
			diff = makeVec3f(from[idx].x-to[i].x, from[idx].y-to[i].y, from[idx].z-to[i].z);
			if ((dist=diff*diff)<minv) {
				minv = dist;
			}
		}
		ret[idx] = sqrtf(minv);
	}
}

__global__ void computeMCPOneWay_d(vec4f* from, vec4f* to, int from_num, int to_num, float* ret){
	int idx = blockDim.x*blockIdx.x+threadIdx.x;

	if (idx<from_num) {
		vec3f diff = {from[idx].x-to[0].x, from[idx].y-to[0].y, from[idx].z-to[0].z};
		float minv = diff*diff, dist;
		for (int i=1; i<to_num; ++i) {
			diff = makeVec3f(from[idx].x-to[i].x, from[idx].y-to[i].y, from[idx].z-to[i].z);
			if ((dist=diff*diff)<minv) {
				minv = dist;
			}
		}
		ret[idx] = sqrtf(minv);
	}
}

extern "C"
float* computeMinClosestPointDistanceMatrix_h(Streamline* stl1, Streamline* stl2, 
															vec3f* points1, vec3f* points2, 
															int num_stl1, int num_stl2)
{
	int num_p1 = stl1[num_stl1-1].start+stl1[num_stl1-1].numPoint;
	int num_p2 = stl2[num_stl2-1].start+stl2[num_stl2-1].numPoint;

	//allocate memory to store depth
	Streamline *stl1_d, *stl2_d;
	vec3f *point1_d, *points2_d;
	bool *filter_d;
	checkCudaErrors(cudaMalloc((void**)&stl1_d, sizeof(Streamline)*num_stl1));
	checkCudaErrors(cudaMalloc((void**)&stl2_d, sizeof(Streamline)*num_stl2));
	checkCudaErrors(cudaMalloc((void**)&point1_d, sizeof(vec3f)*num_p1));
	checkCudaErrors(cudaMalloc((void**)&points2_d, sizeof(vec3f)*num_p2));
	checkCudaErrors(cudaMalloc((void**)&filter_d, sizeof(bool)*num_p1));
	checkCudaErrors(cudaMemcpy(stl1_d, stl1, sizeof(Streamline)*num_stl1, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(stl2_d, stl2, sizeof(Streamline)*num_stl2, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(point1_d, points1, sizeof(vec3f)*num_p1, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(points2_d, points2, sizeof(vec3f)*num_p2, cudaMemcpyHostToDevice));

	float *ret = new float[num_stl1*num_stl2], *ret_d;
	checkCudaErrors(cudaMalloc((void**)&ret_d, sizeof(float)*num_stl1*num_stl2));

	float one_step_percentage = (100.0f*(BLOCK_WIDTH*GRID_WIDTH))/num_stl1, percentage=0.0f;
	checkCudaErrors(cudaThreadSynchronize());
	dim3 threadsPerBlock(BLOCK_WIDTH,BLOCK_HEIGHT);
	dim3 numBlocks(GRID_WIDTH,iDivUp(num_stl2,BLOCK_HEIGHT));
	for (int i=0; i<num_stl1; i+=BLOCK_WIDTH*GRID_WIDTH) {
		computeClosestPointDistanceMin_d<<<numBlocks,threadsPerBlock>>>(stl1_d, stl2_d, point1_d, points2_d, num_stl1, num_stl2,
			i, BLOCK_WIDTH*GRID_WIDTH, ret_d);
		checkCudaErrors(cudaThreadSynchronize());

		percentage += one_step_percentage;
		if (percentage<99.999999f) {
			printf("\rDistance Matrix Computation: %6.4f%%", percentage);
		} else {
			printf("\rDistance Matrix Computation: 100.000%%\n");
		}
	}

	checkCudaErrors(cudaThreadSynchronize());
	checkCudaErrors(cudaMemcpy(ret, ret_d, sizeof(float)*num_stl1*num_stl2, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(stl1_d));
	checkCudaErrors(cudaFree(stl2_d));
	checkCudaErrors(cudaFree(point1_d));
	checkCudaErrors(cudaFree(points2_d));
	checkCudaErrors(cudaFree(ret_d));

	return ret;
}

extern "C"
float *genMinMeanLineToGroupDistance_h(Streamline* stl1, Streamline* stl2, vec3f* points1, vec3f* points2, 
									   const int& num_stl1, const int& num_stl2, int* group_member, const int& group_size)
{
	int num_p1 = stl1[num_stl1-1].start+stl1[num_stl1-1].numPoint;
	int num_p2 = stl2[num_stl2-1].start+stl2[num_stl2-1].numPoint;

	//allocate memory to store depth
	Streamline *stl1_d, *stl2_d;
	vec3f *point1_d, *points2_d;
	int* group_member_d;
	cudaMalloc((void**)&stl1_d, sizeof(Streamline)*num_stl1);
	cudaMalloc((void**)&stl2_d, sizeof(Streamline)*num_stl2);
	cudaMalloc((void**)&point1_d, sizeof(vec3f)*num_p1);
	cudaMalloc((void**)&points2_d, sizeof(vec3f)*num_p2);
	cudaMalloc((void**)&group_member_d, sizeof(int)*group_size);
	cudaMemcpy(stl1_d, stl1, sizeof(Streamline)*num_stl1, cudaMemcpyHostToDevice);
	cudaMemcpy(stl2_d, stl2, sizeof(Streamline)*num_stl2, cudaMemcpyHostToDevice);
	cudaMemcpy(point1_d, points1, sizeof(vec3f)*num_p1, cudaMemcpyHostToDevice);
	cudaMemcpy(points2_d, points2, sizeof(vec3f)*num_p2, cudaMemcpyHostToDevice);
	cudaMemcpy(group_member_d, group_member, sizeof(int)*group_size, cudaMemcpyHostToDevice);

	float *mat = new float[num_stl1*num_stl2], *mat_d;
	cudaMalloc((void**)&mat_d, sizeof(float)*num_stl1*num_stl2);

	dim3 threadsPerBlock(16,16);
	dim3 numBlocks(iDivUp(num_stl1,16),iDivUp(group_size,16));
	computeMinMeanLineToGroupDistance_d<<<numBlocks,threadsPerBlock>>>(stl1_d, stl2_d, point1_d, points2_d, group_member_d,num_stl1, group_size, mat_d);
	cudaMemcpy(mat, mat_d, sizeof(float)*num_stl1*num_stl2, cudaMemcpyDeviceToHost);

	float *ret = new float[num_stl1];
	for (int i=0, k=0; i<num_stl1; ++i) {
		ret[i] = mat[k];
		++k;
		for (int j=1; j<group_size; ++j, ++k) {
			if (mat[k]<ret[i]) {
				ret[i] = mat[k];
			}
		}
	}

	cudaFree(stl1_d);
	cudaFree(stl2_d);
	cudaFree(point1_d);
	cudaFree(points2_d);
	cudaFree(mat_d);
	delete[] mat;

	return ret;
}

extern "C"
void computeClosestPointDistanceWithHist_h(Streamline* stls, vec3f* points, const int& num_stls,
								   const float& bin_width, const int& num_bin, 
								   float* avg, float* minv, float* maxv, short* hist)
{
	int num_points = stls[num_stls-1].start+stls[num_stls-1].numPoint;

	//allocate memory to store depth
	Streamline *stls_d;
	vec3f *points_d;
	checkCudaErrors(cudaMalloc((void**)&stls_d, sizeof(Streamline)*num_stls));
	checkCudaErrors(cudaMalloc((void**)&points_d, sizeof(vec3f)*num_points));
	checkCudaErrors(cudaMemcpy(stls_d, stls, sizeof(Streamline)*num_stls, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(points_d, points, sizeof(vec3f)*num_points, cudaMemcpyHostToDevice));

	float *avg_d, *minv_d, *maxv_d;
	short *hist_d;
	checkCudaErrors(cudaMalloc((void**)&avg_d, sizeof(float)*num_stls*num_stls));
	checkCudaErrors(cudaMalloc((void**)&minv_d, sizeof(float)*num_stls*num_stls));
	checkCudaErrors(cudaMalloc((void**)&maxv_d, sizeof(float)*num_stls*num_stls));
	checkCudaErrors(cudaMalloc((void**)&hist_d, sizeof(short)*num_stls*num_stls*num_bin));

	float one_step_percentage = (100.0f*(BLOCK_WIDTH*GRID_WIDTH))/num_stls, percentage=0.0f;
	checkCudaErrors(cudaThreadSynchronize());
	dim3 threadsPerBlock(BLOCK_WIDTH,BLOCK_HEIGHT);
	dim3 numBlocks(GRID_WIDTH,iDivUp(num_stls,BLOCK_HEIGHT));
	for (int i=0; i<num_stls; i+=BLOCK_WIDTH*GRID_WIDTH) {
		computeClosestPointDistance_d<<<numBlocks,threadsPerBlock>>>(stls_d, points_d, num_stls, i, BLOCK_WIDTH*GRID_WIDTH, 
			bin_width, num_bin, avg_d, minv_d, maxv_d, hist_d);
		checkCudaErrors(cudaThreadSynchronize());

		percentage += one_step_percentage;
		if (percentage<99.999999f) {
			printf("\rDistance Matrix Computation: %6.4f%%", percentage);
		} else {
			printf("\rDistance Matrix Computation: 100.000%%\n");
		}
	}

	checkCudaErrors(cudaMemcpy(avg, avg_d, sizeof(float)*num_stls*num_stls, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(minv, minv_d, sizeof(float)*num_stls*num_stls, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(maxv, maxv_d, sizeof(float)*num_stls*num_stls, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hist, hist_d, sizeof(short)*num_stls*num_stls*num_bin, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(avg_d));
	checkCudaErrors(cudaFree(minv_d));
	checkCudaErrors(cudaFree(maxv_d));
}

extern "C"
float* genClosestPointDistanceMatrixInSmallPools_h(Streamline* stls, vec3f* points, const int& num_stls)
{
	int num_points = stls[num_stls-1].start+stls[num_stls-1].numPoint;

	//allocate memory to store depth
	Streamline *stls_d;
	vec3f *points_d;
	checkCudaErrors(cudaMalloc((void**)&stls_d, sizeof(Streamline)*num_stls));
	checkCudaErrors(cudaMalloc((void**)&points_d, sizeof(vec3f)*num_points));
	checkCudaErrors(cudaMemcpy(stls_d, stls, sizeof(Streamline)*num_stls, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(points_d, points, sizeof(vec3f)*num_points, cudaMemcpyHostToDevice));

	float *ret_d;
	checkCudaErrors(cudaMalloc((void**)&ret_d, sizeof(float)*num_stls*num_stls));

	dim3 threadsPerBlock(BLOCK_WIDTH,BLOCK_HEIGHT);
	dim3 numBlocks(iDivUp(num_stls,BLOCK_WIDTH),iDivUp(num_stls,BLOCK_HEIGHT));

	float *ret_h = new float[num_stls*num_stls];
	computeClosestPointDistanceMatrixInSmallPools_d<<<numBlocks,threadsPerBlock>>>(stls_d, points_d, num_stls, ret_d);
	checkCudaErrors(cudaThreadSynchronize());

	checkCudaErrors(cudaMemcpy(ret_h, ret_d, sizeof(float)*num_stls*num_stls, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(ret_d));

	return ret_h;
}

extern "C"
float* genClosestPointDistanceFromLineToSmallPools_h(Streamline* stls, vec3f* points, const int& num_stls, vec3f* line_points, const int& line_point_num)
{
	int num_points = stls[num_stls-1].start+stls[num_stls-1].numPoint;

	//allocate memory to store depth
	Streamline *stls_d;
	vec3f *points_d, *line_points_d;
	checkCudaErrors(cudaMalloc((void**)&stls_d, sizeof(Streamline)*num_stls));
	checkCudaErrors(cudaMalloc((void**)&points_d, sizeof(vec3f)*num_points));
	checkCudaErrors(cudaMalloc((void**)&line_points_d, sizeof(vec3f)*line_point_num));
	checkCudaErrors(cudaMemcpy(stls_d, stls, sizeof(Streamline)*num_stls, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(points_d, points, sizeof(vec3f)*num_points, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(line_points_d, line_points, sizeof(vec3f)*line_point_num, cudaMemcpyHostToDevice));

	float *ret_d;
	checkCudaErrors(cudaMalloc((void**)&ret_d, sizeof(float)*num_stls));

	dim3 threadsPerBlock(BLOCK_WIDTH,BLOCK_HEIGHT);
	dim3 numBlocks(iDivUp(num_stls,BLOCK_WIDTH),iDivUp(num_stls,BLOCK_HEIGHT));

	float *ret_h = new float[num_stls*num_stls];
	computeClosestPointDistanceFromLineToSmallPools_d<<<iDivUp(num_stls,32),32>>>(stls_d, points_d, num_stls, line_points_d, line_point_num, ret_d);
	checkCudaErrors(cudaThreadSynchronize());

	checkCudaErrors(cudaMemcpy(ret_h, ret_d, sizeof(float)*num_stls, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(ret_d));

	return ret_h;
}

extern "C"
float computeMCPOneWay_h(vec3f* from, vec3f* to, int from_num, int to_num){
	vec3f *from_d, *to_d;
	checkCudaErrors(cudaMalloc((void**)&from_d, sizeof(vec3f)*from_num));
	checkCudaErrors(cudaMalloc((void**)&to_d, sizeof(vec3f)*to_num));
	checkCudaErrors(cudaMemcpy(from_d, from, sizeof(vec3f)*from_num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(to_d, to, sizeof(vec3f)*to_num, cudaMemcpyHostToDevice));

	float *ret_d;
	checkCudaErrors(cudaMalloc((void**)&ret_d, sizeof(float)*from_num));

	float *ret_h = new float[from_num];
	computeMCPOneWay_d<<<iDivUp(from_num,64),64>>>(from_d, to_d, from_num, to_num, ret_d);
	checkCudaErrors(cudaThreadSynchronize());

	checkCudaErrors(cudaMemcpy(ret_h, ret_d, sizeof(float)*from_num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(ret_d));

	float ret = 0.0f;
	for (int i=0; i<from_num; ++i) {
		ret += ret_h[i];
	}
	ret /= (float)from_num;

	delete[] ret_h;
	return ret;
}

extern "C"
float computeMCPOneWayWithDeviceMemoryInput_h(vec4f* from_d, vec4f* to_d, int from_num, int to_num){
	float *ret_d;
	checkCudaErrors(cudaMalloc((void**)&ret_d, sizeof(float)*from_num));

	float *ret_h = new float[from_num];
	computeMCPOneWay_d<<<iDivUp(from_num,64),64>>>(from_d, to_d, from_num, to_num, ret_d);
	checkCudaErrors(cudaThreadSynchronize());

	checkCudaErrors(cudaMemcpy(ret_h, ret_d, sizeof(float)*from_num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(ret_d));

	float ret = 0.0f;
	for (int i=0; i<from_num; ++i) {
		ret += ret_h[i];
	}
	ret /= (float)from_num;

	delete[] ret_h;
	return ret;
}