#ifndef CUDA_SELECTION_CU
#define CUDA_SELECTION_CU

#include "typeOperation.h"
#include "cudaSelection.h"
#include <vector>
#include <algorithm> 

__global__ void getNDCPos(vec3f *ret, vec3f *points, float mat[16], int size){
	int idx = blockDim.x*blockIdx.x+threadIdx.x;
	__shared__ float smat[16];
	if(threadIdx.x<16){
		smat[threadIdx.x] = mat[threadIdx.x];
	}
	__syncthreads();

	if(idx<size){
		ret[idx] = getNDCPos(points[idx], smat);
	}
}

//******************************************************
// for streamlines
//******************************************************

__device__ float distToLineSegProj(vec2f p, vec3f A, vec3f B){//ignore z value for A and B
	float t = ((p.x-A.x)*(B.x-A.x)+(p.y-A.y)*(B.y-A.y))/(sqrt((A.x-B.x)*(A.x-B.x)+(A.y-B.y)*(A.y-B.y)));
	vec2f p2;
	if(t<=0.0f){
		p2.x = A.x;
		p2.y = A.y;
	} else if(t>=1.0f){
		p2.x = B.x;
		p2.y = B.y;
	} else {
		p2.x = (1-t)*A.x+t*B.x;
		p2.y = (1-t)*A.y+t*B.y;
	}
	return dist2d(p, p2);
}

__global__ void distToStreamline(vec2f p, Streamline *pool, vec3f *points, int numStl, float *dists, float *zvals, int* pids){
	int idx = blockDim.x*blockIdx.x+threadIdx.x;
	if(idx<numStl){
		int start = pool[idx].start;
		int end = start+pool[idx].numPoint;
		float minDist = 1e30, minz = 1e30, dist, z;
		int min_id = -1;
		for(int i=start+1; i<end-2; i++){
			dist=distToLineSegProj(p, points[i], points[i+1]);
			z = points[i+1].z;
			if((dist<0.01f && z<minz) || (minDist>=0.01f && dist<minDist)){
				minDist = dist;
				minz = z;
				min_id = i;
			}
		}
		dists[idx] = minDist;
		zvals[idx] = minz;
		pids[idx] = min_id;
	}
}

extern "C" 
StreamlineClickInfo computeClickStreamline_h(vec2f p, vec3f* points_d, Streamline* stls_d, int num_point, int num_stl, bool* stl_mark_h, float mat[16]){
	vec3f *proj_points_d;
	cudaMalloc((void**)&proj_points_d, sizeof(vec3f)*num_point);
	float *mat_d;
	cudaMalloc((void**)&mat_d, sizeof(float)*16);
	cudaMemcpy(mat_d, mat, sizeof(float)*16, cudaMemcpyHostToDevice);
	getNDCPos<<<ceilf(num_point/256.0f),256>>>(proj_points_d, points_d, mat_d, num_point);
	cudaFree(mat_d);

	float *dists_d;
	cudaMalloc((void**)&dists_d, sizeof(float)*num_stl);
	float *zvals_d;
	cudaMalloc((void**)&zvals_d, sizeof(float)*num_stl);
	int *pids_d;
	cudaMalloc((void**)&pids_d, sizeof(int)*num_stl);

	distToStreamline<<<ceilf(num_stl/256.0f),256>>>(p, stls_d, proj_points_d, num_stl, dists_d, zvals_d, pids_d);
	float *dists = new float[num_stl];
	float *zvals = new float[num_stl];
	int *pids = new int[num_stl];
	cudaMemcpy(dists, dists_d, sizeof(float)*num_stl, cudaMemcpyDeviceToHost);
	cudaMemcpy(zvals, zvals_d, sizeof(float)*num_stl, cudaMemcpyDeviceToHost);
	cudaMemcpy(pids, pids_d, sizeof(int)*num_stl, cudaMemcpyDeviceToHost);

	int sid = 0, pid = pids[0];
	float minDist = dists[0], minz = zvals[0];
	for(int i=1; i<num_stl; i++) if (stl_mark_h[i]){
		if(pids[i]>=0 && ((dists[i]<0.01f && zvals[i]<minz) || (minDist>=0.01f && dists[i]<minDist))){
			minDist = dists[i];
			minz = zvals[i];
			sid = i;
			pid = pids[i];
		}
	}

	if(minDist>0.1f){
		sid = -1;
		pid = -1;
	}

	//clean
	delete[] dists;
	delete[] zvals;
	delete[] pids;
	cudaFree(proj_points_d);
	cudaFree(dists_d);
	cudaFree(zvals_d);
	cudaFree(pids_d);

	StreamlineClickInfo ret = {sid, pid};
	return ret;
}


//******************************************************
// for grids
//******************************************************

__global__ void testBlockClicked(float *ret_z, vec2f p, vec3i grid_dim, vec3f block_size, float mat[16]){
	int bid = blockDim.x*blockIdx.x+threadIdx.x;

	if(bid<grid_dim.x*grid_dim.y*grid_dim.z){
		vec3f pos[8];
		pos[0].x = (bid % grid_dim.x)*block_size.x;
		pos[0].y = ((bid/grid_dim.x)%grid_dim.y)*block_size.y;
		pos[0].z = (bid/(grid_dim.x*grid_dim.y))*block_size.z;
		pos[1] = pos[0] + makeVec3f(block_size.x, 0.0f, 0.0f);
		pos[2] = pos[0] + makeVec3f(0.0f, block_size.y, 0.0f);
		pos[3] = pos[0] + makeVec3f(block_size.x, block_size.y, 0.0f);
		pos[4] = pos[0] + makeVec3f(0.0f, 0.0f, block_size.z);
		pos[5] = pos[1] + makeVec3f(0.0f, 0.0f, block_size.z);
		pos[6] = pos[2] + makeVec3f(0.0f, 0.0f, block_size.z);
		pos[7] = pos[3] + makeVec3f(0.0f, 0.0f, block_size.z);

		float z = 0.0f;
		for (int i=0; i<8; ++i) {
			pos[i] = getNDCPos(pos[i], mat);
			z += pos[i].z;
		}

		if(testPointInTriangle(pos[0], pos[1], pos[2], p)||
			testPointInTriangle(pos[3], pos[1], pos[2], p)||
			testPointInTriangle(pos[1], pos[3], pos[5], p)||
			testPointInTriangle(pos[7], pos[3], pos[5], p)||
			testPointInTriangle(pos[2], pos[3], pos[6], p)||
			testPointInTriangle(pos[7], pos[3], pos[6], p)||
			testPointInTriangle(pos[0], pos[1], pos[4], p)||
			testPointInTriangle(pos[5], pos[1], pos[4], p)||
			testPointInTriangle(pos[0], pos[4], pos[2], p)||
			testPointInTriangle(pos[6], pos[4], pos[2], p)||
			testPointInTriangle(pos[4], pos[5], pos[6], p)||
			testPointInTriangle(pos[7], pos[5], pos[6], p)){
				ret_z[bid]=z;
		} else {
			ret_z[bid]=QUAD_NOT_SELECTED;
		}
	}
}

extern "C" 
void computeClickBlockDepth_h(float* ret_depth, vec2f p, vec3i grid_dim, vec3f block_size, float mat[16]){
	float *mat_d;
	checkCudaErrors(cudaMalloc((void**)&mat_d, sizeof(float)*16));
	checkCudaErrors(cudaMemcpy(mat_d, mat, sizeof(float)*16, cudaMemcpyHostToDevice));
	
	int total_grid = grid_dim.x*grid_dim.y*grid_dim.z;
	float *block_depth_d;
	checkCudaErrors(cudaMalloc((void**)&block_depth_d, sizeof(float)*total_grid));
	testBlockClicked<<<iDivUp(total_grid,32),32>>>(block_depth_d, p, grid_dim, block_size, mat_d);
	checkCudaErrors(cudaMemcpy(ret_depth, block_depth_d, sizeof(float)*total_grid, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(mat_d));
	checkCudaErrors(cudaFree(block_depth_d));
}

//******************************************************
// for quads
//******************************************************

__global__ void testQuadClicked(float *ret_z, vec2f p, vec4f* v, vec4i* quads, int num_quads, float mat[16]){
	int qid = blockDim.x*blockIdx.x+threadIdx.x;
      
	if(qid<num_quads){
		vec4i q = quads[qid];
		vec3f pos[4];
		pos[0] = v[q.x].xyz;
		pos[1] = v[q.y].xyz;
		pos[2] = v[q.z].xyz;
		pos[3] = v[q.w].xyz;

		float z = 0.0f;
		for (int i=0; i<4; ++i) {
			pos[i] = getNDCPos(pos[i], mat);
			z += pos[i].z;
		}

		if(testPointInTriangle(pos[0], pos[1], pos[2], p)||
			testPointInTriangle(pos[3], pos[0], pos[2], p)){
				ret_z[qid]=z;
		} else {
			ret_z[qid]=QUAD_NOT_SELECTED;
		}
	}
}

__global__ void testTriangleClicked(float *ret_z, vec2f p, vec4f* v, vec3i* tris, int num_tris, float mat[16]){
	int tid = blockDim.x*blockIdx.x+threadIdx.x;

	if(tid<num_tris){
		vec3i t = tris[tid];
		vec3f pos[3];
		pos[0] = v[t.x].xyz;
		pos[1] = v[t.y].xyz;
		pos[2] = v[t.z].xyz;

		float z = 0.0f;
		for (int i=0; i<3; ++i) {
			pos[i] = getNDCPos(pos[i], mat);
			z += pos[i].z;
		}

		if(testPointInTriangle(pos[0], pos[1], pos[2], p)){
				ret_z[tid] = z;
		} else {
			ret_z[tid] = TRIANGLE_NOT_SELECTED;
		}
	}
}

extern "C" 
void computeAllClickQuadDepth_h(float* ret, vec4f* v_d, vec4i* quads_d, const int& num_quads, const vec2f& p, float mat[16]){
	float *mat_d;
	checkCudaErrors(cudaMalloc((void**)&mat_d, sizeof(float)*16));
	checkCudaErrors(cudaMemcpy(mat_d, mat, sizeof(float)*16, cudaMemcpyHostToDevice));

	float *quad_depth_d;
	checkCudaErrors(cudaMalloc((void**)&quad_depth_d, sizeof(float)*num_quads));
	testQuadClicked<<<iDivUp(num_quads,256),256>>>(quad_depth_d, p, v_d, quads_d, num_quads, mat_d);
	checkCudaErrors(cudaMemcpy(ret, quad_depth_d, sizeof(float)*num_quads, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(mat_d));
	checkCudaErrors(cudaFree(quad_depth_d));
}

extern "C" 
int computeClickQuad_h(vec4f* v_d, vec4i* quads_d, const int& num_quads, const vec2f& p, float mat[16]){
	float *quad_depth = new float[num_quads];
	computeAllClickQuadDepth_h(quad_depth, v_d, quads_d, num_quads, p, mat);
	
	int ret_qid = -1;
	float min_z = QUAD_NOT_SELECTED;
	for (int i=0; i<num_quads; ++i) {
		if (quad_depth[i]<min_z) {
			min_z = quad_depth[i];
			ret_qid = i;
		}
	}
	
	delete[] quad_depth;

	return ret_qid;
}

extern "C" 
void computeAllClickTriangleDepth_h(float* ret, vec4f* v_d, vec3i* tris_d, const int& num_tris, const vec2f& p, float mat[16]){
	float *mat_d;
	checkCudaErrors(cudaMalloc((void**)&mat_d, sizeof(float)*16));
	checkCudaErrors(cudaMemcpy(mat_d, mat, sizeof(float)*16, cudaMemcpyHostToDevice));

	float *tri_depth_d;
	checkCudaErrors(cudaMalloc((void**)&tri_depth_d, sizeof(float)*num_tris));
	testTriangleClicked<<<iDivUp(num_tris,256),256>>>(tri_depth_d, p, v_d, tris_d, num_tris, mat_d);
	checkCudaErrors(cudaMemcpy(ret, tri_depth_d, sizeof(float)*num_tris, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(mat_d));
	checkCudaErrors(cudaFree(tri_depth_d));
}

extern "C" 
int computeClickTriangle_h(vec4f* v_d, vec3i* tris_d, const int& num_tris, const vec2f& p, float mat[16]){
	float *tri_depth = new float[num_tris];
	computeAllClickTriangleDepth_h(tri_depth, v_d, tris_d, num_tris, p, mat);

	int ret_qid = -1;
	float min_z = TRIANGLE_NOT_SELECTED;
	for (int i=0; i<num_tris; ++i) {
		if (tri_depth[i]<min_z) {
			min_z = tri_depth[i];
			ret_qid = i;
		}
	}

	delete[] tri_depth;

	return ret_qid;
}

//*********************************************
//          map triangles to 2d
//*********************************************

inline __device__ bool in_screen(const vec2f& p) {
	if (p.x<-1.0f || p.x>1.0f || p.y<-1.0f || p.y>1.0f) {
		return false;
	}
	return true;
}

__global__ void mapTriangleArea_d(float* ret_area, vec3f* vertices, vec3i* triangles, float mat[16],
	int num_per_thread, int num_triangles)
{
	int block_total = blockDim.x*num_per_thread;
	int start = blockIdx.x*block_total + threadIdx.x;
	int end = start + block_total;
	if (end > num_triangles) end = num_triangles;

	__shared__ float smat[16];
	if (threadIdx.x < 16) {
		smat[threadIdx.x] = mat[threadIdx.x];
	}
	__syncthreads();

	vec3i idx;
	vec2f p1, p2, p3;
	vec3f v12, v13, n;
	v12.z = v13.z = 0.0f;

	for (int i = start; i < end; i += blockDim.x) {
		idx = triangles[i];

		p1 = getNDCPosXY(vertices[idx.x], smat);
		p2 = getNDCPosXY(vertices[idx.y], smat);
		p3 = getNDCPosXY(vertices[idx.z], smat);

		if ((in_screen(p1) || in_screen(p2) || in_screen(p3))) {
			v12.xy = p2 - p1;
			v13.xy = p3 - p1;
			n = cross(v12, v13);
			ret_area[i] = (n.z < 0.0f) ? sqrtf(n*n) : -sqrtf(n*n);
		} else {
			ret_area[i] = 0.0f;
		}
	}
}

extern "C"
__host__ void mapTriangleArea_h(float* ret_area_d, vec3f* vertices_d, vec3i* triangles_d, float mat[16],
	int num_per_thread, int num_triangles)
{
	float* mat_d;
	cudaMalloc(&mat_d, sizeof(float) * 16);
	cudaMemcpy(mat_d, mat, sizeof(float) * 16, cudaMemcpyHostToDevice);

	int block_size, min_grid_size, grid_size;
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, mapTriangleArea_d, 0, 0);
	grid_size = iDivUp(num_triangles, num_per_thread*block_size);
	mapTriangleArea_d << < grid_size, block_size >> > (ret_area_d, vertices_d, triangles_d, mat_d, num_per_thread, num_triangles);

	cudaFree(mat_d);
}

//*********************************************
//          map quads to 2d
//*********************************************

__global__ void mapQuadArea_d(float* ret_area, vec3f* vertices, vec4i* quads, float mat[16],
	int num_per_thread, int num_quads)
{
	int block_total = blockDim.x*num_per_thread;
	int start = blockIdx.x*block_total + threadIdx.x;
	int end = start + block_total;
	if (end > num_quads) end = num_quads;

	__shared__ float smat[16];
	if (threadIdx.x < 16) {
		smat[threadIdx.x] = mat[threadIdx.x];
	}
	__syncthreads();

	vec4i idx;
	vec2f p1, p2, p3, p4;
	vec3f v12, v13, v14, n;
	v12.z = v13.z = v14.z = 0.0f;
	float ret;

	for (int i = start; i < end; i += blockDim.x) {
		idx = quads[i];

		p1 = getNDCPosXY(vertices[idx.x], smat);
		p2 = getNDCPosXY(vertices[idx.y], smat);
		p3 = getNDCPosXY(vertices[idx.z], smat);
		p4 = getNDCPosXY(vertices[idx.w], smat);

		if ((in_screen(p1) || in_screen(p2) || in_screen(p3) || in_screen(p4))) {
			v12.xy = p2 - p1;
			v13.xy = p3 - p1;
			v14.xy = p4 - p1;

			n = cross(v12, v13);
			ret = (n.z < 0.0f) ? sqrtf(n*n) : -sqrtf(n*n);
			n = cross(v13, v14);
			ret += (n.z < 0.0f) ? sqrtf(n*n) : -sqrtf(n*n);
			ret_area[i] = ret;
		} else {
			ret_area[i] = 0.0f;
		}
	}
}

extern "C"
__host__ void mapQuadArea_h(float* ret_area_d, vec3f* vertices_d, vec4i* quads_d, float mat[16],
	int num_per_thread, int num_quads)
{
	float* mat_d;
	cudaMalloc(&mat_d, sizeof(float) * 16);
	cudaMemcpy(mat_d, mat, sizeof(float) * 16, cudaMemcpyHostToDevice);

	int block_size, min_grid_size, grid_size;
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, mapQuadArea_d, 0, 0);
	grid_size = iDivUp(num_quads, num_per_thread*block_size);
	mapQuadArea_d << < grid_size, block_size >> > (ret_area_d, vertices_d, quads_d, mat_d, num_per_thread, num_quads);

	cudaFree(mat_d);
}

#endif //CUDA_SELECTION_CU