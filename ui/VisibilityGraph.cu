#ifndef VISIBILITY_GRAPH_CU
#define VISIBILITY_GRAPH_CU

#include "typeOperation.h"
#include "helper_cuda.h"
#include "VisibilityGraph.h"
#include "cudaDeviceMem.h"
#include <cuda_occupancy.h>

#define FLOYD_WARSHALL_KERNEL(n) iDivUp(n, 16*CUDA_WARP_SIZE), 16*CUDA_WARP_SIZE //each warp handles a map
#define FILTER_MAP_KERNEL(n) iDivUp(n, 32*CUDA_WARP_SIZE), 32*CUDA_WARP_SIZE //each warp handles a map
#define FIND_CLOSEST_KERNEL(n,m) iDivUp(n, 1024*m), 1024
#define COMPUTE_DIST_MAT_KERNEL(n) iDivUp(n, 512), 512
#define COMPUTE_PERPLEXITY_KERNEL(n) iDivUp(n, 512), 512
#define FIND_DIST_TO_PATTERN_KERNEL(n) iDivUp(n, 512), 512

__global__ void computeVisibiltyGraphs_d(float* ret, VisGraphCompTask* tasks, vec3f* points, 
	int num_per_thread, int num_tasks)
{
	int block_total = blockDim.x*num_per_thread;
	int start = blockIdx.x*block_total + threadIdx.x;
	int end = start + block_total;
	if (end > num_tasks) end = num_tasks;

	vec3f u, v;
	float d;
	VisGraphCompTask t;
	for (int i = start; i < end; i += blockDim.x) {
		t = tasks[i];
		//当前点
		u = points[t.uid];
		//第j个点
		v = points[t.vid];
		//计算两点的距离
		d = length(u - v);
		ret[t.dist_loc] = d;
	}
}

extern "C" 
__host__ void computeVisibiltyGraphs_h(float* ret, VisGraphCompTask* tasks, vec3f* points_d,
	int ret_mat_size, int num_per_thread, int num_tasks)
{
	float* ret_d;
	checkCudaErrors(cudaMalloc(&ret_d, sizeof(float)*ret_mat_size));

	VisGraphCompTask* tasks_d;
	checkCudaErrors(cudaMalloc(&tasks_d, sizeof(VisGraphCompTask)*num_tasks));
	checkCudaErrors(cudaMemcpy(tasks_d, tasks, sizeof(VisGraphCompTask)*num_tasks, cudaMemcpyHostToDevice));

	int kernel_block_size, min_grid_size, grid_size;
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &kernel_block_size, computeVisibiltyGraphs_d, 0, 0);
	grid_size = iDivUp(num_tasks, num_per_thread*kernel_block_size);

	computeVisibiltyGraphs_d<<<grid_size, kernel_block_size>>>(ret_d, tasks_d, points_d, num_per_thread, num_tasks);
	checkCudaErrors(cudaMemcpy(ret, ret_d, sizeof(float)*ret_mat_size, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(ret_d));
	checkCudaErrors(cudaFree(tasks_d));
}

//visibility graph for curves with a small fixed length
__global__ void computeVisibilityGraphFixedLength_d(float *ret, VisGraphCompTask* tasks, vec3f* points, 
	int num_point_per_line, int num_tasks_per_graph, int num_per_thread, int num_lines) 
{
	int block_total = blockDim.x*num_per_thread;
	int start = block_total*blockIdx.x;
	int end = min(start + block_total, num_tasks_per_graph*num_lines);

	//extern __shared__ VisGraphCompTask s_tasks[];
	//if (threadIdx.x < num_tasks_per_graph) {
	//	s_tasks[threadIdx.x] = tasks[threadIdx.x];
	//}
	//__syncthreads();

	VisGraphCompTask t;
	int mat_size = num_point_per_line*num_point_per_line;
	int task_id, line_id, point_offset;
	vec3f u, v;
	for (int i = start; i < end; ++i) {
		line_id = i / num_tasks_per_graph;
		task_id = i % num_tasks_per_graph;
		t = tasks[task_id];
		point_offset = line_id*num_point_per_line;
		u = points[t.uid + point_offset];
		v = points[t.vid + point_offset];
		ret[mat_size*line_id+t.dist_loc] = length(u - v);
	}
}

extern "C"
void computeVisiblityGraphFixedLength_h(float *ret, const vec3f* points,
	int num_point_per_line, int num_per_thread, int num_lines)
{
	float* ret_d;
	int ret_mat_size = sizeof(float)*num_point_per_line*num_point_per_line*num_lines;
	checkCudaErrors(cudaMalloc(&ret_d, ret_mat_size));

	int num_tasks_per_graph = num_point_per_line*num_point_per_line;// (num_point_per_line - 1)*num_point_per_line / 2;
	int num_total_tasks = num_tasks_per_graph*num_lines;
	std::vector<VisGraphCompTask> tasks;
	VisGraphCompTask t;
	tasks.reserve(num_tasks_per_graph);
	for (int i = 0; i < num_point_per_line; ++i) {
		t.uid = i;
		t.dist_loc = i*num_point_per_line;
		for (int j =0 ; j < num_point_per_line; ++j, ++t.dist_loc) {
			t.vid = j;
			tasks.push_back(t);
		}
	}
	VisGraphCompTask* tasks_d;
	checkCudaErrors(cudaMalloc(&tasks_d, sizeof(VisGraphCompTask)*num_tasks_per_graph));
	checkCudaErrors(cudaMemcpy(tasks_d, tasks.data(), sizeof(VisGraphCompTask)*num_tasks_per_graph, cudaMemcpyHostToDevice));

	vec3f* points_d;
	checkCudaErrors(cudaMalloc(&points_d, sizeof(vec3f)*num_point_per_line*num_lines));
	checkCudaErrors(cudaMemcpy(points_d, points, sizeof(vec3f)*num_point_per_line*num_lines, cudaMemcpyHostToDevice));


	int kernel_block_size, min_grid_size, grid_size;
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &kernel_block_size, computeVisibilityGraphFixedLength_d, 0, 0);
	grid_size = iDivUp(num_total_tasks, num_per_thread*kernel_block_size);

	computeVisibilityGraphFixedLength_d <<<grid_size, kernel_block_size>>> (ret_d, tasks_d, points_d, 
		num_point_per_line, num_tasks_per_graph, num_per_thread, num_lines);
	checkCudaErrors(cudaMemcpy(ret, ret_d, ret_mat_size, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(ret_d));
	checkCudaErrors(cudaFree(tasks_d));
	checkCudaErrors(cudaFree(points_d));
}

//compute degrees upto a certain distance
__global__ void computeDegreeMap_d(float* ret, float* dist_mats, int* matrix_offsets,
	int* line_num_points, int total_num_points, float* dist_threshes, int num_dist_thresh)
{
	int num_point_per_block = blockDim.x / num_dist_thresh;
	int in_block_point_id = threadIdx.x / num_dist_thresh;
	int point_id = blockIdx.x*num_point_per_block + in_block_point_id;

	if (point_id >= total_num_points) return;

	float* dist_mat = dist_mats+matrix_offsets[point_id];

	int dist_thresh_id = threadIdx.x % num_dist_thresh;
	float d_thresh = dist_threshes[dist_thresh_id];

	float d;
	int n = line_num_points[point_id], count = 0;
	for (int i = 0; i < n; ++i) {
		d = dist_mat[i];
		if (d < d_thresh) {
			++count;
		}
	}
	ret[point_id*num_dist_thresh + dist_thresh_id] = count*0.01f;
}

//adjust degree so that it represent the degree between two distance thresholds
__global__ void computeDegreeMapPerRange_d(float* degrees, int num_degree_per_point, int num_per_thread, int num_points)
{
	int block_total = blockDim.x*num_per_thread;
	int start = blockIdx.x*block_total + threadIdx.x;
	int end = min(start + block_total, num_points*num_degree_per_point);

	if (threadIdx.x%num_degree_per_point == 0) return;

	volatile int prev_degree;
	for (int i = start; i < end; i += blockDim.x) {
		prev_degree = degrees[i - 1];
		__syncthreads();
		degrees[i] -= prev_degree;
	}
}

extern "C"
__host__ void computeDegreeMap_h(float* ret_d, float* dist_mats, const int& dist_mat_size, int* matrix_offsets,
	int* line_num_points, const int& total_num_points, float* dist_threshes, const int& num_dist_thresh, const bool& b_acc)
{
	int *matrix_offsets_d, *line_num_points_d; 
	float *dist_mats_d, *dist_threshes_d;
	
	checkCudaErrors(cudaMalloc(&matrix_offsets_d, sizeof(int)*total_num_points));
	checkCudaErrors(cudaMemcpy(matrix_offsets_d, matrix_offsets, sizeof(int)*total_num_points, cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMalloc(&line_num_points_d, sizeof(int)*total_num_points));
	checkCudaErrors(cudaMemcpy(line_num_points_d, line_num_points, sizeof(int)*total_num_points, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&dist_mats_d, sizeof(float)*dist_mat_size));
	checkCudaErrors(cudaMemcpy(dist_mats_d, dist_mats, sizeof(float)*dist_mat_size, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&dist_threshes_d, sizeof(float)*num_dist_thresh));
	checkCudaErrors(cudaMemcpy(dist_threshes_d, dist_threshes, sizeof(float)*num_dist_thresh, cudaMemcpyHostToDevice));

	int kernel_block_size, min_grid_size, grid_size;
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &kernel_block_size, computeVisibiltyGraphs_d, 0, 0);
	kernel_block_size = (kernel_block_size / num_dist_thresh)*num_dist_thresh;
	grid_size = iDivUp(total_num_points, kernel_block_size/num_dist_thresh);

	computeDegreeMap_d <<< grid_size, kernel_block_size >>> (ret_d, dist_mats_d, matrix_offsets_d, line_num_points_d,
		total_num_points, dist_threshes_d, num_dist_thresh);

	if (!b_acc){
		computeDegreeMapPerRange_d<<< grid_size, kernel_block_size>>>(ret_d, num_dist_thresh, 32, total_num_points);
	}

	checkCudaErrors(cudaFree(dist_mats_d));
	checkCudaErrors(cudaFree(dist_threshes_d));
	checkCudaErrors(cudaFree(matrix_offsets_d));
	checkCudaErrors(cudaFree(line_num_points_d));
}

__global__ void matchPattern_d(bool *ret, float* match_pattern, int pattern_len, int pattern_offset, float match_thresh,
	float* feature_map, int feature_dim, int num_per_thread, int num_points)
{
	int block_total = blockDim.x*num_per_thread;
	int start = blockIdx.x*block_total + threadIdx.x;
	int end = start + block_total;
	if (end > num_points) end = num_points;

	//extern __shared__ float s_pattern[];
	//float* s_weight = (float*)&s_pattern[pattern_len];
	//if (threadIdx.x<pattern_len) {
	//	s_pattern[threadIdx.x] = match_pattern[threadIdx.x];
	//	s_weight[threadIdx.x] = pattern_weights[threadIdx.x];
	//}
	//__syncthreads();

	float thresh_square = match_thresh*match_thresh;
	float d, dj;
	float* point_pattern;
	for (int i = start; i < end; i += blockDim.x) {
		point_pattern = &feature_map[i*feature_dim + pattern_offset];
		d = 0.0f;
		for (int j = 0; j < pattern_len; ++j) {
			dj = (match_pattern[j]-point_pattern[j]);
			d += (dj*dj);
		}
		if (d < thresh_square) {
			ret[i] = true;
		}
	}
}

extern "C"
__host__ void matchPattern_h(bool *ret, float* match_pattern_d, int pattern_len, int pattern_offset, float match_thresh,
	float* feature_map_d, int feature_dim, int num_per_thread, int num_points)
{
	bool* ret_d;
	checkCudaErrors(cudaMalloc(&ret_d, sizeof(bool)*num_points));
	checkCudaErrors(cudaMemset(ret_d, 0, sizeof(bool)*num_points));

	int kernel_block_size, min_grid_size, grid_size;
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &kernel_block_size, matchPattern_d, 0, 0);
	grid_size = iDivUp(num_points, kernel_block_size*num_per_thread);

	matchPattern_d <<< grid_size, kernel_block_size/*, pattern_len*2*/>>>(ret_d, match_pattern_d, pattern_len, pattern_offset,
		match_thresh, feature_map_d, feature_dim, num_per_thread, num_points);
	checkCudaErrors(cudaMemcpy(ret, ret_d, sizeof(bool)*num_points, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(ret_d));
}

__global__ void findDistToPattern_d(float *ret_dist, float* match_pattern, int pattern_len, int pattern_offset, 
	float* feature_map, int feature_dim, int num_per_thread, int num_points)
{
	int block_total = blockDim.x*num_per_thread;
	int start = blockIdx.x*block_total + threadIdx.x;
	int end = start + block_total;
	if (end > num_points) end = num_points;

	float d, dj;
	float* point_pattern;
	for (int i = start; i < end; i += blockDim.x) {
		point_pattern = &feature_map[i*feature_dim + pattern_offset];
		d = 0.0f;
		for (int j = 0; j < pattern_len; ++j) {
			dj = (match_pattern[j] - point_pattern[j]);
			d += (dj*dj);
		}
		ret_dist[i] = d;
	}
}

extern "C"
__host__ void findDistToPattern_h(float* ret, float* match_pattern_d, int pattern_len, int pattern_offset, float match_thresh,
	float* feature_map_d, int feature_dim, int num_per_thread, int num_points)
{
	cudaDeviceMem<float> ret_d(num_points);
	findDistToPattern_d <<<FIND_DIST_TO_PATTERN_KERNEL(num_points)>>> (ret_d.data_d, match_pattern_d, pattern_len, pattern_offset, 
		feature_map_d, feature_dim, num_per_thread, num_points);
	cudaDeviceSynchronize();
	ret_d.dump(ret);
}

__global__ void findClosestTemplate_d(int* ret, float* templates, float* data, int num_templates, int num_data, 
	int dim, int num_per_thread) 
{	
	int block_total = blockDim.x*num_per_thread;
	int start = blockIdx.x*block_total + threadIdx.x;
	int end = min(start + block_total, num_data);

	float min_dist, d_sq, d;
	int min_id;
	for (int i = start; i < end; i+= blockDim.x) {
		float* di = data+i*dim;
		min_dist = 1e30;
		min_id = -1;
		for (int j = 0; j < num_templates; ++j) {
			float* tj = templates + j*dim;
			d_sq = 0.0f;
			for (int k = 0; k < dim; ++k) {
				d = di[k] - tj[k];
				d_sq += d*d;
			}
			if (d_sq < min_dist) {
				min_id = j;
				min_dist = d_sq;
			}
		}
		ret[i] = min_id;
	}
}

extern "C"
__host__ void findClosestTemplate_h(int* ret_d, float* templates_d, float* data_d, int num_templates, int num_data,
	int dim, int num_per_thread) 
{
	findClosestTemplate_d <<<FIND_CLOSEST_KERNEL(num_data, num_per_thread) >>> (ret_d, templates_d, data_d, num_templates,
		num_data, dim, num_per_thread);
	cudaThreadSynchronize();
}

//assume blockDim.x/num_cols is integer
__global__ void computeNarrowMatrixColumnSum_d(float* ret, float* matrix, int num_cols, int num_rows, int num_per_thread) 
{
	int block_total = blockDim.x*num_per_thread;
	int start = blockIdx.x*block_total + threadIdx.x;
	int end = min(start + block_total, num_rows*num_cols);

	float sum = 0;
	for (int i = start; i < end; i += blockDim.x) {
		sum += matrix[i];
	}
	atomicAdd(&ret[threadIdx.x%num_cols], sum);
}

__global__ void computeNarrowMatrixColumnTotalVar_d(float* ret, float* matrix, float* col_sums, int num_cols, int num_rows,
	int num_per_thread)
{
	int block_total = blockDim.x*num_per_thread;
	int start = blockIdx.x*block_total + threadIdx.x;
	int end = min(start + block_total, num_rows*num_per_thread);

	float total_var = 0;
	int deg_id = threadIdx.x%num_cols;
	float deg_avg = col_sums[deg_id] / num_rows;
	float diff;
	for (int i = start; i < end; i += blockDim.x) {
		diff = matrix[i] - deg_avg;
		total_var += diff*diff;
	}
	atomicAdd(&ret[deg_id], total_var);
}

extern "C"
__host__ void computeNarrowMatrixColumnAverageVariance_h(float* ret_avg, float* ret_var, float* matrix_d, 
	int num_cols, int num_rows, int num_per_thread)
{
	float* total_d;
	checkCudaErrors(cudaMalloc(&total_d, sizeof(float)*num_cols));
	checkCudaErrors(cudaMemset(total_d, 0, sizeof(float)*num_cols));
	float* total_var_d;
	checkCudaErrors(cudaMalloc(&total_var_d, sizeof(float)*num_cols));
	checkCudaErrors(cudaMemset(total_var_d, 0, sizeof(float)*num_cols));

	int kernel_block_size, min_grid_size, grid_size;
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &kernel_block_size, computeNarrowMatrixColumnTotalVar_d, 0, 0);
	kernel_block_size = (kernel_block_size / num_cols)*num_cols;
	grid_size = iDivUp(num_rows, kernel_block_size*num_per_thread);

	computeNarrowMatrixColumnSum_d <<< grid_size, kernel_block_size >>> (total_d, matrix_d, num_cols, num_rows, num_per_thread);
	cudaThreadSynchronize();
	computeNarrowMatrixColumnTotalVar_d <<< grid_size, kernel_block_size >>> (total_var_d, matrix_d, total_d, 
		num_cols, num_rows, num_per_thread);

	std::vector<float> total_deg(num_cols);
	checkCudaErrors(cudaMemcpy(total_deg.data(), total_d, sizeof(float)*num_cols, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(ret_var, total_var_d, sizeof(float)*num_cols, cudaMemcpyDeviceToHost));

	for (int i = 0; i < num_cols; ++i) {
		ret_avg[i] = total_deg[i] / num_rows;
		ret_var[i] = sqrtf(ret_var[i]/(num_rows-1));
	}

	//checkCudaErrors(cudaMemcpy(ret, ret_d, sizeof(bool)*num_points, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(total_d));
	checkCudaErrors(cudaFree(total_var_d));
}

__global__ void computeMatrixColumnSum_d(float* ret, float* matrix, int num_cols, int num_rows, int num_per_thread)
{
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	if (j > num_rows) return;
	
	float *dij = matrix + j;
	float sum = 0.0f;
	for (int i = 0; i < num_rows; ++i, dij += num_cols) {
		sum += *dij;
	}
	ret[j] = sum;
}

__global__ void computeMatrixColumnVar_d(float* ret, float* matrix, float* col_sums, int num_cols, int num_rows,
	int num_per_thread)
{
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	if (j > num_rows) return;

	float *dij = matrix + j;
	float avg = col_sums[j]/num_rows;
	float diff, total_var = 0.0f;
	for (int i = 0; i < num_rows; ++i, dij += num_cols){
		diff = *dij - avg;
		total_var += diff*diff;
	}
	ret[j] = total_var/(num_rows-1);
}

__global__ void computePerplexity_d(float* matrix, float* var, int n) {
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	if (j > n) return;

	float *dji = matrix + j;
	float dji_val, sum = 0.0f, pji;
	float factor = 0.5f / (var[j]*var[j]);
	for (int i = 0; i < n; ++i, dji += n) {
		dji_val = *dji;
		pji = expf(-dji_val*dji_val*factor);
		sum += pji;
		*dji = factor;
	}
	dji = matrix + j;
	for (int i = 0; i < n; ++i, dji += n) {
		*dji /= sum;
	}
}

__global__ void computeSymmetricMatrix_d(float* matrix, int n) {
	int ii = blockDim.x*blockIdx.x + threadIdx.x;

	if (ii >= (n + 1) / 2) return;//each thread compute 2 columns

	float avg = 0.0f;
	//if idx>j compute ret[j][idx] otherwise compute ret[j][num-1-idx]
	for (int i, j, jj = 0; jj < n; ++jj) if (jj != ii) {
		j = (ii > jj) ? jj : (n - 1 - jj);
		i = (ii > jj) ? ii : (n - 1 - ii);
		avg = 0.5f*(matrix[i*n + j] + matrix[j*n + i]);
		matrix[i*n + j] = matrix[j*n + i] = avg;
	}
}

extern "C"
__host__ void computePerplexity_h(float* matrix_d, int n) {
	cudaDeviceMem<float> total(n);
	cudaDeviceMem<float> var(n);
	computeMatrixColumnSum_d <<< COMPUTE_PERPLEXITY_KERNEL(n) >>> (total.data_d, matrix_d, n, n, 32);
	cudaDeviceSynchronize();
	computeMatrixColumnVar_d <<< COMPUTE_PERPLEXITY_KERNEL(n)>>> (var.data_d, matrix_d, total.data_d, n, n, 32);
	cudaDeviceSynchronize();
	computePerplexity_d<<<COMPUTE_PERPLEXITY_KERNEL(n)>>>(matrix_d, var.data_d, n);
	cudaDeviceSynchronize();
	computeSymmetricMatrix_d <<<COMPUTE_PERPLEXITY_KERNEL(n) >>> (matrix_d, n);
	cudaDeviceSynchronize();
}

__global__ void filterLargeValues_d(float* data, float thresh, float replace_value, int num_per_thread, int num) {
	int block_total = blockDim.x*num_per_thread;
	int start = blockIdx.x*block_total + threadIdx.x;
	int end = min(start + block_total, num);

	for (int i = start; i < end; i+=blockDim.x) {
		if (data[i] > thresh) {
			data[i] = replace_value;
		}
	}
}

extern "C"
__host__ void filterLargeValues_h(float* data_d, const float& thresh, const float& replace_value, const int& num_per_thread,
	const int& num) 
{
	int kernel_block_size, min_grid_size, grid_size;
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &kernel_block_size, filterLargeValues_d, 0, 0);
	grid_size = iDivUp(num, kernel_block_size*num_per_thread);

	filterLargeValues_d <<< grid_size, kernel_block_size >>> (data_d, thresh, replace_value, num_per_thread, num);
	cudaThreadSynchronize();
}

__global__ void filterMapUpperTriangle_d(float* maps, int* offsets, int* sizes, float thresh, float replace_value, int num) {
	int map_per_block = blockDim.x / CUDA_WARP_SIZE;
	int warp_id = threadIdx.x / CUDA_WARP_SIZE;
	int thread_id = threadIdx.x % CUDA_WARP_SIZE;
	int map_id = map_per_block*blockIdx.x + warp_id;
	if (map_id > num) return;

	float* map = maps + offsets[map_id];
	int n = sizes[map_id], ij;
	bool is_valid;

	for (int i = 0; i < n; ++i) {
		is_valid = true;
		for (int j = thread_id; j < i; j+=CUDA_WARP_SIZE) {
			ij = i*n + j;
			if (!is_valid) {
				map[ij] = replace_value;
			} else if (map[ij] > thresh) {
				map[ij] = replace_value;
				is_valid = false;
			}
		}
	}
}

extern "C"
__host__ void filterMapUpperTriangle_h(float* maps_d, int* offsets_d, int* sizes_d, const float& thresh, 
	const float& replace_value, const int& num) 
{
	filterMapUpperTriangle_d <<< FILTER_MAP_KERNEL(num) >>> (maps_d, offsets_d, sizes_d, thresh, replace_value, num);
	cudaThreadSynchronize();
}

__global__ void floydWarshall_d(float* maps, int* offsets, int* sizes, int num) {
	int map_per_block = blockDim.x / CUDA_WARP_SIZE;
	int warp_id = threadIdx.x / CUDA_WARP_SIZE;
	int thread_id = threadIdx.x % CUDA_WARP_SIZE;
	int map_id = map_per_block*blockIdx.x+warp_id;
	if (map_id > num) return;

	float* map = maps+offsets[map_id];
	int n = sizes[map_id];
	float dik, dikj, dij;

	for (int k = 0; k < n; ++k) {
		for (int i = 1; i < n; ++i) if (i!=k) {
			dik = (i>k)?(map[i*n + k]):(map[k*n+i]);
			for (int j = thread_id; j < i; j+=CUDA_WARP_SIZE) if (j!=k) {
				dij = map[i*n + j];
				dikj = dik+ (k>j)?(map[k*n + j]):(map[j*n+k]);
				if (dij > dikj) {
					map[i*n + j] = dikj;
				}
			}
		}
	}

	for (int i = 0; i < n; ++i) {
		for (int j = i + 1+thread_id; j < n; j+=CUDA_WARP_SIZE) {
			map[j*n + i] = map[i*n + j];
		}
	}
}

extern "C"
__host__ void floydWarshall_h(float* maps_d, int* offsets_d, int* sizes_d, const int& num) {
	floydWarshall_d <<< FLOYD_WARSHALL_KERNEL(num) >>> (maps_d, offsets_d, sizes_d, num);
	cudaThreadSynchronize();
}

__global__ void computeDistanceMatrix_d(float* ret, float* vectors, int dim, int num) {
	int ii = blockDim.x*blockIdx.x + threadIdx.x;

	if (ii >= (num + 1) / 2) return;//each thread compute 2 columns

	float *vi, *vj;
	float d_sq, diff;
	//if idx>j compute ret[j][idx] otherwise compute ret[j][num-1-idx]
	for (int i, j, jj = 0; jj < num; ++jj) if (jj!=ii) {
		j = (ii > jj) ? jj : (num - 1 - jj);
		i = (ii > jj) ? ii : (num - 1 - ii);
		vj = vectors + j*dim;
		vi = vectors + i*dim;
		d_sq = 0.0f;
		for (int d = 0; d < dim; ++d) {
			diff = vj[d] - vi[d];
			d_sq += diff*diff;
		}
		ret[i*num+j] = ret[j*num + i] = sqrt(d_sq);
	}
}

extern "C"
__host__ void computeDistanceMatrix_h(float* ret_d, float* vectors_d, int dim, int num) {
	computeDistanceMatrix_d <<< COMPUTE_DIST_MAT_KERNEL(num) >>> (ret_d, vectors_d, dim, num);
	cudaThreadSynchronize();
}
#endif//VISIBILITY_GRAPH_CU