#include "typeOperation.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include "cudaSelection.h"

__global__ void getVertexDepth(float *ret, vec4f *v, vec4f third_row, int num){
	int idx = blockDim.x*blockIdx.x+threadIdx.x;

	if(idx<num){
		ret[idx] = v[idx].x*third_row.x+v[idx].y*third_row.y+v[idx].z*third_row.z;
	}
}

__global__ void removeVertex_d(vec4f *v, int remain_lower, int remain_upper, int num){
	int idx = blockDim.x*blockIdx.x+threadIdx.x;

	if(idx<num){
		if(idx<remain_lower || idx>remain_upper){//to be removed
			if (v[idx].w>0.1f) {
				v[idx].w = 0.05f;
			}
		} else {//to be displayed
			if (v[idx].w<0.1f) {
				v[idx].w = 1.0f;
			}
		}
	}
}

__global__ void resetRemovedVertex_d(vec4f *v, int num){
	int idx = blockDim.x*blockIdx.x+threadIdx.x;

	if(idx<num){
		if(v[idx].w<0.1f){
			v[idx].w = 1.0f;
		}
	}
}

__global__ void getQuadDepth(float *ret, vec4f *v, vec4i* indices, vec4f third_row, 
	int num_per_thread, int num)
{
	int block_total = blockDim.x*num_per_thread;
	int start = blockIdx.x*block_total + threadIdx.x;
	int end = min(start + block_total, num);

	vec4i q;
	for (int i = start; i < end; i += blockDim.x) {
		q = indices[i];
		ret[i] = (v[q.x].xyz + v[q.y].xyz + v[q.z].xyz + v[q.w].xyz)*third_row.xyz;
	}
}

__global__ void getQuadDepth(float *ret, vec3f* points, vec4i* indices, vec3f third_row,
	int num_per_thread, int num_quads) 
{
	int block_total = blockDim.x*num_per_thread;
	int start = blockIdx.x*block_total + threadIdx.x;
	int end = min(start + block_total, num_quads);

	vec4i quad;
	for (int i = start; i < end; i += blockDim.x) {
		quad = indices[i];
		ret[i] = (points[quad.x] + points[quad.y] + points[quad.z] + points[quad.w])*third_row;
	}
}

__global__ void getTriangleDepth(float *ret, vec4f *v, vec3i* indices, vec4f third_row, int num){
	int idx = blockDim.x*blockIdx.x+threadIdx.x;

	if(idx<num){
		vec3i q = indices[idx];
		float d = v[q.x].x*third_row.x+v[q.x].y*third_row.y+v[q.x].z*third_row.z;
		d += v[q.y].x*third_row.x+v[q.y].y*third_row.y+v[q.y].z*third_row.z;
		d += v[q.z].x*third_row.x+v[q.z].y*third_row.y+v[q.z].z*third_row.z;
		ret[idx] = d;
	}
}


extern "C" void getVertexDepthHost(vec4f* vertices_d, float* depth, const int& num, const vec4f& third_row){
	//allocate memory to store depth
	float *depth_d;
	cudaMalloc((void**)&depth_d, sizeof(float)*num);

	getVertexDepth<<<iDivUp(num,256),256>>>(depth_d, vertices_d, third_row, num);
	cudaMemcpy(depth, depth_d, sizeof(float)*num, cudaMemcpyDeviceToHost);
	cudaFree(depth_d);
}

extern "C" void getQuadDepthHost(vec4f* vertices_d, vec4i* indices_d, float* depth, const int& num, const vec4f& third_row){
	//allocate memory to store depth
	float *depth_d;
	cudaMalloc((void**)&depth_d, sizeof(float)*num);

	getQuadDepth<<<iDivUp(num,256),256>>>(depth_d, vertices_d, indices_d, third_row, 16, num);
	cudaMemcpy(depth, depth_d, sizeof(float)*num, cudaMemcpyDeviceToHost);
	cudaFree(depth_d);
}

extern "C" void getTriangleDepthHost(vec4f* vertices_d, vec3i* indices_d, float* depth, const int& num, const vec4f& third_row){
	//allocate memory to store depth
	float *depth_d;
	cudaMalloc((void**)&depth_d, sizeof(float)*num);

	getTriangleDepth<<<iDivUp(num,256),256>>>(depth_d, vertices_d, indices_d, third_row, num);
	cudaMemcpy(depth, depth_d, sizeof(float)*num, cudaMemcpyDeviceToHost);
	cudaFree(depth_d);
}

extern "C" void sortDeviceQuadSurfaceHost(vec3f* vertices_d, vec4i* indices_d, const int& num, const vec3f& third_row){
	float* depth_d;
	cudaMalloc((void**)&depth_d, sizeof(float)*num);

	getQuadDepth<<<iDivUp(num,512),512>>>(depth_d, vertices_d, indices_d, third_row, 16, num);

	thrust::device_ptr<vec4i> indices_ptr(indices_d);
	thrust::device_ptr<float> depth_ptr(depth_d);
	thrust::sort_by_key(depth_ptr, depth_ptr+num, indices_ptr, thrust::greater<float>());

	cudaFree(depth_d);
}

extern "C" void removeVertexDisplay_h(vec4f* vertices_d, const int& remain_lower, const int& remain_upper, const int& num){
	removeVertex_d<<<iDivUp(num,256),256>>>(vertices_d, remain_lower, remain_upper, num);
}

extern "C" void resetRemovedVertex_h(vec4f* vertices_d, const int& num){
	resetRemovedVertex_d<<<iDivUp(num,256),256>>>(vertices_d, num);
}