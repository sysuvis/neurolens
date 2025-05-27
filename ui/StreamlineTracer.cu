#ifndef STREAMLINE_TRACER_CU
#define STREAMLINE_TRACER_CU

#include "typeOperation.h"
#include "helper_cuda.h"
#include <cuda_occupancy.h>
#include <curand_kernel.h>

#define TRACER_KERNEL(n) iDivUp(n,64),64
#define GET_VELO_KERNEL(n) iDivUp(n,256),256

texture<float4, 3, cudaReadModeElementType>	vector_field_tex;
texture<float4, 3, cudaReadModeElementType> time_prev_vec_field_tex;
texture<float4, 3, cudaReadModeElementType> time_next_vec_field_tex;

typedef bool(*getVectorFuncPtr_t)(vec3f&, const vec3f&, const vec3f&);

inline __device__ bool inBound_d(const vec3f& pos, const vec3f& bound){
	return (pos.x>0.0000001f && pos.x<bound.x && pos.y>0.0000001f && pos.y<bound.y
		&& pos.z>0.0000001f && pos.z<bound.z);
}

inline __device__ bool inBound_d(const vec4f& pos, const vec4f& bound){
	return (pos.x>0.0000001f && pos.x<bound.x && pos.y>0.0000001f && pos.y<bound.y
		&& pos.z>0.0000001f && pos.z<bound.z && pos.w<=(bound.w+0.0001f));
}

inline __device__ bool getVector_d(vec3f& ret, const vec3f& pos, const vec3f& bound){
	if (inBound_d(pos, bound))
	{
		float4 val = tex3D(vector_field_tex, pos.x+0.5f, pos.y+0.5f, pos.z+0.5f);
		ret = makeVec3f(val.x, val.y, val.z);
		return true;
	}
	return false;
}

inline __device__ bool getVector_d(vec4f& ret, const vec4f& pos, const vec4f& bound, const float& time_prev){
	if (inBound_d(pos, bound) && pos.w>=time_prev)
	{
		float4 prev = tex3D(time_prev_vec_field_tex, pos.x+0.5f, pos.y+0.5f, pos.z+0.5f);
		float4 next = tex3D(time_next_vec_field_tex, pos.x+0.5f, pos.y+0.5f, pos.z+0.5f);
		float fac = clamp((pos.w-time_prev)/(bound.w-time_prev), 0.0f, 1.0f);
		ret = (1.0f-fac)*makeVec4f(prev.x, prev.y, prev.z, prev.w)+fac*makeVec4f(next.x, next.y, next.z, next.w);
		//normalize(ret);
		return true;
	}
	return false;
}

__device__ bool getBinormal_d(vec3f& ret, const vec3f& pos, const vec3f& bound){
	if (inBound_d(pos, bound))
	{
		float4 a, b;
		vec3f n, v;
		a = tex3D(vector_field_tex, pos.x+0.5f, pos.y+0.5f, pos.z+0.5f);
		v = makeVec3f(a.x, a.y, a.z);
		a = tex3D(vector_field_tex, pos.x+1.0f, pos.y+0.5f, pos.z+0.5f);
		b = tex3D(vector_field_tex, pos.x, pos.y+0.5f, pos.z+0.5f);
		n = v.x*makeVec3f(a.x-b.x, a.y-b.y, a.z-b.z);
		a = tex3D(vector_field_tex, pos.x+0.5f, pos.y+1.0f, pos.z+0.5f);
		b = tex3D(vector_field_tex, pos.x+0.5f, pos.y, pos.z+0.5f);
		n += v.y*makeVec3f(a.x-b.x, a.y-b.y, a.z-b.z);
		a = tex3D(vector_field_tex, pos.x+0.5f, pos.y+0.5f, pos.z+1.0f);
		b = tex3D(vector_field_tex, pos.x+0.5f, pos.y+0.5f, pos.z);
		n += v.z*makeVec3f(a.x-b.x, a.y-b.y, a.z-b.z);
		v = v*1000.0f;
		n = n*1000.0f;
		normalize(v);
		normalize(n);
		ret = cross(v, n);
		return true;
	}
	return false;
}

__device__ bool getNormal_d(vec3f& ret, const vec3f& pos, const vec3f& bound) {
	if (inBound_d(pos, bound))
	{
		float4 a, b;
		vec3f n, v;
		a = tex3D(vector_field_tex, pos.x + 0.5f, pos.y + 0.5f, pos.z + 0.5f);
		v = makeVec3f(a.x, a.y, a.z);
		a = tex3D(vector_field_tex, pos.x + 1.0f, pos.y + 0.5f, pos.z + 0.5f);
		b = tex3D(vector_field_tex, pos.x, pos.y + 0.5f, pos.z + 0.5f);
		n = v.x*makeVec3f(a.x - b.x, a.y - b.y, a.z - b.z);
		a = tex3D(vector_field_tex, pos.x + 0.5f, pos.y + 1.0f, pos.z + 0.5f);
		b = tex3D(vector_field_tex, pos.x + 0.5f, pos.y, pos.z + 0.5f);
		n += v.y*makeVec3f(a.x - b.x, a.y - b.y, a.z - b.z);
		a = tex3D(vector_field_tex, pos.x + 0.5f, pos.y + 0.5f, pos.z + 1.0f);
		b = tex3D(vector_field_tex, pos.x + 0.5f, pos.y + 0.5f, pos.z);
		n += v.z*makeVec3f(a.x - b.x, a.y - b.y, a.z - b.z);
		n = n*1000.0f;
		normalize(n);
		ret = n;
		return true;
	}
	return false;
}

__device__ getVectorFuncPtr_t getVectorFuncPtr = getVector_d;
__device__ getVectorFuncPtr_t getBinormalFuncPtr = getBinormal_d;
__device__ getVectorFuncPtr_t getNormalFuncPtr = getNormal_d;
getVectorFuncPtr_t getVectorFuncPtr_h;

__device__ bool rk4_d(vec3f& pos, const float& interval, const vec3f& bound, getVectorFuncPtr_t get_vector){
	vec3f k1, k2, k3, k4;

	if ((*get_vector)(k1, pos, bound)) {
		if ((*get_vector)(k2, pos+0.5f*interval*k1, bound)) {
			if ((*get_vector)(k3, pos+0.5f*interval*k2, bound)) {
				if ((*get_vector)(k4, pos+interval*k3, bound)) {
					pos = pos+(interval/6.0f)*(k1+2*k2+2*k3+k4);
					if(inBound_d(pos, bound)){
						return true;
					}
				}
			}
		}
	}
	return false;
}

__device__ bool rk4_d(vec4f& pos, const float& interval, const vec4f& bound, const float& time_prev){
	vec4f k1, k2, k3, k4;

	if (getVector_d(k1, pos, bound, time_prev)) {
		if (getVector_d(k2, pos+0.5f*interval*k1, bound, time_prev)) {
			if (getVector_d(k3, pos+0.5f*interval*k2, bound, time_prev)) {
				if (getVector_d(k4, pos+interval*k3, bound, time_prev)) {
					pos = pos+(interval/6.0f)*(k1+2*k2+2*k3+k4);
					if(inBound_d(pos, bound)){
						return true;
					}
				}
			}
		}
	}
	return false;
}

__global__ void cudaTrace4d_d(vec4f* points, int num, vec4f bound, float time_prev, PathlineTraceParameter param){
	int idx = blockDim.x*blockIdx.x+threadIdx.x;

	if (idx<num){
		vec4f pos = points[idx];
		vec4f last_pos = pos;
		vec4f d;

		int step = 0;
		while(rk4_d(pos, param.trace_interval, bound, time_prev) && step<param.max_step){
			d = last_pos-pos;
			if (sqrtf(d*d)>=param.segment_length) {
				last_pos = pos;
				step = 0;
			} else {
				++step;
			}
		}

		points[idx] = pos;
	}
}

__global__ void cudaTrace4d_d(int num, Pathline* lines, vec4f* points, vec3f* velos, vec4f bound, float prev_time, PathlineTraceParameter param) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;

	if (idx < num) {
		Pathline line = lines[idx];
		if (line.numPoint >= param.max_point) return;

		vec4f *local_points = points + line.start;
		vec3f *local_velos = velos + line.start;
		vec4f pos = local_points[line.numPoint];
		vec4f last_pos = local_points[line.numPoint - 1];
		vec4f velo;

		if (line.numPoint == 1) {
			getVector_d(velo, last_pos, bound, prev_time);
			local_velos[0] = velo.xyz;
		}

		int stepCount = 0;
		while (rk4_d(pos, param.trace_interval, bound, prev_time) && line.numPoint < param.max_point && stepCount < param.max_step) {
			if (length(last_pos.xyz - pos.xyz) >= param.segment_length) {
				last_pos = pos;
				getVector_d(velo, pos, bound, prev_time);
				local_velos[line.numPoint] = velo.xyz;
				local_points[line.numPoint++] = pos;
				stepCount = 0;
				continue;
			}
			++stepCount;
		}
		if (line.numPoint < param.max_point) {
			local_points[line.numPoint] = pos;
		}
		lines[idx].numPoint = line.numPoint;
	}
}

__global__ void cudaTrace_d(vec3f *seeds, bool *forward, int num, Streamline* lines, vec3f* points, StreamlineTraceParameter param, vec3f bound,
							getVectorFuncPtr_t get_vector){
	int idx = blockDim.x*blockIdx.x+threadIdx.x;

	if (idx<num) {
		bool isForward = forward[idx];
		vec3f pos = seeds[idx], lastPos;
		int offset, inc;
		float interval;
		if (isForward) {
			inc = 1;
			interval = param.trace_interval;
			offset = idx*param.max_point;
		} else {
			inc = -1;
			interval = -param.trace_interval;
			offset = (idx+1)*param.max_point-1;
		}

		lastPos = pos;
		points[offset] = pos;

		int count=1, stepCount=0, store_gap=0;
		//backward tracing first
		while (rk4_d(pos, interval, bound, get_vector) && count<param.max_point && stepCount<param.max_step) {
			if (dist3d(lastPos, pos)>=param.segment_length) {
				lastPos = pos;
				if (store_gap == param.store_gap) {
					offset += inc;
					points[offset] = pos;
					++count;
					store_gap = 0;
				}
				++store_gap;
				stepCount = -1;
			}
			++stepCount;
		}

		lines[idx].sid = idx;
		lines[idx].numPoint = count;
		if (isForward) {
			lines[idx].start = idx*param.max_point;
		} else {
			lines[idx].start = (idx+1)*param.max_point-count;
		}
	}
}

__global__ void cudaTrace_d(vec3f *seeds, int num, Streamline* lines, vec3f* points, StreamlineTraceParameter param, vec3f bound,
							getVectorFuncPtr_t get_vector){
	int idx = blockDim.x*blockIdx.x+threadIdx.x;

	if (idx<(2*num)) {
		bool isForward = idx<num;
		vec3f pos, lastPos;
		int offset, inc;
		float interval;
		if (isForward) {
			pos = seeds[idx];
			inc = 1;
			interval = param.trace_interval;
			offset = (2*idx+1)*param.max_point;
		} else {
			pos = seeds[idx-num];
			inc = -1;
			interval = -param.trace_interval;
			offset = (2*(idx-num)+1)*param.max_point;
		}
		
		lastPos = pos;
		points[offset] = pos;

		int count=1, stepCount=0, store_gap=0;
		while (rk4_d(pos, interval, bound, get_vector) && count<param.max_point && stepCount<param.max_step) {
			if (dist3d(lastPos, pos)>=param.segment_length) {
				lastPos = pos;
				if (store_gap == param.store_gap) {
					offset += inc;
					points[offset] = pos;
					++count;
					store_gap = 0;
				}
				++store_gap;
				stepCount = -1;
			}
			++stepCount;
		}

		lines[idx].sid = idx;
		if (isForward) {
			lines[idx].start = (2*idx+1)*param.max_point;
			lines[idx].numPoint = count;
		} else {
			lines[idx].start = (2*(idx-num)+1)*param.max_point-count+1;
			lines[idx].numPoint = count-1;
		}
	}
}

__global__ void cudaTrace_d(vec3f *seeds, int num, Streamline* lines, vec3f* points, float* acc_curvs, 
	StreamlineTraceParameter param, vec3f bound, getVectorFuncPtr_t get_vector) 
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;

	if (idx < (2 * num)) {
		bool isForward = idx < num;
		vec3f pos, last_saved_pos, p_pos, p_seg, seg;
		float acc_curv = 0.0f;
		int offset, inc;
		float interval;
		if (isForward) {
			pos = seeds[idx];
			inc = 1;
			interval = param.trace_interval;
			offset = (2 * idx + 1)*param.max_point;
		} else {
			pos = seeds[idx - num];
			inc = -1;
			interval = -param.trace_interval;
			offset = (2 * (idx - num) + 1)*param.max_point;
		}

		p_pos = last_saved_pos = pos;
		p_seg = makeVec3f(0.0f);
		points[offset] = pos;

		int count = 1, stepCount = 0, store_gap = 0;
		float* store_acc_curv = isForward ? acc_curvs : (acc_curvs + 1);
		while (rk4_d(pos, interval, bound, get_vector) && count < param.max_point && stepCount < param.max_step) {
			//update curvature
			seg = pos - p_pos;
			normalize(seg);
			acc_curv += acosf(clamp(seg*p_seg, -1.0f, 1.0f));
			p_pos = pos;
			p_seg = seg;
			//save point if distance threshold is reached
			if (dist3d(last_saved_pos, pos) >= param.segment_length) {
				last_saved_pos = pos;
				if (store_gap == param.store_gap) {
					offset += inc;
					points[offset] = pos;
					store_acc_curv[offset] = acc_curv;
					++count;
					store_gap = 0;
					acc_curv = 0.0f;
				}
				++store_gap;
				stepCount = -1;
			}
			++stepCount;
		}

		lines[idx].sid = idx;
		if (isForward) {
			lines[idx].start = (2 * idx + 1)*param.max_point;
			lines[idx].numPoint = count;
		} else {
			lines[idx].start = (2 * (idx - num) + 1)*param.max_point - count + 1;
			lines[idx].numPoint = count - 1;
		}
	}
}

extern "C" 
void cudaTrace4d_h(cudaArray *vec_time_prev, cudaArray* vec_time_next, 
				   const float& time_prev, const float& time_next,
				   const vec3i& dim, 
				   vec4f* points_d, Pathline* lines_d, vec3f* velos_d,
				   const int& num, 
				   const PathlineTraceParameter& param)
{
	time_prev_vec_field_tex.normalized = false;                      // access with normalized texture coordinates
	time_prev_vec_field_tex.filterMode = cudaFilterModeLinear;      // linear interpolation
	time_prev_vec_field_tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	time_prev_vec_field_tex.addressMode[1] = cudaAddressModeClamp;
	time_prev_vec_field_tex.addressMode[2] = cudaAddressModeClamp;

	time_next_vec_field_tex.normalized = false;                      // access with normalized texture coordinates
	time_next_vec_field_tex.filterMode = cudaFilterModeLinear;      // linear interpolation
	time_next_vec_field_tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	time_next_vec_field_tex.addressMode[1] = cudaAddressModeClamp;
	time_next_vec_field_tex.addressMode[2] = cudaAddressModeClamp;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	checkCudaErrors(cudaBindTextureToArray(time_prev_vec_field_tex, vec_time_prev, channelDesc));
	checkCudaErrors(cudaBindTextureToArray(time_next_vec_field_tex, vec_time_next, channelDesc));
	vec4f bound = makeVec4f(dim.x-1.0000001f, dim.y-1.0000001f, dim.z-1.0000001f, time_next);

	checkCudaErrors(cudaThreadSynchronize());
	cudaTrace4d_d<<<TRACER_KERNEL(num)>>>(num, lines_d, points_d, velos_d, bound, time_prev, param);
	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());
} 

extern "C" 
void cudaTracePoint4d_h(cudaArray *vec_time_prev, cudaArray* vec_time_next, const float& time_prev, const float& time_next,
				   const vec3i& dim, vec4f* points_h, const int& num, const PathlineTraceParameter& param)
{
	time_prev_vec_field_tex.normalized = false;                      // access with normalized texture coordinates
	time_prev_vec_field_tex.filterMode = cudaFilterModeLinear;      // linear interpolation
	time_prev_vec_field_tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	time_prev_vec_field_tex.addressMode[1] = cudaAddressModeClamp;
	time_prev_vec_field_tex.addressMode[2] = cudaAddressModeClamp;

	time_next_vec_field_tex.normalized = false;                      // access with normalized texture coordinates
	time_next_vec_field_tex.filterMode = cudaFilterModeLinear;      // linear interpolation
	time_next_vec_field_tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	time_next_vec_field_tex.addressMode[1] = cudaAddressModeClamp;
	time_next_vec_field_tex.addressMode[2] = cudaAddressModeClamp;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	checkCudaErrors(cudaBindTextureToArray(time_prev_vec_field_tex, vec_time_prev, channelDesc));
	checkCudaErrors(cudaBindTextureToArray(time_next_vec_field_tex, vec_time_next, channelDesc));
	vec4f bound = makeVec4f(dim.x-1.0000001f, dim.y-1.0000001f, dim.z-1.0000001f, time_next);

	vec4f *points_d;
	checkCudaErrors(cudaMalloc((void**)&points_d, sizeof(vec4f)*num));
	checkCudaErrors(cudaMemcpy(points_d, points_h, sizeof(vec4f)*num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaThreadSynchronize());
	cudaTrace4d_d<<<TRACER_KERNEL(num)>>>(points_d, num, bound, time_prev, param);
	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(points_h, points_d, sizeof(vec4f)*num, cudaMemcpyDeviceToHost));

	cudaFree(points_d);
} 

extern "C" 
void cudaTrace_h(cudaArray *vec_field_d, const vec3i& dim, vec3f* seeds_h, const int& num, const StreamlineTraceParameter& param, Streamline* lines_h, vec3f* points_h){
	vector_field_tex.normalized = false;                      // access with normalized texture coordinates
	vector_field_tex.filterMode = cudaFilterModeLinear;      // linear interpolation
	vector_field_tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	vector_field_tex.addressMode[1] = cudaAddressModeClamp;
	vector_field_tex.addressMode[2] = cudaAddressModeClamp;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	checkCudaErrors(cudaBindTextureToArray(vector_field_tex, vec_field_d, channelDesc));
	vec3f bound = makeVec3f(dim.x-1.0000001f, dim.y-1.0000001f, dim.z-1.0000001f);

	vec3f *seeds_d, *points_d;
	Streamline* lines_d;
	checkCudaErrors(cudaMalloc((void**)&seeds_d, sizeof(vec3f)*num));
	checkCudaErrors(cudaMalloc((void**)&points_d, sizeof(vec3f)*num*2*param.max_point));
	checkCudaErrors(cudaMalloc((void**)&lines_d, sizeof(Streamline)*num*2));
	checkCudaErrors(cudaMemcpy(seeds_d, seeds_h, sizeof(vec3f)*num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaThreadSynchronize());
	if (param.trace_type==STREAMLINE_TRACE_VELOCITY) {
		cudaMemcpyFromSymbol(&getVectorFuncPtr_h, getVectorFuncPtr, sizeof(getVectorFuncPtr_t));
	} else if (param.trace_type==STREAMLINE_TRACE_NORMAL) {
		cudaMemcpyFromSymbol(&getVectorFuncPtr_h, getNormalFuncPtr, sizeof(getVectorFuncPtr_t));
	} else {
		cudaMemcpyFromSymbol(&getVectorFuncPtr_h, getBinormalFuncPtr, sizeof(getVectorFuncPtr_t));
	}
	cudaTrace_d<<<TRACER_KERNEL(2*num)>>>(seeds_d, num, lines_d, points_d, param, bound, getVectorFuncPtr_h);
	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(points_h, points_d, sizeof(vec3f)*num*2*param.max_point, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(lines_h, lines_d, sizeof(Streamline)*num*2, cudaMemcpyDeviceToHost));

	cudaFree(seeds_d);
	cudaFree(points_d);
	cudaFree(lines_d);
}

extern "C"
void cudaTracePointCurvature_h(cudaArray *vec_field_d, const vec3i& dim, const vec3f* seeds_h, const int& num, 
	const StreamlineTraceParameter& param, Streamline* lines_h, vec3f* points_h, float* curv_h) {
	vector_field_tex.normalized = false;                      // access with normalized texture coordinates
	vector_field_tex.filterMode = cudaFilterModeLinear;      // linear interpolation
	vector_field_tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	vector_field_tex.addressMode[1] = cudaAddressModeClamp;
	vector_field_tex.addressMode[2] = cudaAddressModeClamp;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	checkCudaErrors(cudaBindTextureToArray(vector_field_tex, vec_field_d, channelDesc));
	vec3f bound = makeVec3f(dim.x - 1.0000001f, dim.y - 1.0000001f, dim.z - 1.0000001f);

	vec3f *seeds_d, *points_d;
	float* curv_d;
	Streamline* lines_d;
	checkCudaErrors(cudaMalloc((void**)&seeds_d, sizeof(vec3f)*num));
	checkCudaErrors(cudaMalloc((void**)&points_d, sizeof(vec3f)*num * 2 * param.max_point));
	checkCudaErrors(cudaMalloc((void**)&curv_d, sizeof(vec3f)*num * 2 * param.max_point));
	checkCudaErrors(cudaMalloc((void**)&lines_d, sizeof(Streamline)*num * 2));
	checkCudaErrors(cudaMemcpy(seeds_d, seeds_h, sizeof(vec3f)*num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaThreadSynchronize());
	if (param.trace_type == STREAMLINE_TRACE_VELOCITY) {
		cudaMemcpyFromSymbol(&getVectorFuncPtr_h, getVectorFuncPtr, sizeof(getVectorFuncPtr_t));
	} else if (param.trace_type == STREAMLINE_TRACE_NORMAL) {
		cudaMemcpyFromSymbol(&getVectorFuncPtr_h, getNormalFuncPtr, sizeof(getVectorFuncPtr_t));
	} else {
		cudaMemcpyFromSymbol(&getVectorFuncPtr_h, getBinormalFuncPtr, sizeof(getVectorFuncPtr_t));
	}
	cudaTrace_d << <TRACER_KERNEL(2 * num) >> > (seeds_d, num, lines_d, points_d, curv_d, param, bound, getVectorFuncPtr_h);
	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(points_h, points_d, sizeof(vec3f)*num * 2 * param.max_point, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(curv_h, curv_d, sizeof(float)*num * 2 * param.max_point, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(lines_h, lines_d, sizeof(Streamline)*num * 2, cudaMemcpyDeviceToHost));

	cudaFree(seeds_d);
	cudaFree(points_d);
	cudaFree(lines_d);
}

extern "C" 
void cudaTraceOneDirection_h(cudaArray *vec_field_d, const vec3i& dim, vec3f* seeds_h, bool* forward_h, const int& num, const StreamlineTraceParameter& param, Streamline* lines_h, vec3f* points_h){
	vector_field_tex.normalized = false;                      // access with normalized texture coordinates
	vector_field_tex.filterMode = cudaFilterModeLinear;      // linear interpolation
	vector_field_tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	vector_field_tex.addressMode[1] = cudaAddressModeClamp;
	vector_field_tex.addressMode[2] = cudaAddressModeClamp;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	checkCudaErrors(cudaBindTextureToArray(vector_field_tex, vec_field_d, channelDesc));
	vec3f bound = makeVec3f(dim.x-1.0000001f, dim.y-1.0000001f, dim.z-1.0000001f);

	vec3f *seeds_d, *points_d;
	Streamline* lines_d;
	bool* forward_d;
	checkCudaErrors(cudaMalloc((void**)&seeds_d, sizeof(vec3f)*num));
	checkCudaErrors(cudaMalloc((void**)&forward_d, sizeof(bool)*num));
	checkCudaErrors(cudaMalloc((void**)&points_d, sizeof(vec3f)*num*param.max_point));
	checkCudaErrors(cudaMalloc((void**)&lines_d, sizeof(Streamline)*num));
	checkCudaErrors(cudaMemcpy(seeds_d, seeds_h, sizeof(vec3f)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(forward_d, forward_h, sizeof(bool)*num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaThreadSynchronize());
	if (param.trace_type==STREAMLINE_TRACE_VELOCITY) {
		cudaMemcpyFromSymbol(&getVectorFuncPtr_h, getVectorFuncPtr, sizeof(getVectorFuncPtr_t));
	} else {
		cudaMemcpyFromSymbol(&getVectorFuncPtr_h, getBinormalFuncPtr, sizeof(getVectorFuncPtr_t));
	}
	cudaTrace_d<<<TRACER_KERNEL(num)>>>(seeds_d, forward_d, num, lines_d, points_d, param, bound, getVectorFuncPtr_h);
	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(points_h, points_d, sizeof(vec3f)*num*param.max_point, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(lines_h, lines_d, sizeof(Streamline)*num, cudaMemcpyDeviceToHost));

	cudaFree(seeds_d);
	cudaFree(points_d);
	cudaFree(lines_d);
} 

__global__ void getVelos_d(vec3f* points, vec3f* velos, int num, vec3f bound) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;

	if (idx < num) {
		getVector_d(velos[idx], points[idx], bound);
	}
}

extern "C"
void cudaGetVelos_h(cudaArray *vec_field_d, const vec3i& dim, vec3f* points_h, vec3f* velos_h, const int& num) {
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	checkCudaErrors(cudaBindTextureToArray(vector_field_tex, vec_field_d, channelDesc));
	vec3f bound = makeVec3f(dim.x - 1.0000001f, dim.y - 1.0000001f, dim.z - 1.0000001f);

	vec3f *velos_d, *points_d;
	cudaMalloc((void**)&points_d, sizeof(vec3f)*num);
	cudaMalloc((void**)&velos_d, sizeof(vec3f)*num);
	cudaMemcpy(points_d, points_h, sizeof(vec3f)*num, cudaMemcpyHostToDevice);

	getVelos_d << <GET_VELO_KERNEL(num) >> > (points_d, velos_d, num, bound);
	cudaMemcpy(velos_h, velos_d, sizeof(vec3f)*num, cudaMemcpyDeviceToHost);

	cudaFree(points_d);
	cudaFree(velos_d);
}

inline __device__ vec4f newRandPoint_d(const vec3f& bound, curandState* state) {
	vec4f ret;
	ret.x = bound.x*curand_uniform(state);
	ret.y = bound.y*curand_uniform(state);
	ret.z = bound.z*curand_uniform(state);
	ret.w = 1.0f;
	return ret;
}

inline __device__ bool toReproducePoint_d(const float& t, const float& t_upper, const float& t_lower, 
	curandState* state)
{
	if (t < t_lower) return false;
	else if (t > t_upper) return true;
	else {
		float r = curand_uniform(state);
		if (((t - t_lower)/(t_upper - t_lower))< r) {
			return false;
		}
	}
	return true;
}

__global__ void setupRand_d(curandState* states, int total) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= total) return;
	curand_init(0xbeef, idx, 0, &states[idx]);
}

__global__ void generateRandPoints_d(vec4f* points, int total, int num_per_thread, vec3f bound, 
	curandState* states) 
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	int block_total = blockDim.x*num_per_thread;
	int start = blockIdx.x*block_total + threadIdx.x;
	int end = start + block_total;
	if (end > total) end = total;
	
	for (int i = start; i < end; i += blockDim.x) {
		points[i] = newRandPoint_d(bound, states+idx);
	}
}

extern "C"
__host__ void generateRandPoints_h(vec4f* points, int total, int num_per_thread, vec3i dim)
{
	vec3f bound = makeVec3f(dim.x - 1, dim.y - 1, dim.z - 1);
	curandState *states_d;
	checkCudaErrors(cudaMalloc(&states_d, total*sizeof(curandState)));

	int block_size, min_grid_size, grid_size;
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, generateRandPoints_d, 0, 0);
	grid_size = iDivUp(total, num_per_thread*block_size);

	setupRand_d <<< iDivUp(total, block_size), block_size >>> (states_d, total);
	generateRandPoints_d <<< grid_size, block_size >>>(points, total, num_per_thread, bound, states_d);
	
	checkCudaErrors(cudaFree(states_d));
}

__global__ void updatePointsRand_d(vec4f* points, int total, int num_per_thread, vec3f bound,
	float t_upper, float t_lower, curandState* states)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	int block_total = blockDim.x*num_per_thread;
	int start = blockIdx.x*block_total + threadIdx.x;
	int end = start + block_total;
	if (end > total) end = total;

	for (int i = start; i < end; i += blockDim.x) {
		vec4f p = points[i];
		if (toReproducePoint_d(p.w, t_upper, t_lower, states + idx)) {
			points[i] = newRandPoint_d(bound, states + idx);
		}
	}
}

extern "C"
__host__ void updatePointsRand_h(vec4f* points, int total, int num_per_thread, vec3i dim, float t_upper, float t_lower)
{
	vec3f bound = makeVec3f(dim.x - 1, dim.y - 1, dim.z - 1);
	curandState *states_d;
	checkCudaErrors(cudaMalloc(&states_d, sizeof(curandState)));

	int block_size, min_grid_size, grid_size;
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, updatePointsRand_d, 0, 0);
	grid_size = iDivUp(total, num_per_thread*block_size);

	setupRand_d <<< iDivUp(total, block_size), block_size >>> (states_d, total);
	updatePointsRand_d <<< grid_size, block_size >>>(points, total, num_per_thread, bound, t_upper, t_lower, states_d);

	checkCudaErrors(cudaFree(states_d));
}

extern "C"
__host__ void cudaUpdateVectorFieldTex_h(cudaArray* vec_time_prev, cudaArray* vec_time_next) {
	time_prev_vec_field_tex.normalized = false;                      // access with normalized texture coordinates
	time_prev_vec_field_tex.filterMode = cudaFilterModeLinear;      // linear interpolation
	time_prev_vec_field_tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	time_prev_vec_field_tex.addressMode[1] = cudaAddressModeClamp;
	time_prev_vec_field_tex.addressMode[2] = cudaAddressModeClamp;

	time_next_vec_field_tex.normalized = false;                      // access with normalized texture coordinates
	time_next_vec_field_tex.filterMode = cudaFilterModeLinear;      // linear interpolation
	time_next_vec_field_tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	time_next_vec_field_tex.addressMode[1] = cudaAddressModeClamp;
	time_next_vec_field_tex.addressMode[2] = cudaAddressModeClamp;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	checkCudaErrors(cudaBindTextureToArray(time_prev_vec_field_tex, vec_time_prev, channelDesc));
	checkCudaErrors(cudaBindTextureToArray(time_next_vec_field_tex, vec_time_next, channelDesc));
}

#endif // #ifndef STREAMLINE_TRACER_CU
