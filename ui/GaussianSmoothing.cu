#ifndef GAUSSIAN_SMOOTHING_CU
#define GAUSSIAN_SMOOTHING_CU

#include "typeOperation.h"
#include "helper_cuda.h"

#define SMOOTHING_KERNEL(n) iDivUp(n,64),64

texture<float4, 3, cudaReadModeElementType>	vec_field_tex;

__global__ void gaussianSmooth3D_d(vec3f* ret, vec3i dim, float delta, int num_sample, int step_size){
	int idx = blockDim.x*blockIdx.x+threadIdx.x;

	if (idx<dim.x*dim.y*dim.z) {
		vec3i c = makeVec3i(idx%dim.x, (idx/dim.x)%dim.y, idx/dim.x/dim.y);
		float init_offset = -num_sample*step_size;
		vec3f s = makeVec3f(c.x+init_offset, c.y+init_offset, c.z+init_offset);
		vec3f p, kp;

		vec3f conv = makeVec3f(0.0f, 0.0f, 0.0f);
		float4 val;
		vec3f vec;
		float b= -0.5f/(delta*delta);

		p.z = s.z;
		kp.z = init_offset;
		float w, w_sum = 0.0f;
		for (int i=-num_sample; i<=num_sample; ++i, p.z+=step_size, kp.z+=step_size) {
			p.y=s.y;
			kp.y=init_offset;
			for (int j=-num_sample; j<=num_sample; ++j, p.y+=step_size, kp.y+=step_size) {
				p.x=s.x;
				kp.x=init_offset;
				for (int k=-num_sample; k<=num_sample; ++k, p.x+=step_size, kp.x+=step_size){
					val = tex3D(vec_field_tex, p.x+0.5f, p.y+0.5f, p.z+0.5f);
					vec = makeVec3f(val.x, val.y, val.z);
					w = exp(kp*kp*b);
					w_sum += w;
					conv += w*vec;
				}
			}
		}
		
		ret[idx] = conv/w_sum;
	}
}

extern "C" 
__host__ vec3f* gaussianSmooth3D_h(cudaArray *vec_field_d, const vec3i& dim, 
								   const float& delta, const int& num_sample, const float& step_size)
{
	vec_field_tex.normalized = false;                      // access with normalized texture coordinates
	vec_field_tex.filterMode = cudaFilterModeLinear;      // linear interpolation
	vec_field_tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	vec_field_tex.addressMode[1] = cudaAddressModeClamp;
	vec_field_tex.addressMode[2] = cudaAddressModeClamp;

	int size = dim.x*dim.y*dim.z;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	checkCudaErrors(cudaBindTextureToArray(vec_field_tex, vec_field_d, channelDesc));

	vec3f *ret_d;
	checkCudaErrors(cudaMalloc((void**)&ret_d, sizeof(vec3f)*size));

	gaussianSmooth3D_d<<<SMOOTHING_KERNEL(size)>>>(ret_d, dim, delta, num_sample, step_size);

	vec3f *ret_h = new vec3f[size];
	checkCudaErrors(cudaMemcpy(ret_h, ret_d, sizeof(vec3f)*size, cudaMemcpyDeviceToHost));
	cudaFree(ret_d);

	return ret_h;
}

#endif//GAUSSIAN_SMOOTHING_CU