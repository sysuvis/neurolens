#include "GaussianSmoothing.h"
#include <string>

void gaussianSmooth1D(float* org, const int& num, const int& kernel_size, float* ret){
	int convSize = 6*kernel_size+1;
	float* kernel = computeGaussianKernel1D(kernel_size);

	memset(ret, 0, sizeof(float)*num);

	float curr;
	for (int i=0; i<num; ++i){
		for (int j=i-3*kernel_size, k=0; j<=i+3*kernel_size; ++j, ++k){
			if (j<0) curr = org[0];
			else if (j>=num) curr = org[num-1];
			else curr = org[j];
			ret[i] += curr*kernel[k];
		}
	}

	delete[] kernel;
}

void gaussianSmooth1D(float* vals, const int& num, const int& kernel_size){
	float* tmp = new float[num];
	gaussianSmooth1D(vals, num, kernel_size, tmp);
	memcpy(vals, tmp, sizeof(float)*num);
	delete[] tmp;
}

extern "C" 
vec3f* gaussianSmooth3D_h(cudaArray *vec_field_d, const vec3i& dim, 
								   const float& delta, const int& num_sample, const float& step_size);

vec3f* cudaGaussianSmooth3D(cudaArray *vec_field_d, const vec3i& dim, 
							const float& delta, const int& num_sample, const float& step_size )
{
	return gaussianSmooth3D_h(vec_field_d, dim, delta, num_sample, step_size);	
}
