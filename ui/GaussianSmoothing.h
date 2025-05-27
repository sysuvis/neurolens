#ifndef GAUSSIAN_SMOOTHING_H
#define GAUSSIAN_SMOOTHING_H

#include "typeOperation.h"
#include <cmath>

static float* computeGaussianKernel1D(const float& delta){
	int x;
	int size = 6*delta+1;
	float a=1.0f/(sqrtf(6.2831852f)*delta), b = -0.5f/(delta*delta);
	//compute Kernel
	float* ret = new float[size];
	for(int idx=0; idx<size; idx++){
		x = idx-3*delta;
		ret[idx] = a*exp(x*x*b);
	}
	return ret;
}

static float* computeGaussianKernel3D(const float& delta){
	int r = ceilf(3*delta);
	vec3i x;
	int size = (2*r+1)*(2*r+1)*(2*r+1);
	float a = 1.0f/(sqrtf(6.2831852f)*6.2831852f*delta*delta*delta), b= -0.5f/(delta*delta);

	float *ret = new float[size];
	int count = 0;
	for (x.z=-r ;x.z<=r; ++x.z) {
		for (x.y=-r ;x.y<=r; ++x.y) {
			for (x.x=-r ;x.x<=r; ++x.x) {
				ret[count++] = a*exp(x*x*b);
			}
		}
	}

	return ret;
}

template<class T>
static T convolution3D(const vec3i& p, T*** data, const vec3i& data_dim, float* kernel, const int& r){
	vec3i vol_up = data_dim-makeVec3i(1,1,1), vol_low = makeVec3i(0,0,0);
	vec3i up = p+makeVec3i(r,r,r), low = p-makeVec3i(r,r,r);
	up = clamp(up, vol_low, vol_up);
	low = clamp(low, vol_low, vol_up);
	
	T ret = data[p.z][p.y][p.x]-data[p.z][p.y][p.x];
	vec3i curr, kernel_pos;
	int kernel_len = (2*r+1);
	for (curr.z = low.z, kernel_pos.z = low.z-p.z+r; curr.z<=up.z; ++curr.z, ++kernel_pos.z) {
		for (curr.y = low.y, kernel_pos.y = low.y-p.y+r; curr.y<=up.y; ++curr.y, ++kernel_pos.y) {
			for (curr.x = low.x, kernel_pos.x = low.x-p.x+r; curr.x<=up.x; ++curr.x, ++kernel_pos.x) {
				ret = ret+data[curr.z][curr.y][curr.x]*kernel[(kernel_pos.z*kernel_len+kernel_pos.y)*kernel_len+kernel_pos.x];
			}
		}
	}

	return ret;
}

template<class T>
static T convolutionQuick3D(const vec3i& p, T*** data, const vec3i& data_dim, float* kernel, const int& r){
	vec3i up = p+makeVec3i(r,r,r), low = p-makeVec3i(r,r,r);
	vec3i curr;
	int count = 0;
	T ret = data[p.z][p.y][p.x]-data[p.z][p.y][p.x];

	for (curr.z = low.z; curr.z<=up.z; ++curr.z) {
		for (curr.y = low.y; curr.y<=up.y; ++curr.y) {
			for (curr.x = low.x; curr.x<=up.x; ++curr.x, ++count) {
				ret += data[curr.z][curr.y][curr.x]*kernel[count];
			}
		}
	}

	return ret;
}

template<class T>
static T* gaussianSmooth3D(T* data, const vec3i& dim, const float& delta){
	T *ret;
	T **data_slice, **ret_slice;
	T ***ret_vol, ***data_vol;
	allocateVolume(ret, ret_slice, ret_vol, dim.x, dim.y, dim.z);
	allocateVolumeAccess(data, data_slice, data_vol, dim.x, dim.y, dim.z);

	float* kernel = computeGaussianKernel3D(delta);
	int r = ceilf(3*delta);
	vec3i curr = makeVec3i(0,0,0);
	vec3i low = makeVec3i(r,r,r), up = dim-makeVec3i(r+1,r+1,r+1);

	for (curr.z=0; curr.z<dim.z; ++curr.z) {
		for (curr.y=0; curr.y<dim.y; ++curr.y) {
			for (curr.x=0; curr.x<dim.x; ++curr.x) {
				if (inBound(curr, low, up)) {
					ret_vol[curr.z][curr.y][curr.x] = convolutionQuick3D(curr, data_vol, dim, kernel, r);
				} else {
					ret_vol[curr.z][curr.y][curr.x] = convolution3D(curr, data_vol, dim, kernel, r);
				}
			}
		}
	}

	delete[] data_slice;
	delete[] data_vol;
	delete[] ret_slice;
	delete[] ret_vol;

	return ret;
}

void gaussianSmooth1D(float* org, const int& num, const int& kernel_size, float* ret);
void gaussianSmooth1D(float* vals, const int& num, const int& kernel_size);

vec3f* cudaGaussianSmooth3D(cudaArray *vec_field_d, const vec3i& dim, 
						  const float& delta, const int& num_sample, const float& step_size);

#endif //GAUSSIAN_SMOOTHING_H