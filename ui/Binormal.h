#ifndef BINORMAL_H
#define BINORMAL_H

#include "typeOperation.h"
#include <cuda.h>

__host__ vec3f* computeBinormalField(vec3f* vf, const vec3i& dim);
__host__ void computeNormalBinormalField(vec3f* vf, const vec3i& dim, vec3f* binormal, vec3f* normal);

__host__ bool writeField(const char* filename, vec3f* vf, const int& size);

inline __device__ __host__ vec3f getNormalVector(const vec3i& pos, vec3f*** vf, const vec3i& dim){
	vec3f v = vf[pos.z][pos.y][pos.x], ret;
	
	if (pos.x==0) {
		ret = v.x*(vf[pos.z][pos.y][1]-vf[pos.z][pos.y][0]);
	} else if (pos.x==(dim.x-1)) {
		ret = v.x*(vf[pos.z][pos.y][dim.x-1]-vf[pos.z][pos.y][dim.x-2]);
	} else {
		ret = v.x*0.5f*(vf[pos.z][pos.y][pos.x+1]-vf[pos.z][pos.y][pos.x-1]);
	}

	if (pos.y==0) {
		ret = ret + v.y*(vf[pos.z][1][pos.x]-vf[pos.z][0][pos.x]);
	} else if (pos.y==(dim.y-1)) {
		ret = ret + v.y*(vf[pos.z][dim.y-1][pos.x]-vf[pos.z][dim.y-2][pos.x]);
	} else {
		ret = ret + v.y*0.5f*(vf[pos.z][pos.y+1][pos.x]-vf[pos.z][pos.y-1][pos.x]);
	}

	if (pos.z==0) {
		ret = ret + v.z*(vf[1][pos.y][pos.x]-vf[0][pos.y][pos.x]);
	} else if (pos.z==(dim.z-1)) {
		ret = ret + v.z*(vf[dim.z-1][pos.y][pos.x]-vf[dim.z-2][pos.y][pos.x]);
	} else {
		ret = ret + v.z*0.5f*(vf[pos.z+1][pos.y][pos.x]-vf[pos.z-1][pos.y][pos.x]);
	}

	return ret;
}

inline __device__ __host__ vec3f getBinormalVector(const vec3i& pos, vec3f*** vf, const vec3i& dim){
	vec3f v = vf[pos.z][pos.y][pos.x], n;
	n = getNormalVector(pos, vf, dim);

	if (isZero(n) || isZero(v)) {
		return makeVec3f(0.0f, 0.0f ,0.0f);
	}
	v = v*1000.0f;
	n = n*1000.0f;
	normalize(v);
	normalize(n);

	vec3f ret = cross(v, n);

	return ret;
}



#endif