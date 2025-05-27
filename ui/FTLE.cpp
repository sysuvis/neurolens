#include "FTLE.h"
#include "cudaPathlineTracer.h"
#include <cuda.h>
#include <string>
#include <vector>
#include "QLAlgorithm.h"

inline void ftleAddTracePoint(const vec3i& p,const vec3i& bound, const float& t, 
							  std::vector<vec4f>& points, std::vector<float>& delta)
{
	if (p.x==0){
		points.push_back(makeVec4f(p.x, p.y, p.z, t));
		points.push_back(makeVec4f(p.x+0.25f, p.y, p.z, t));
		delta.push_back(0.25f);
	} else if (p.x==bound.x){
		points.push_back(makeVec4f(p.x-0.25f, p.y, p.z, t));
		points.push_back(makeVec4f(p.x, p.y, p.z, t));
		delta.push_back(0.25f);
	} else {
		points.push_back(makeVec4f(p.x-0.25f, p.y, p.z, t));
		points.push_back(makeVec4f(p.x+0.25f, p.y, p.z, t));
		delta.push_back(0.5f);
	}
	if (p.y==0){
		points.push_back(makeVec4f(p.x, p.y, p.z, t));
		points.push_back(makeVec4f(p.x, p.y+0.25f, p.z, t));
		delta.push_back(0.25f);
	} else if (p.y==bound.y){
		points.push_back(makeVec4f(p.x, p.y-0.25f, p.z, t));
		points.push_back(makeVec4f(p.x, p.y, p.z, t));
		delta.push_back(0.25f);
	} else {
		points.push_back(makeVec4f(p.x, p.y-0.25f, p.z, t));
		points.push_back(makeVec4f(p.x, p.y+0.25f, p.z, t));
		delta.push_back(0.5f);
	}
	if (p.z==0){
		points.push_back(makeVec4f(p.x, p.y, p.z, t));
		points.push_back(makeVec4f(p.x, p.y, p.z+0.25f, t));
		delta.push_back(0.25f);
	} else if (p.z==bound.z){
		points.push_back(makeVec4f(p.x, p.y, p.z-0.25f, t));
		points.push_back(makeVec4f(p.x, p.y, p.z, t));
		delta.push_back(0.25f);
	} else {
		points.push_back(makeVec4f(p.x, p.y, p.z-0.25f, t));
		points.push_back(makeVec4f(p.x, p.y, p.z+0.25f, t));
		delta.push_back(0.5f);
	}
}

inline float ftleAtPoint(vec4f* points, float* delta){
	float mat[3][3];
	mat[0][0] = (points[1].x-points[0].x)/delta[0];
	mat[1][0] = (points[1].y-points[0].y)/delta[0];
	mat[2][0] = (points[1].z-points[0].z)/delta[0];
	mat[0][1] = (points[3].x-points[2].x)/delta[1];
	mat[1][1] = (points[3].y-points[2].y)/delta[1];
	mat[2][1] = (points[3].z-points[2].z)/delta[1];
	mat[0][2] = (points[5].x-points[4].x)/delta[2];
	mat[1][2] = (points[5].y-points[4].y)/delta[2];
	mat[2][2] = (points[5].z-points[4].z)/delta[2];
	float diag[3], subd[3];

	tridiagonal3(mat, diag, subd);
	QLAlgorithm3(diag, subd, mat);
	float max_eigen = diag[0];
	for (int i=1; i<3; ++i){
		if (diag[i]>max_eigen){
			max_eigen = diag[i];
		}
	}

	return log(sqrtf(max_eigen));
}

VolumeData<float>* FTLE(const char* directory, const char* filename_format, 
				 const int& start_time, const int& end_time,
				 const int& w, const int& h, const int& d)
{
	printf("Compute FTLE t=%d, T=%d.\n", start_time, end_time-start_time);
	int max_per_iteration = 10240;

	cudaPathlineTracer tracer(directory, filename_format, start_time, end_time, w, h, d);
	PathlineTraceParameter par;

	VolumeData<float>* ftle = new VolumeData<float>(w, h, d);
	int size = ftle->volumeSize();
	float* ftle_data = ftle->getData();

	float one_over_t = 1.0f/(end_time-start_time);

	std::vector<vec4f> points;
	std::vector<float> delta;
	vec3i p, bound=makeVec3i(w-1,h-1,d-1);
	for (int i=0; i<size; i+=max_per_iteration){
		points.clear();
		delta.clear();
		for (int j=i; j<i+max_per_iteration && j<size; ++j){
			ftle->idxToPos(j, p.x, p.y, p.z);
			ftleAddTracePoint(p, bound, start_time, points, delta);
		}
		tracer.trace(&points[0], points.size(), par);
		for (int j=i, k=0; k<max_per_iteration && j<size; ++j, ++k){
			ftle_data[j] = one_over_t*ftleAtPoint(&points[k*6], &delta[k*3]);
		}
		printf("\rFinish %5.4f", clamp((i+max_per_iteration)/(float)size, 0.0f, 1.0f));
	}
	printf("\rFinished.      \n");

	return ftle;
}
