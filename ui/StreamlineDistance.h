#ifndef STREAMLINE_DISTANCE_H
#define STREAMLINE_DISTANCE_H

#include "typeOperation.h"
#include <cuda.h>

inline __host__ __device__ float computePointSegmentDistance(const vec3f& p, const vec3f& s0, const vec3f& s1){
	vec3f v = s1-s0, w = p-s0;

	float vw = v*w;
	if (vw<=0.0f) return dist3d(p, s0);
	float vv = v*v;
	if(vw>=vv) return dist3d(p, s1);

	vec3f pb = s0+vw/vv*v;
	return dist3d(p, pb);
}

inline __host__ __device__ float computeMinimumClosestPointDistanceWithFilter(vec3f* arr1, vec3f* arr2,
									const int& num1, const int& num2, bool* filter)
{
	float minv=1e30, dist;
	for (int i=0; i<num1; ++i) if(filter[i]) {
		for (int j=0; j<num2; ++j) {
			dist = dist3d(arr1[i], arr2[j]);
			if(dist<minv) minv = dist;
		}
	}

	return minv;

}

inline __host__ __device__ float computeClosestPointDistance(vec3f* arr, const int& num, const vec3f& p){
	float minv = dist3d(arr[0], p), dist;
	for (int i=1; i<num; ++i) {
		dist = dist3d(arr[i], p);
		if (dist<minv) minv = dist; 
	}
	return minv;
}

inline __host__ __device__ float computeClosestPointDistanceSquare(vec3f* arr, const int& num, const vec3f& p){
	float minv = 1e30, dist;
	vec3f diff;
	for (int i=0; i<num; ++i) {
		diff = arr[i]-p;
		dist = diff*diff;
		if (dist<minv) minv = dist; 
	}
	return minv;
}

inline __host__ __device__ float computeClosestPointDistance(vec3f* arr1, vec3f* arr2, const int& num1, const int& num2){
	int i, j;
	float minv, dist, sum=0.0f;
	for (i=0; i<num1; ++i) {
		minv = dist3d(arr1[i], arr2[0]);
		for (j=1; j<num2; ++j) {
			dist = dist3d(arr1[i], arr2[j]);
			if(dist<minv) minv = dist;
		}
		sum += minv;
	}

	return (sum/num1);
}

inline __host__ __device__ void computeClosestPointDistance(vec3f* arr1, vec3f* arr2, const int& num1, const int& num2,
															float& avg, float& minv, float& maxv)
{
	int i, j;
	float closest, dist, sum=0.0f;
	minv = 1e30;
	maxv = -1e30;
	for (i=0; i<num1; ++i) {
		closest = dist3d(arr1[i], arr2[0]);
		for (j=1; j<num2; ++j) {
			dist = dist3d(arr1[i], arr2[j]);
			if(dist<closest) closest = dist;
		}
		if (closest<minv) minv = closest;
		if (closest>maxv) maxv = closest;
		sum += closest;
	}

	avg = (sum/num1);
} 

inline __host__ __device__ float computeClosestPointDistanceMin(vec3f* arr1, vec3f* arr2, const int& num1, const int& num2){
	int i, j;
	float dist;
	float minv = 1e30;
	for (i=0; i<num1; ++i) {
		for (j=0; j<num2; ++j) {
			dist = dist3d(arr1[i], arr2[j]);
			if(dist<minv) minv = dist;
		}
	}

	return minv;
}  

//assume hist start from all-zero entries
inline __host__ __device__ void computeClosestPointDisanceDistribution(vec3f* arr1, vec3f* arr2, 
																	   const int& num1, const int& num2,
																	   const float& bin_width, const int& num_bin,
																	   short* hist,
																	   float& avg, float& minv, float& maxv)
{
	int i, j, bin;
	float closest, dist, sum=0.0f;
	minv = 1e30;
	maxv = -1e30;
	for (i=0; i<num1; ++i) {
		closest = dist3d(arr1[i], arr2[0]);
		for (j=1; j<num2; ++j) {
			dist = dist3d(arr1[i], arr2[j]);
			if(dist<closest) closest = dist;
		}
		if (closest<minv) minv = closest;
		if (closest>maxv) maxv = closest;
		bin = closest/bin_width;
		if (bin>=num_bin) bin = num_bin-1;
		++hist[bin];
		sum += closest;
	}

	avg = (sum/num1);
}

//return MCP distance matrix
float* genClosestPointDistanceMatrix(Streamline* stls, vec3f* points, vec2i* pairs, const int& numStls, const int& numPairs);
float* genClosestPointDistanceMatrix(Streamline* stls, vec3f* points, const int& numStls);

//return mean, max and min of closest point distance
void computeClosestPointDistanceMatrix(Streamline* stls, vec3f* points, const int& numStls, float* avg, float* minv, float* maxv);

//return mean, max, min and histogram of closest point distance
void computeClosestPointDistanceMatrix(Streamline* stls, vec3f* points, const int& numStls,
												const float& bin_width, const int& num_bin, float* avg, float* minv, float* maxv, short* hist);

//return JSD matrix from curvature and torsion
void computeCurvatureTorsionJSDMatrix(Streamline* stls, vec3f* points, const int& numStls, const int& numBin,
									  float* curvatureJSDMat, float* torsionJSDMat);

//connection distance (by lines traced from binormal fields, probably)
int* genConnectionDistance(Streamline* stls, Streamline* connects, vec3f* stls_points, vec3f* connect_points,
							 const int& numStls, const int& numConnects, const float& dist_thresh);

int* genConnectionDistance(float* dist_mat, const int& n, const int& m, const float& dist_thresh);

float* genLineToLineSmallestDistance(Streamline* stls1, Streamline* stls2, vec3f* points1, vec3f* points2,
								  const int& numStls1, const int& numStls2);

float* cudaGenMinMeanLineToGroupDistance(Streamline* stl1, Streamline* stl2, vec3f* points1, vec3f* points2, 
										 const int& num_stl1, const int& num_stl2, int* group_member, const int& group_size);

void cudaComputeClosestPointDistanceMatrix(Streamline* stls, vec3f* points, const int& num_stls,
										   const float& bin_width, const int& num_bin, 
										   float* avg, float* minv, float* maxv, short* hist);

float* cudaGenLineToLineSmallestDistance(Streamline* stls1, Streamline* stls2, vec3f* points1, vec3f* points2,
									 const int& numStls1, const int& numStls2);

float* cudaGenClosestPointDistanceMatrixInSmallPools(Streamline* stls, vec3f* points, const int& num_stls);
float* cudaGenClosestPointDistanceFromLineToSmallPools(Streamline* stls, vec3f* points, const int& num_stls, vec3f* line_points, const int& line_point_num);

float cudaComputeMCPOneWay(vec3f* from, vec3f* to, int from_num, int to_num);

float cudaComputeMCPOneWayWithDeviceMemoryInput(vec4f* from_d, vec4f* to_d, int from_num, int to_num);

float* genDiscreteFrechetDistanceMatrix(Streamline* stls, const int& num_stls, vec3f* points);
#endif//STREAMLINE_DISTANCE_H