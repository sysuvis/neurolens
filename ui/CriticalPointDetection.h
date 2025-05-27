#ifndef CRITICAL_POINT_DETECTION_H
#define CRITICAL_POINT_DETECTION_H

#include "typeOperation.h"
#include <vector>

#define CP_REPEL_NODE 0x0
#define CP_REPEL_FOCUS 0x4
#define CP_REPEL_NODE_SADDLE 0x1
#define CP_REPEL_FOCUS_SADDLE 0x5
#define CP_ATTRACT_NODE_SADDLE 0x3
#define CP_ATTRACT_FOCUS_SADDLE 0x7
#define CP_ATTRACT_NODE 0x2
#define CP_ATTRACT_FOCUS 0x6

#define CP_CENTER 0x8
#define CP_FOCUS 0x4
#define CP_ATTRACT 0x2
#define CP_SADDLE 0x1


void locateAllCriticalPoints(vec3f* vecField, const vec3i& dim, std::vector<vec3f>& ret, std::vector<int>& ret_type);
void groupCriticalPoints(std::vector<vec3f>& cp, std::vector<int>& type, const float& dist_thresh);
void groupCriticalPoints(std::vector<vec3f>& cp, std::vector<int>& type, std::vector<float>& scales, const float& dist_thresh);

int classifyCriticalPoint(const vec3f& pos, vec3f*** vf, const vec3i& dim);
float computeIndexGA(const vec3f& p, vec3f*** vf, const float& cube_size, const int& sample_num_per_edge);

//read and write
template<typename T>
bool readCriticalPoints(std::vector<T>& ret_points, std::vector<int>& types, std::vector<float>& scales, const char* file_name);
template<typename T>
bool saveCriticalPoints(std::vector<T>& points, std::vector<int>& types, std::vector<float>& scales, const char* file_name);

static __host__ __device__ void getJacobianAtGridPoint(const vec3i& pos, vec3f*** vf, const vec3i& dim, vec3f ret[3]){
	vec3f v = vf[pos.z][pos.y][pos.x];

	if (pos.x==0) {
		ret[0] = vf[pos.z][pos.y][1]-vf[pos.z][pos.y][0];
	} else if (pos.x==(dim.x-1)) {
		ret[0] = vf[pos.z][pos.y][dim.x-1]-vf[pos.z][pos.y][dim.x-2];
	} else {
		ret[0] = 0.5f*(vf[pos.z][pos.y][pos.x+1]-vf[pos.z][pos.y][pos.x-1]);
	}

	if (pos.y==0) {
		ret[1] =vf[pos.z][1][pos.x]-vf[pos.z][0][pos.x];
	} else if (pos.y==(dim.y-1)) {
		ret[1] = vf[pos.z][dim.y-1][pos.x]-vf[pos.z][dim.y-2][pos.x];
	} else {
		ret[1] = 0.5f*(vf[pos.z][pos.y+1][pos.x]-vf[pos.z][pos.y-1][pos.x]);
	}

	if (pos.z==0) {
		ret[2] = vf[1][pos.y][pos.x]-vf[0][pos.y][pos.x];
	} else if (pos.z==(dim.z-1)) {
		ret[2] = vf[dim.z-1][pos.y][pos.x]-vf[dim.z-2][pos.y][pos.x];
	} else {
		ret[2] = 0.5f*(vf[pos.z+1][pos.y][pos.x]-vf[pos.z-1][pos.y][pos.x]);
	}
}

#endif//CRITICAL_POINT_DETECTION_H