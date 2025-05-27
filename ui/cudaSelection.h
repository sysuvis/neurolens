#ifndef CUDA_SELECTION_H
#define CUDA_SELECTION_H

#include "typeOperation.h"
#include <vector>

typedef struct{
	int idx;
	float depth;
} DepthItem;

inline CUDA_HOST_DEVICE bool operator < (const DepthItem& a, const DepthItem& b){
	return (a.depth<b.depth);
}

//******************************************************
// cuda implementation
//******************************************************
StreamlineClickInfo cudaComputeClickStreamline(vec2f p, vec3f* points_d, Streamline* stls_d, int num_point, int num_stl, bool* stl_mark_h, float modelview[16], float projection[16]);
int cudaComputeClickBlock(std::vector<int>& ret_blocks, vec2f p, vec3i grid_dim, vec3f block_size, float modelview[16], float projection[16]);
int cudaComputeClickQuad(vec4f* v_d, vec4i* quads_d, const int& num_quads, vec2f p, float modelview[16], float projection[16]);
void cudaComputeClickQuad(std::vector<DepthItem>& ret_depth, vec4f* v_d, vec4i* quads_d, const int& num_quads, const vec2f& p, float modelview[16], float projection[16]);
int cudaComputeClickTriangle(vec4f* v_d, vec3i* tris_d, const int& num_tris, vec2f p, float modelview[16], float projection[16]);
void cudaComputeClickTriangle(std::vector<DepthItem>& ret_depth, vec4f* v_d, vec3i* tris_d, const int& num_tris, const vec2f& p, float modelview[16], float projection[16]);

//******************************************************
// cuda compute area on scree
//******************************************************

void cudaTriangleAreaOnScreen(float* ret_area_d, vec3f* vertices_d, vec3i* triangles_d,
	float* modelview, float* projection, int num_per_thread, int num_triangles);
void cudaTriangleAreaOnScreenHostMem(float* ret_area, vec3f* vertices, vec3i* triangles,
	float* modelview, float* projection, int num_per_thread, int num_vertices, int num_triangles);
void cudaQuadAreaOnScreen(float* ret_area_d, vec3f* vertices_d, vec4i* quads_d,
	float* modelview, float* projection, int num_per_thread, int num_quads);
void cudaQuadAreaOnScreenHostMem(float* ret_area, vec3f* vertices, vec4i* quads,
	float* modelview, float* projection, int num_per_thread, int num_vertices, int num_quads);

//******************************************************
// host implementation
//******************************************************
int computeClickBlock(std::vector<int>& ret_blocks, vec2f p, vec3i grid_dim, vec3f block_size, float modelview[16], float projection[16]);
int computeClickPoint(vec2f p, std::vector<vec3f>& points, float modelview[16], float projection[16]);

//******************************************************
// utility function
//******************************************************
inline CUDA_HOST_DEVICE vec3f getNDCPos(vec3f p, float mat[16]) {
	float w = p.x*mat[12] + p.y*mat[13] + p.z*mat[14] + mat[15];
	vec3f ret;
	ret.x = p.x*mat[0] + p.y*mat[1] + p.z*mat[2] + mat[3];
	ret.y = p.x*mat[4] + p.y*mat[5] + p.z*mat[6] + mat[7];
	ret.z = p.x*mat[8] + p.y*mat[9] + p.z*mat[10] + mat[11];
	ret *= (1.0f / w);

	return ret;
}

inline CUDA_HOST_DEVICE vec2f getNDCPosXY(vec3f p, float mat[16]) {
	float w = p.x*mat[12] + p.y*mat[13] + p.z*mat[14] + mat[15];
	vec2f ret;
	ret.x = p.x*mat[0] + p.y*mat[1] + p.z*mat[2] + mat[3];
	ret.y = p.x*mat[4] + p.y*mat[5] + p.z*mat[6] + mat[7];
	ret *= (1.0f / w);
	return ret;
}

static CUDA_HOST_DEVICE bool testPointInTriangle(vec3f A, vec3f B, vec3f C, vec2f p){
	vec2f v0 = {C.x-A.x, C.y-A.y};
	vec2f v1 = {B.x-A.x, B.y-A.y};
	vec2f v2 = {p.x-A.x, p.y-A.y};

	float dot00 = v0*v0;
	float dot01 = v0*v1;
	float dot02 = v0*v2;
	float dot11 = v1*v1;
	float dot12 = v1*v2;

	float inv = 1/(dot00*dot11-dot01*dot01);
	float u = (dot11*dot02-dot01*dot12)*inv;
	float v = (dot00*dot12-dot01*dot02)*inv;

	return ((u>=0.0f)&&(v>=0.0f)&&(u+v)<1.0f);
}

static CUDA_HOST_DEVICE void computeMVPMatrix(float* ret, float modelview[16], float project[16]){
	memset(ret, 0, sizeof(float)*16);
	for(int i=0; i<4; i++){
		for(int j=0; j<4; j++){
			for(int k=0; k<4; k++){
				//note that modelview and projection matrix are column-majored
				ret[i*4+k] += project[j*4+i]*modelview[k*4+j];
			}
		}
	}
}
#define BLOCK_NOT_SELECTED -1e30
#define QUAD_NOT_SELECTED 1e30
#define TRIANGLE_NOT_SELECTED 1e30

#endif //CUDA_SELECTION_H