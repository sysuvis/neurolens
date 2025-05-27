#include "cudaSelection.h"
#include <algorithm>

inline DepthItem makeDepthItem(const int& idx, const float& depth){
	DepthItem ret = {idx, depth};
	return ret;
}

extern "C" 
StreamlineClickInfo computeClickStreamline_h(vec2f p, vec3f* points_d, Streamline* stls_d, int num_point, int num_stl, bool* stl_mark_h, float mat[16]);

StreamlineClickInfo cudaComputeClickStreamline(vec2f p, vec3f* points_d, Streamline* stls_d, int num_point, int num_stl, bool* stl_mark_h, float modelview[16], float projection[16]){
	float mvp_mat[16];
	computeMVPMatrix(mvp_mat, modelview, projection);

	return computeClickStreamline_h(p, points_d, stls_d, num_point, num_stl, stl_mark_h, mvp_mat);
}

int computeClickBlock(std::vector<int>& ret_blocks, vec2f p, vec3i grid_dim, vec3f block_size, float modelview[16], float projection[16]){
	float mvp_mat[16];
	computeMVPMatrix(mvp_mat, modelview, projection);

	int total_grid = grid_dim.x*grid_dim.y*grid_dim.z;
	float *block_depth = new float[total_grid];
	
	for (int bid = 0; bid<total_grid; ++bid){
		vec3f pos[8];
		pos[0].x = (bid % grid_dim.x)*block_size.x;
		pos[0].y = ((bid/grid_dim.x)%grid_dim.y)*block_size.y;
		pos[0].z = (bid/(grid_dim.x*grid_dim.y))*block_size.z;
		pos[1] = pos[0] + makeVec3f(block_size.x, 0.0f, 0.0f);
		pos[2] = pos[0] + makeVec3f(0.0f, block_size.y, 0.0f);
		pos[3] = pos[0] + makeVec3f(block_size.x, block_size.y, 0.0f);
		pos[4] = pos[0] + makeVec3f(0.0f, 0.0f, block_size.z);
		pos[5] = pos[1] + makeVec3f(0.0f, 0.0f, block_size.z);
		pos[6] = pos[2] + makeVec3f(0.0f, 0.0f, block_size.z);
		pos[7] = pos[3] + makeVec3f(0.0f, 0.0f, block_size.z);

		float z = 0.0f;
		for (int i=0; i<8; ++i) {
			pos[i] = getNDCPos(pos[i], mvp_mat);
			z += pos[i].z;
		}

		if(testPointInTriangle(pos[0], pos[1], pos[2], p)||
			testPointInTriangle(pos[3], pos[1], pos[2], p)||
			testPointInTriangle(pos[1], pos[3], pos[5], p)||
			testPointInTriangle(pos[7], pos[3], pos[5], p)||
			testPointInTriangle(pos[2], pos[3], pos[6], p)||
			testPointInTriangle(pos[7], pos[3], pos[6], p)||
			testPointInTriangle(pos[0], pos[1], pos[4], p)||
			testPointInTriangle(pos[5], pos[1], pos[4], p)||
			testPointInTriangle(pos[0], pos[4], pos[2], p)||
			testPointInTriangle(pos[6], pos[4], pos[2], p)||
			testPointInTriangle(pos[4], pos[5], pos[6], p)||
			testPointInTriangle(pos[7], pos[5], pos[6], p)){
				block_depth[bid]=z;
		} else {
			block_depth[bid]=BLOCK_NOT_SELECTED;
		}
	}

	std::vector<sortElemInc> sort_array;
	for (int i=0; i<total_grid; ++i) {
		if (block_depth[i]>BLOCK_NOT_SELECTED) {
			sort_array.push_back(makeSortElemInc(block_depth[i], i));
		}
	}
	std::sort(sort_array.begin(), sort_array.end());

	ret_blocks.clear();
	for (int i=0; i<sort_array.size(); ++i) {
		ret_blocks.push_back(sort_array[i].idx);
	}

	if (ret_blocks.empty()) {
		return -1;
	}

	delete[] block_depth;

	return ret_blocks[0];
}

extern "C" 
int computeClickBlockDepth_h(float* ret_depth, vec2f p, vec3i grid_dim, vec3f block_size, float mat[16]);

int cudaComputeClickBlock(std::vector<int>& ret_blocks, vec2f p, vec3i grid_dim, vec3f block_size, float modelview[16], float projection[16]){
	float mvp_mat[16];
	computeMVPMatrix(mvp_mat, modelview, projection);

	int total_grid = grid_dim.x*grid_dim.y*grid_dim.z;
	float *block_depth = new float[total_grid];
	computeClickBlockDepth_h(block_depth, p, grid_dim, block_size, mvp_mat);

	std::vector<sortElemInc> sort_array;
	for (int i=0; i<total_grid; ++i) {
		if (block_depth[i]!=BLOCK_NOT_SELECTED) {
			sort_array.push_back(makeSortElemInc(block_depth[i], i));
		}
	}
	std::sort(sort_array.begin(), sort_array.end());

	ret_blocks.clear();
	for (int i=0; i<sort_array.size(); ++i) {
		ret_blocks.push_back(sort_array[i].idx);
	}

	if (ret_blocks.empty()) {
		return -1;
	}

	delete[] block_depth;

	return ret_blocks[0];
}

extern "C"
void computeAllClickQuadDepth_h(float* ret, vec4f* v_d, vec4i* quads_d, const int& num_quads, const vec2f& p, float mat[16]);

void cudaComputeClickQuad(std::vector<DepthItem>& ret_depth, 
						  vec4f* v_d, vec4i* quads_d, const int& num_quads, 
						  const vec2f& p, float modelview[16], float projection[16])
{
	float mvp_mat[16];
	computeMVPMatrix(mvp_mat, modelview, projection);

	ret_depth.clear();
	float* depths = new float[num_quads];
	computeAllClickQuadDepth_h(depths, v_d, quads_d, num_quads, p, mvp_mat);
	for (int i=0; i<num_quads; ++i) {
		if (depths[i]<QUAD_NOT_SELECTED) {
			ret_depth.push_back(makeDepthItem(i, depths[i]));
		}
	}

	delete[] depths;
}

extern "C" 
int computeClickQuad_h(vec4f* v_d, vec4i* quads_d, const int& num_quads, const vec2f& p, float mat[16]);

int cudaComputeClickQuad(vec4f* v_d, vec4i* quads_d, const int& num_quads, vec2f p, float modelview[16], float projection[16]){
	float mvp_mat[16];
	computeMVPMatrix(mvp_mat, modelview, projection);
	return computeClickQuad_h(v_d, quads_d, num_quads, p, mvp_mat);
}

extern "C"
void computeAllClickTriangleDepth_h(float* ret, vec4f* v_d, vec3i* tris_d, const int& num_tris, const vec2f& p, float mat[16]);

void cudaComputeClickTriangle(std::vector<DepthItem>& ret_depth, 
						  vec4f* v_d, vec3i* tris_d, const int& num_tris, 
						  const vec2f& p, float modelview[16], float projection[16])
{
	float mvp_mat[16];
	computeMVPMatrix(mvp_mat, modelview, projection);

	ret_depth.clear();
	float* depths = new float[num_tris];
	computeAllClickTriangleDepth_h(depths, v_d, tris_d, num_tris, p, mvp_mat);
	for (int i=0; i<num_tris; ++i) {
		if (depths[i]<TRIANGLE_NOT_SELECTED) {
			ret_depth.push_back(makeDepthItem(i, depths[i]));
		}
	}

	delete[] depths;
}

extern "C" 
int computeClickTriangle_h(vec4f* v_d, vec3i* tris_d, const int& num_tris, const vec2f& p, float mat[16]);

int cudaComputeClickTriangle(vec4f* v_d, vec3i* tris_d, const int& num_tris, vec2f p, float modelview[16], float projection[16]){
	float mvp_mat[16];
	computeMVPMatrix(mvp_mat, modelview, projection);
	return computeClickTriangle_h(v_d, tris_d, num_tris, p, mvp_mat);
}

int computeClickPoint(vec2f p, std::vector<vec3f>& points, float modelview[16], float projection[16]){
	float mvp_mat[16];
	computeMVPMatrix(mvp_mat, modelview, projection);

	int ret = -1;
	float dist, min_dist = 1e30;
	vec3f projected_point;
	for (int i=0; i<points.size(); ++i) {
		projected_point = getNDCPos(points[i], mvp_mat);
		dist=dist2d(projected_point.xy, p);
		if (dist<min_dist && dist<0.2f) {
			min_dist = dist;
			ret = i;
		}
	}

	return ret;
}

extern "C"
void mapTriangleArea_h(float* ret_area_d, vec3f* vertices_d, vec3i* indices_d, float mat[16],
	int num_per_thread, int num_triangles);

void cudaTriangleAreaOnScreen(float* ret_area_d, vec3f* vertices_d, vec3i* triangles_d,
	float* modelview, float* projection, int num_per_thread, int num_triangles)
{
	float mvp_mat[16];
	computeMVPMatrix(mvp_mat, modelview, projection);
	mapTriangleArea_h(ret_area_d, vertices_d, triangles_d, mvp_mat, num_per_thread, num_triangles);
}

void cudaTriangleAreaOnScreenHostMem(float* ret_area, vec3f* vertices, vec3i* triangles,
	float* modelview, float* projection, int num_per_thread, int num_vertices, int num_triangles)
{
	float mvp_mat[16];
	computeMVPMatrix(mvp_mat, modelview, projection);

	//allocate memory
	float* ret_area_d;
	cudaMalloc(&ret_area_d, sizeof(float)*num_triangles);
	vec3f* vertices_d;
	cudaMalloc(&vertices_d, sizeof(vec3f)*num_vertices);
	cudaMemcpy(vertices_d, vertices, sizeof(vec3f)*num_vertices, cudaMemcpyHostToDevice);
	vec3i* triangles_d;
	cudaMalloc(&triangles_d, sizeof(vec3i)*num_triangles);
	cudaMemcpy(triangles_d, triangles, sizeof(vec3i)*num_triangles, cudaMemcpyHostToDevice);
	//compute
	mapTriangleArea_h(ret_area_d, vertices_d, triangles_d, mvp_mat, num_per_thread, num_triangles);
	//copy data back
	cudaMemcpy(ret_area, ret_area_d, sizeof(float)*num_triangles, cudaMemcpyDeviceToHost);
	cudaFree(ret_area_d);
	cudaFree(vertices_d);
	cudaFree(triangles_d);
}

extern "C"
void mapQuadArea_h(float* ret_area_d, vec3f* vertices_d, vec4i* quads_d, float mat[16],
	int num_per_thread, int num_quads);

void cudaQuadAreaOnScreen(float* ret_area_d, vec3f* vertices_d, vec4i* quads_d,
	float* modelview, float* projection, int num_per_thread, int num_quads)
{
	float mvp_mat[16];
	computeMVPMatrix(mvp_mat, modelview, projection);
	mapQuadArea_h(ret_area_d, vertices_d, quads_d, mvp_mat, num_per_thread, num_quads);
}

void cudaQuadAreaOnScreenHostMem(float* ret_area, vec3f* vertices, vec4i* quads,
	float* modelview, float* projection, int num_per_thread, int num_vertices, int num_quads)
{
	float mvp_mat[16];
	computeMVPMatrix(mvp_mat, modelview, projection);

	//allocate memory
	float* ret_area_d;
	cudaMalloc(&ret_area_d, sizeof(float)*num_quads);
	vec3f* vertices_d;
	cudaMalloc(&vertices_d, sizeof(vec3f)*num_vertices);
	cudaMemcpy(vertices_d, vertices, sizeof(vec3f)*num_vertices, cudaMemcpyHostToDevice);
	vec4i* quads_d;
	cudaMalloc(&quads_d, sizeof(vec4i)*num_quads);
	cudaMemcpy(quads_d, quads, sizeof(vec4i)*num_quads, cudaMemcpyHostToDevice);
	//compute
	mapQuadArea_h(ret_area_d, vertices_d, quads_d, mvp_mat, num_per_thread, num_quads);
	//copy data back
	cudaMemcpy(ret_area, ret_area_d, sizeof(float)*num_quads, cudaMemcpyDeviceToHost);
	cudaFree(ret_area_d);
	cudaFree(vertices_d);
	cudaFree(quads_d);
}