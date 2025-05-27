#include <algorithm>
#include "cudaSelection.h"
#include "SurfaceSorting.h"

typedef struct {
	int v[4];
	float d;
} SurfaceSortingQuad;

bool operator < (const SurfaceSortingQuad& a, const SurfaceSortingQuad& b){
	return (a.d<b.d);
}

typedef struct {
	int v[3];
	float d;
} SurfaceSortingTriangle;

bool operator < (const SurfaceSortingTriangle& a, const SurfaceSortingTriangle& b){
	return (a.d<b.d);
}

extern "C" void getVertexDepthHost(vec4f* vertices_d, float* depth, const int& num, const vec4f& third_row);
extern "C" void getQuadDepthHost(vec4f* vertices_d, vec4i* indices_d, float* depth, const int& num, const vec4f& third_row);
extern "C" void getTriangleDepthHost(vec4f* vertices_d, vec3i* indices_d, float* depth, const int& num, const vec4f& third_row);
extern "C" void sortDeviceQuadSurfaceHost(vec3f* vertices_d, vec4i* indices_d, const int& num, const vec3f& third_row);

void cudaSortDeviceQuadSurface(vec4f* vertices_d, int* indices_d, int* indices, 
						   const int& num_quads, const int& num_vertices, const vec4f& third_row)
{
	float* depth = new float[num_quads];
	getQuadDepthHost(vertices_d, (vec4i*)indices_d, depth, num_quads, third_row);

	SurfaceSortingQuad *quads = new SurfaceSortingQuad[num_quads];
	for (int i=0, j=0; i<num_quads; ++i, j+=4) {
		quads[i].v[0] = indices[j];
		quads[i].v[1] = indices[j+1];
		quads[i].v[2] = indices[j+2];
		quads[i].v[3] = indices[j+3];
		quads[i].d = depth[i];
	}
	std::sort(quads, quads+num_quads);

	int *new_indices = new int[num_quads*4];
	for (int i=0, j=0; i<num_quads; ++i, j+=4) {
		indices[j] = quads[i].v[0];
		indices[j+1] = quads[i].v[1];
		indices[j+2] = quads[i].v[2];
		indices[j+3] = quads[i].v[3];
	}

	cudaMemcpy(indices_d, indices, sizeof(int)*4*num_quads, cudaMemcpyHostToDevice);

	delete[] depth;
	delete[] quads;
	delete[] new_indices;
}

void cudaSortDeviceQuadSurface(vec3f* vertices_d, vec4i* indices_d, const int& num_quads, 
	float* modelview, float* projection)
{
	float mvp_mat[16];
	computeMVPMatrix(mvp_mat, modelview, projection);
	vec3f third_row = makeVec3f(mvp_mat[8], mvp_mat[9], mvp_mat[10]);

	sortDeviceQuadSurfaceHost(vertices_d, indices_d, num_quads, third_row);
}

void cudaSortDeviceTriangleSurface(vec4f* vertices_d, int* indices_d, int* indices, 
							   const int& num_tris, const int& num_vertices, const vec4f& third_row)
{
	float* depth = new float[num_tris];
	getTriangleDepthHost(vertices_d, (vec3i*)indices_d, depth, num_tris, third_row);

	SurfaceSortingTriangle*tris = new SurfaceSortingTriangle[num_tris];
	for (int i=0, j=0; i<num_tris; ++i, j+=3) {
		tris[i].v[0] = indices[j];
		tris[i].v[1] = indices[j+1];
		tris[i].v[2] = indices[j+2];
		tris[i].d = depth[i];
	}
	std::sort(tris, tris+num_tris);

	int *new_indices = new int[num_tris*4];
	for (int i=0, j=0; i<num_tris; ++i, j+=4) {
		new_indices[j] = tris[i].v[0];
		new_indices[j+1] = tris[i].v[1];
		new_indices[j+2] = tris[i].v[2];
	}

	cudaMemcpy(indices_d, new_indices, sizeof(int)*3*num_tris, cudaMemcpyHostToDevice);

	delete[] depth;
	delete[] tris;
	delete[] new_indices;
}