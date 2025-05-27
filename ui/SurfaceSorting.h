#ifndef SURFACE_SORTING_H
#define SURFACE_SORTING_H

#include "typeOperation.h"

//third_row is the third row of the model_view matrix
void cudaSortDeviceQuadSurface(vec4f* vertices_d, int* indices_d, int* indices,
							   const int& num_quads, const int& num_vertices, const vec4f& third_row);
void cudaSortDeviceTriangleSurface(vec4f* vertices_d, int* indices_d, int* indices, 
								   const int& num_tris, const int& num_vertices, const vec4f& third_row);

//
void cudaSortDeviceQuadSurface(vec3f* vertices_d, vec4i* indices_d, const int& num_quads, 
	float* modelview, float* projection);


#endif //SURFACE_SORTING_H