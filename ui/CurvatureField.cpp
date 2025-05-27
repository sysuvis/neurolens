#include "CurvatureField.h"
#include "Binormal.h"

float* computeCurvatureField(vec3f* vec_field, const vec3i& dim){
	float *ret, **ret_rows, ***ret_vol;
	vec3f **vf_rows, ***vf_vol;

	allocateVolume(ret, ret_rows, ret_vol, dim.x, dim.y, dim.z);
	allocateVolumeAccess(vec_field, vf_rows, vf_vol, dim.x, dim.y, dim.z);

	vec3f v, n, b;
	float v_len;
	vec3i pos = makeVec3i(0, 0, 0);
	for (pos.z=0; pos.z<dim.z; ++pos.z) {
		for (pos.y=0; pos.y<dim.y; ++pos.y) {
			for (pos.x=0; pos.x<dim.x; ++pos.x) {
				v = vf_vol[pos.z][pos.y][pos.x];
				n = getNormalVector(pos, vf_vol, dim);
				b = cross(v, n);
				v_len = sqrtf(v*v);
				ret_vol[pos.z][pos.y][pos.x] = sqrtf(b*b)/(v_len*v_len*v_len);
				if (ret_vol[pos.z][pos.y][pos.x]!=ret_vol[pos.z][pos.y][pos.x]) {
					ret_vol[pos.z][pos.y][pos.x] = 0.0f;
				}
			}
		}
	}

	delete[] ret_rows;
	delete[] vf_rows;
	delete[] ret_vol;
	delete[] vf_vol;

	return ret;
}

