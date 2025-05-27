#include "Binormal.h"
#include <fstream>

vec3f* computeBinormalField(vec3f* vf, const vec3i& dim){
	int size = dim.x*dim.y*dim.z;

	vec3f *ret, *normalized_vf;
	vec3f** binormal_rows, **vf_rows;
	vec3f*** binormal_vol, ***vf_vol;

	normalized_vf = new vec3f[size];
	memcpy(normalized_vf, vf, sizeof(vec3f)*dim.x*dim.y*dim.z);
	for (int i=0; i<size; ++i) {
		if (isZero(normalized_vf[i])) {
			normalized_vf[i] = makeVec3f(0.0f, 0.0f, 0.0f);
		} else {
			normalize(normalized_vf[i]);
		}
	}

	allocateVolume(ret, binormal_rows, binormal_vol, dim.x, dim.y, dim.z);
	allocateVolumeAccess(normalized_vf, vf_rows, vf_vol, dim.x, dim.y, dim.z);

	vec3i pos = makeVec3i(0, 0, 0);
	for (pos.z=0; pos.z<dim.z; ++pos.z) {
		for (pos.y=0; pos.y<dim.y; ++pos.y) {
			for (pos.x=0; pos.x<dim.x; ++pos.x) {
				binormal_vol[pos.z][pos.y][pos.x] = getBinormalVector(pos, vf_vol, dim);
			}
		}
	}

	delete[] normalized_vf;
	delete[] binormal_rows;
	delete[] vf_rows;
	delete[] binormal_vol;
	delete[] vf_vol;

	return ret;
}

void computeNormalBinormalField(vec3f* vf, const vec3i& dim, vec3f* binormal, vec3f* normal){
	int size = dim.x*dim.y*dim.z;

	vec3f *ret, *normalized_vf;
	vec3f **vf_rows;
	vec3f ***vf_vol;

	normalized_vf = new vec3f[size];
	memcpy(normalized_vf, vf, sizeof(vec3f)*dim.x*dim.y*dim.z);
	for (int i=0; i<size; ++i) {
		if (isZero(normalized_vf[i])) {
			normalized_vf[i] = makeVec3f(0.0f, 0.0f, 0.0f);
		} else {
			normalize(normalized_vf[i]);
		}
	}

	allocateVolumeAccess(normalized_vf, vf_rows, vf_vol, dim.x, dim.y, dim.z);

	int count = 0;
	vec3i pos = makeVec3i(0, 0, 0);
	vec3f v, n;
	for (pos.z=0; pos.z<dim.z; ++pos.z) {
		for (pos.y=0; pos.y<dim.y; ++pos.y) {
			for (pos.x=0; pos.x<dim.x; ++pos.x, ++count) {
				v = normalized_vf[count];
				n = getNormalVector(pos, vf_vol, dim);
				if (isZero(n) || isZero(v)) {
					normal[count] = n;
					binormal[count] = makeVec3f(0.0f, 0.0f ,0.0f);
					continue;
				}
				v = 1000.0f*v;
				n = 1000.0f*n;
				normalize(v);
				normalize(n);

				normal[count] = n;
				binormal[count] = cross(v, n);
			}
		}
	}

	delete[] normalized_vf;
	delete[] vf_rows;
	delete[] vf_vol;
}

bool writeField(const char* filename, vec3f* vf, const int& size){
	std::ofstream outfile;
	outfile.open(filename, std::ios::binary);
	if(!outfile.is_open()){
		printf("Unable to write file: %s.", filename);
		return false;
	}

	outfile.write((char*)vf, sizeof(vec3f)*size);
	outfile.close();

	return true;
}