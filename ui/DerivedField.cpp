#include "CriticalPointDetection.h"


void deriveFromVectorField( float* ret_curl_mag, float* ret_div, float* ret_vel_mag, vec3f* vec_field, const vec3i& dim ){
	vec3f ***vf, **vf_slice;
	allocateVolumeAccess(vec_field, vf_slice, vf, dim.x, dim.y, dim.z);
	
	vec3i p;
	vec3f jacob[3], v, curl;
	int count = 0;
	for (p.z=0; p.z<dim.z; ++p.z) {
		for (p.y=0; p.y<dim.y; ++p.y) {
			for (p.x=0; p.x<dim.x; ++p.x) {
				getJacobianAtGridPoint(p, vf, dim, jacob);
				v = vf[p.z][p.y][p.x];
				curl = makeVec3f(jacob[2].y-jacob[1].z, jacob[0].z-jacob[2].x, jacob[1].x-jacob[0].y);
				ret_curl_mag[count] = sqrtf(curl*curl);
				ret_div[count] = jacob[0].x+jacob[1].y+jacob[2].z;
				ret_vel_mag[count] = sqrtf(v*v);
				++count;
			}
		}
	}
}
