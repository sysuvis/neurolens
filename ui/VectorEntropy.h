#ifndef SCALAR_FIELD_3D_H
#define SCALAR_FIELD_3D_H

#include "StreamlinePool3d.h"

void entropyForStreamlinePool(StreamlinePool3d *pool, float* entropy);
void entropyFromStreamlinePool(StreamlinePool3d *pool, float *entropy, const int &numx, const int &numy, const int &numz);
void entropyFromVectorField(float *entropy, vec3f *vecf, const int &w, const int &h, const int &d,\
							const int &def_w, const int &def_h, const int &def_d);
void entropyFromVectorField(float *entropy, vec3f *vecf, const int &w, const int &h, const int &d);
void entropyFromVectorFieldSequential(float *entropy, vec3f *vecf, const int &w, const int &h, const int &d,\
									  const int &def_w, const int &def_h, const int &def_d);
void entropyFromVectorFieldSequential(float *entropy, vec3f *vecf, const int &w, const int &h, const int &d);

void scaleSalarFieldSizeDown(float *dst, float *src, const int &dst_w, const int &dst_h, const int &dst_d,\
					   const int &src_w, const int &src_h, const int &src_d);
#endif //SCALAR_FIELD_3D_H
