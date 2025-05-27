#ifndef DERIVED_FIELD_H
#define DERIVED_FIELD_H

#include "typeOperation.h"

void deriveFromVectorField(float* ret_curl_mag, float* ret_div, float* ret_vel_mag, vec3f* vec_field, const vec3i& dim);

#endif//DERIVED_FIELD_H