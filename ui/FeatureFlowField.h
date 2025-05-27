#ifndef FEATURE_FLOW_FIELD_H
#define FEATURE_FLOW_FIELD_H

#include "typeOperation.h"
#include "VolumeData.h"

VolumeData<vec4f>* computeFeatureFlow(VolumeData<vec3f>* prev, VolumeData<vec3f>* curr, VolumeData<vec3f>* next, const float& scale_diff);
VolumeData<vec4f>* computeFeatureFlowFirst(VolumeData<vec3f>* curr, VolumeData<vec3f>* next, const float& scale_diff);
VolumeData<vec4f>* computeFeatureFlowLast(VolumeData<vec3f>* prev, VolumeData<vec3f>* curr, const float& scale_diff);

#endif //FEATURE_FLOW_FIELD_H

