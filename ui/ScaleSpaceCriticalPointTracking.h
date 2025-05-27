#ifndef SCALE_SPACE_CRITICAL_POINT_TRACKING_H
#define SCALE_SPACE_CRITICAL_POINT_TRACKING_H

#include "typeOperation.h"
#include <vector>

void trackCriticalPointInScaleSpace(vec3f* vec_field, const vec3i& dim, 
									const int& gaussian_sample_size,
									const float& min_scale, const float& max_scale, const float& scale_interval);

void trackCriticalPointInScaleSpaceSimple(vec3f* vec_field, const vec3i& dim, 
										  const int& gaussian_sample_size, 
										  const float& max_scale, const float& scale_interval, 
										  const float& dist_thresh,
										  std::vector<vec3f>& ret_points, std::vector<int>& types, std::vector<float>& scales);
#endif//SCALE_SPACE_CRITICAL_POINT_TRACKING_H