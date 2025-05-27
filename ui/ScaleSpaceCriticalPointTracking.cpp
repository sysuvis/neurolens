#include "GaussianSmoothing.h"
#include "CriticalPointDetection.h"
#include "VolumeData.h"
#include <algorithm>
#include "FeatureFlowField.h"

extern "C" 
void cudaTracePoint4d_h(cudaArray *vec_time_prev, cudaArray* vec_time_next, const float& time_prev, const float& time_next,
				   const vec3i& dim, vec4f* points_h, const int& num, const PathlineTraceParameter& param);

void trackCriticalPointInScaleSpace(vec3f* vec_field, const vec3i& dim, 
									const int& gaussian_sample_size,
									const float& min_scale, const float& max_scale, const float& scale_interval)
{
	cudaArray* vf_d =  allocateCudaVectorField(vec_field, dim);

	VolumeData<vec3f> *vf_blur[3];
	float curr_scale = min_scale;
	float step_size_fac = 3.0f/gaussian_sample_size;
	vec3f *tmp;
	tmp = cudaGaussianSmooth3D(vf_d, dim, curr_scale, gaussian_sample_size, step_size_fac*curr_scale);
	curr_scale += scale_interval;
	vf_blur[0] = new VolumeData<vec3f>(dim.x, dim.y, dim.z, tmp);
	tmp = cudaGaussianSmooth3D(vf_d, dim, curr_scale, gaussian_sample_size, step_size_fac*curr_scale);
	curr_scale += scale_interval;
	vf_blur[1] = new VolumeData<vec3f>(dim.x, dim.y, dim.z, tmp);
	tmp = cudaGaussianSmooth3D(vf_d, dim, curr_scale, gaussian_sample_size, step_size_fac*curr_scale);
	curr_scale += scale_interval;
	vf_blur[2] = new VolumeData<vec3f>(dim.x, dim.y, dim.z, tmp);

	std::vector<vec3f> cps;
	std::vector<int> cp_types;
	locateAllCriticalPoints(vf_blur[1]->getData(), dim, cps, cp_types);
	groupCriticalPoints(cps, cp_types, 2.0f);

	std::vector<vec4f> trace_points(cps.size());
	for (int i=0; i<cps.size(); ++i) {
		trace_points[i] = makeVec4f(cps[i].x, cps[i].y, cps[i].z, min_scale+scale_interval);
	}

	cudaArray* feature_flow_d[2];
	VolumeData<vec4f> *feature_flow;
	feature_flow = computeFeatureFlow(vf_blur[0], vf_blur[1], vf_blur[2], 0.2f);
	feature_flow_d[0] = NULL;
	feature_flow_d[1] = allocateCudaVectorField(feature_flow->getData(), dim);
	feature_flow->freeHostMemory();
	delete feature_flow;

	PathlineTraceParameter param;
	param.max_step = 10000;
	param.trace_interval = 0.05f;
	param.segment_length = 0.5f;
	while(curr_scale<=max_scale) {
		vf_blur[0]->freeHostMemory();
		delete vf_blur[0];

		vf_blur[0] = vf_blur[1];
		vf_blur[1] = vf_blur[2];

		tmp = cudaGaussianSmooth3D(vf_d, dim, curr_scale, gaussian_sample_size, step_size_fac*curr_scale);
		curr_scale += scale_interval;
		vf_blur[2] = new VolumeData<vec3f>(dim.x, dim.y, dim.z, tmp);

		if (feature_flow_d[0]!=NULL) {
			cudaFreeArray(feature_flow_d[0]);
		}
		feature_flow_d[0] = feature_flow_d[1];
		feature_flow = computeFeatureFlow(vf_blur[0], vf_blur[1], vf_blur[2], 0.2f);
		feature_flow_d[1] = allocateCudaVectorField(feature_flow->getData(), dim);
		feature_flow->freeHostMemory();
		delete feature_flow;

		cudaTracePoint4d_h(feature_flow_d[0], feature_flow_d[1], curr_scale-3.0f*scale_interval, curr_scale-2.0f*scale_interval, dim, &trace_points[0], trace_points.size(), param);
	}

	cudaFreeArray(vf_d);
	cudaFreeArray(feature_flow_d[0]);
	cudaFreeArray(feature_flow_d[1]);
	vf_blur[0]->freeHostMemory();
	vf_blur[1]->freeHostMemory();
	vf_blur[2]->freeHostMemory();
	delete vf_blur[0];
	delete vf_blur[1];
	delete vf_blur[2];
}

void trackCriticalPointInScaleSpaceSimple(vec3f* vec_field, const vec3i& dim, 
									const int& gaussian_sample_size, 
									const float& max_scale, const float& scale_interval,
									const float& dist_thresh,
									std::vector<vec3f>& ret_points, std::vector<int>& types, std::vector<float>& scales)
{
	cudaArray* vf_d =  allocateCudaVectorField(vec_field, dim);
	std::vector<vec3f> tmp_cps;
	std::vector<int> tmp_cp_types;
	printf("Computing critical point in original vector field.\n");
	locateAllCriticalPoints(vec_field, dim, ret_points, types);
	printf("%i critical points found.\n", ret_points.size());
	scales.assign(ret_points.size(), scale_interval);
	std::vector<bool> tmp_cp_marks;
	std::vector<vec3f> edges;
	float d;

	float step_size_fac = 3.0f/gaussian_sample_size;
	vec3f *vf_blur;
	for (float s = scale_interval; s<max_scale; s+=scale_interval) {
		vf_blur = cudaGaussianSmooth3D(vf_d, dim, s, gaussian_sample_size, step_size_fac*s);
		tmp_cps.clear();
		tmp_cp_types.clear();
		printf("Computing critical point at scale %f.\n", s);
		locateAllCriticalPoints(vf_blur, dim, tmp_cps, tmp_cp_types);
		delete[] vf_blur;

		if (tmp_cps.empty())
			break;

		edges.clear();
		for (int i=0; i<tmp_cps.size(); ++i) {
			for (int j=0; j<ret_points.size(); ++j) {
				if (scales[j]==s && tmp_cp_types[i]==types[j]) {
					d = dist3d(tmp_cps[i],ret_points[j]);
					if (d<dist_thresh) {
						edges.push_back(makeVec3f(d, i, j));
					}
				}
			}
		}

		std::sort(edges.begin(), edges.end());

		tmp_cp_marks.assign(tmp_cps.size(), true);
		int count = 0;
		for (int i=0; i<edges.size();++i) {
			if (scales[(int)(edges[i].z)]==s && tmp_cp_marks[(int)(edges[i].y)]) {
				scales[(int)(edges[i].z)] += scale_interval;
				tmp_cp_marks[(int)(edges[i].y)] = false;
			} else {
				++count;
			}
		}
		printf("%i critical points found.\n", count);
		if (count==0) break;
	}

	cudaFreeArray(vf_d);
}

// void trackCriticalPointInScaleSpaceSimple(vec3f* vec_field, const vec3i& dim, 
// 										  const int& gaussian_sample_size, const float& max_scale, const float& scale_interval,
// 										  const float& dist_thresh,
// 										  std::vector<vec3f>& ret_points, std::vector<int>& types, std::vector<float>& scales)
// {
// 	cudaArray* vf_d =  allocateCudaVectorField(vec_field, dim);
// 	std::vector<vec3f> cur_cps, prev_cps;
// 	std::vector<int> cur_cp_types, prev_cp_types;
// 	locateAllCriticalPoints(vec_field, dim, ret_points, types);
// 	groupCriticalPoints(ret_points, types, 2.0f);
// 	prev_cps.assign(ret_points.begin(), ret_points.end());
// 	prev_cp_types.assign(types.begin(), types.end());
// 	scales.assign(ret_points.size(), 0.0f);
// 
// 	std::vector<int> cur_org_ids;
// 	std::vector<int> prev_org_ids(ret_points.size());
// 	for (int i=0; i<prev_org_ids.size(); ++i) prev_org_ids[i] = i;
// 	std::vector<vec3f> edges;
// 	float d;
// 
// 	std::vector<int> remove_list;
// 
// 	float step_size_fac = 3.0f/gaussian_sample_size;
// 	vec3f *vf_blur;
// 	for (float s = scale_interval; s<max_scale; s+=scale_interval) {
// 		vf_blur = cudaGaussianSmooth3D(vf_d, dim, s, gaussian_sample_size, step_size_fac*s);
// 		cur_cps.clear();
// 		cur_cp_types.clear();
// 		locateAllCriticalPoints(vf_blur, dim, cur_cps, cur_cp_types);
// 		groupCriticalPoints(cur_cps, cur_cp_types, dist_thresh);
// 		cur_org_ids.assign(cur_cps.size(), -1);
// 		delete[] vf_blur;
// 
// 		edges.clear();
// 		for (int i=0; i<cur_cps.size(); ++i) {
// 			for (int j=0; j<prev_cps.size(); ++j) {
// 				d = dist3d(cur_cps[i],ret_points[j]);
// 				if (d<dist_thresh) {
// 					edges.push_back(makeVec3f(d, i, j));
// 				}
// 			}
// 		}
// 
// 		std::sort(edges.begin(), edges.end());
// 
// 		int cur_id, prev_id, org_id;
// 		for (int i=0; i<edges.size();++i) {
// 			d = edges[i].x;
// 			cur_id = edges[i].y;
// 			prev_id = edges[i].z;
// 			org_id = prev_org_ids[prev_id];
// 
// 			if (scales[org_id]<s) {
// 				scales[org_id] = s;
// 				cur_org_ids[cur_id] = org_id;
// 			}
// 		}
// 
// 		remove_list.clear();
// 		for (int i=0; i<cur_org_ids.size(); ++i) {
// 			if (cur_org_ids[i]<0) {
// 				remove_list.push_back(i);
// 			}
// 		}
// 
// 		for (int i=remove_list.size()-1; i>=0; --i) {
// 			cur_cps.erase(cur_cps.begin()+remove_list[i]);
// 			cur_cp_types.erase(cur_cp_types.begin()+remove_list[i]);
// 			cur_org_ids.erase(cur_org_ids.begin()+remove_list[i]);
// 		}
// 
// 		prev_cps.swap(cur_cps);
// 		prev_cp_types.swap(cur_cp_types);
// 		prev_org_ids.swap(cur_org_ids);
// 	}
// 
// 	cudaFreeArray(vf_d);
// }