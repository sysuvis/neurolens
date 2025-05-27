#ifndef CUDA_STREAMLINE_TRACER_H
#define CUDA_STREAMLINE_TRACER_H

#include "typeOperation.h"
#include <cuda.h>

class cudaStreamlineTracer{
public:
	cudaStreamlineTracer(vec3f* vec_field, const int& w, const int& h, const int& d);
	~cudaStreamlineTracer();

	void trace(StreamlinePool& ret, std::vector<vec3f>& seeds, const StreamlineTraceParameter& param);

	void trace(StreamlinePool& ret, vec3f* seeds, const int& num_seeds, const StreamlineTraceParameter& param);

	void trace(vec3f* seeds, const int& num_seeds, const StreamlineTraceParameter& param, 
		Streamline** streamlines, vec3f** points);

	void trace(vec3f* seeds, const int& num_seeds, const StreamlineTraceParameter& param,
		Streamline** streamlines, vec3f** points, int* valid_line_num);
	
	void trace(vec3f* seeds, bool* is_forward, const int& num_seeds, 
		const StreamlineTraceParameter& param, 
		Streamline** streamlines, vec3f** points, int* valid_line_num);
	
	//record point position and accumulate curvature over each segment
	void trace(std::vector<Streamline>& streamlines, std::vector<vec3f>& points, std::vector<float>& acc_curvs, 
		const std::vector<vec3f>& seeds, const StreamlineTraceParameter& param);

	bool genRandomPool(Streamline** ret_stls, vec3f** ret_points, const StreamlineTraceParameter& param, const int& num_keep);
	bool genAndSaveRandomPool(const int& num_gen, const int& num_keep, const char* file);
	bool genAndSaveRandomPool(const StreamlineTraceParameter& param, const int& num_keep, const char* file);

	void allocateAndCombinePools(vec3f** ret_points, Streamline** ret_stls, vec3f* points_1, Streamline* stls_1, const int& num_stl_1, vec3f* points_2, Streamline* stls_2, const int& num_stl_2);
	void combinePools(vec3f* ret_points, Streamline* ret_stls, vec3f* points_1, Streamline* stls_1, const int& num_stl_1, vec3f* points_2, Streamline* stls_2, const int& num_stl_2);

	cudaArray* getCudaVecField(){return mCudaVecField;}
	void getVelos(vec3f* points, vec3f** velos, int num_point);
	void getAccCurvature(std::vector<float>& ret, const std::vector<vec3f>& seeds, const StreamlineTraceParameter& pars);
	
	bool savePool(vec3f* points, vec3f* velos, Streamline* stls, int num_point, int num_stl, float radius, const char* file);


private:
	cudaArray*	mCudaVecField;
	vec3i		mFieldDim;
};

#endif //CUDA_STREAMLINE_TRACER_H