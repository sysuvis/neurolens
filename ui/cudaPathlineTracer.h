#ifndef CUDA_PATHLINE_TRACER_H
#define CUDA_PATHLINE_TRACER_H

#include "typeOperation.h"
#include <cuda.h>

class cudaPathlineTracer{
public:
	cudaPathlineTracer(const char* directory, const char* filename_format, 
		const int& start_time, const int& end_time,
		const int& w, const int& h, const int& d);
	~cudaPathlineTracer();

	void set4DField(const bool& b_4d){mIs4DField=b_4d;}

	void genRandomPool(Pathline** ret_lines, vec4f** ret_pts, vec3f** ret_velos, const PathlineTraceParameter& param);
	void trace(vec4f* seeds, const int& num_seeds, const PathlineTraceParameter& param, Pathline** pathlines, vec4f** points, vec3f **velos);
	void trace(vec4f* points, const int& num_points, const PathlineTraceParameter& param);
	static int computeTotalPoints(const Pathline* ptls, const int& num_lines);
	static void convertPoints(vec3f** ret_pos, float** ret_time, vec4f* points, const int& num);
	static int combinePool(vec4f** ret_points, Pathline** ret_ptls, vec3f** ret_velos, const std::vector<vec4f*>& points, const std::vector<Pathline*>& ptls, const std::vector<vec3f*>& velos, const std::vector<int>& num_ptls);
	static void combinePool(vec4f** ret_points, Pathline** ret_ptls, vec3f** ret_velos, const vec4f* points_1, const Pathline* ptls_1, const vec3f* velos_1, const int& num_ptl_1, const vec4f* points_2, const Pathline* ptls_2, const vec3f* velos_2, const int& num_ptl_2);
	static bool savePool(vec4f* points, vec3f* velos, Pathline* lines, int num_lines, const char* filepath);
	static bool readPool(vec4f** points, vec3f** velos, Pathline** lines, int* num_stl, const char* filepath);
	
private:
	cudaArray* readVectorField(const int& t, const float& t_speed);

	std::vector<std::string> mFilepaths;
	std::vector<int> mTimeMapping;
	cudaArray*	mCudaVecField;
	vec3i		mFieldDim;
	int			mFieldSize;
	int			mStartTime;
	int			mEndTime;
	bool		mIs4DField;
};

#endif //CUDA_PATHLINE_TRACER_H