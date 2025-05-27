#include "cudaPathlineTracer.h"
#include <time.h>

extern "C" 
void cudaTrace4d_h(cudaArray *vec_time_prev, cudaArray* vec_time_next, 
				   const float& time_prev, const float& time_next,
				   const vec3i& dim, 
				   vec4f* points_d, Pathline* lines_d, vec3f* velos_d,
				   const int& num, 
				   const PathlineTraceParameter& param);

extern "C" 
void cudaTracePoint4d_h(cudaArray *vec_time_prev, cudaArray* vec_time_next, const float& time_prev, const float& time_next,
						const vec3i& dim, vec4f* points_h, const int& num, const PathlineTraceParameter& param);

cudaPathlineTracer::cudaPathlineTracer(
	const char* directory, const char* filename_format, 
	const int& start_time, const int& end_time,
	const int& w, const int& h, const int& d)
:mFieldSize(w*h*d),
mStartTime(start_time),
mEndTime(end_time),
mIs4DField(false)
{
	mFieldDim = makeVec3i(w, h, d);

	std::string str_dir(directory);
	if (str_dir[str_dir.size()-1]!='/') {
		str_dir.push_back('/');
	}
	char filename[2048];
	for (int i=start_time; i<=end_time; ++i) {
		sprintf(filename, filename_format, i);
		mFilepaths.push_back(str_dir+filename);
		mTimeMapping.push_back(i);
	}
}

cudaPathlineTracer::~cudaPathlineTracer(){
}

cudaArray* cudaPathlineTracer::readVectorField(const int& t, const float& t_speed){
	if (t<0 || t>=mFilepaths.size()) return NULL;

	std::ifstream vec_file;
	open_file(vec_file, mFilepaths[t].c_str(), true);

	cudaArray* ret;
	if (mIs4DField){
		vec4f *vec_field = new vec4f[mFieldSize];
		vec_file.read((char*)vec_field, sizeof(vec4f)*mFieldSize);
		ret = allocateCudaVectorField(vec_field, mFieldDim);
		delete[] vec_field;
	} else {
		vec3f *vec_field = new vec3f[mFieldSize];
		vec_file.read((char*)vec_field, sizeof(vec3f)*mFieldSize);
		ret = allocateCudaVectorField(vec_field, mFieldDim, t_speed);
		delete[] vec_field;
	}
	
	return ret;
}

void cudaPathlineTracer::genRandomPool(Pathline** ret_lines, vec4f** ret_pts, vec3f** ret_velos, const PathlineTraceParameter& param)
{
	vec4f* seeds = new vec4f[param.max_pathline];
	srand(time(NULL));
	for (int i=0; i<param.max_pathline; ++i) {
		seeds[i].x = rand()%(100*(mFieldDim.x-1))/100.0f;
		seeds[i].y = rand()%(100*(mFieldDim.y-1))/100.0f;
		seeds[i].z = rand()%(100*(mFieldDim.z-1))/100.0f;
		seeds[i].w = 0.0f;
	}

	trace(seeds, param.max_pathline, param, ret_lines, ret_pts, ret_velos);
	delete[] seeds;
}

void cudaPathlineTracer::trace(vec4f* seeds, const int& num_seeds, const PathlineTraceParameter& param, 
		   Pathline** pathlines, vec4f** points, vec3f **velos)
{
	Pathline *lines = new Pathline[num_seeds], *lines_d;
	int num_alloc_points = param.max_point*num_seeds;
	vec4f *pts = new vec4f[num_alloc_points], *pts_d;
	vec3f *velos_d;

	for (int i=0; i<num_seeds; ++i) {
		lines[i].id = i;
		lines[i].numPoint = 1;
		lines[i].start = i*param.max_point;
		pts[lines[i].start] = seeds[i];
		pts[lines[i].start+1] = seeds[i];
	}

	checkCudaErrors(cudaMalloc(&lines_d, sizeof(Pathline)*num_seeds));
	checkCudaErrors(cudaMalloc(&pts_d, sizeof(vec4f)*num_alloc_points));
	checkCudaErrors(cudaMalloc(&velos_d, sizeof(vec3f)*num_alloc_points));
	checkCudaErrors(cudaMemcpy(lines_d, lines, sizeof(Pathline)*num_seeds, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pts_d, pts, sizeof(vec4f)*num_alloc_points, cudaMemcpyHostToDevice));

	cudaArray *prev_vec = readVectorField(0, param.time_speed);
	cudaArray *next_vec;
	for (int t=1; t<mFilepaths.size(); ++t) {
		next_vec = readVectorField(t, param.time_speed);
		cudaTrace4d_h(prev_vec, next_vec, mTimeMapping[t-1], mTimeMapping[t], mFieldDim, pts_d, lines_d, velos_d, num_seeds, param);
		checkCudaErrors(cudaFreeArray(prev_vec));
		prev_vec = next_vec;
	}
	checkCudaErrors(cudaFreeArray(next_vec));
	checkCudaErrors(cudaMemcpy(lines, lines_d, sizeof(Pathline)*num_seeds, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pts, pts_d, sizeof(vec4f)*num_alloc_points, cudaMemcpyDeviceToHost));
	vec3f *velos_h = new vec3f[num_alloc_points];
	checkCudaErrors(cudaMemcpy(velos_h, velos_d, sizeof(vec3f)*num_alloc_points, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(lines_d));
	checkCudaErrors(cudaFree(pts_d));

	int num_pts = 0;
	for (int i=0; i<num_seeds; ++i) {
		num_pts += lines[i].numPoint;
	}

	*points = new vec4f[num_pts];
	*velos = new vec3f[num_pts];
	num_pts = 0;
	for (int i=0; i<num_seeds; ++i) {
		lines[i].start = num_pts;
		memcpy((*points)+num_pts, pts+i*param.max_point, sizeof(vec4f)*lines[i].numPoint);
		memcpy((*velos)+num_pts, velos_h+i*param.max_point, sizeof(vec3f)*lines[i].numPoint);
		num_pts += lines[i].numPoint;
	}

	*pathlines = lines;
	delete[] pts;
	delete[] velos_h;
}

void cudaPathlineTracer::trace(vec4f* points, const int& num_points, const PathlineTraceParameter& param){
	cudaArray *prev_vec = readVectorField(0, param.time_speed);
	cudaArray *next_vec;
	for (int t=1; t<mFilepaths.size(); ++t) {
		next_vec = readVectorField(t, param.time_speed);
		cudaTracePoint4d_h(prev_vec, next_vec, mTimeMapping[t-1], mTimeMapping[t], mFieldDim, points, num_points, param);
		checkCudaErrors(cudaFreeArray(prev_vec));
		prev_vec = next_vec;
	}
	checkCudaErrors(cudaFreeArray(next_vec));
}

void cudaPathlineTracer::convertPoints( vec3f** ret_pos, float** ret_time, vec4f* points, const int& num){
	*ret_pos = new vec3f[num];
	*ret_time = new float[num];

	for (int i=0; i<num; ++i) {
		(*ret_pos)[i] = points[i].xyz;
		(*ret_time)[i] = points[i].w;
	}
}

bool cudaPathlineTracer::savePool( vec4f* points, vec3f* velos, Pathline* lines, 
								  int num_lines, const char* filepath )
{
	std::ofstream outfile;
	if(!open_file(outfile, filepath, true)){
		return false;
	}
	Pathline last_line = lines[num_lines-1];
	int num_point = last_line.start+last_line.numPoint;
	outfile.write((char*)&num_point, sizeof(int));
	outfile.write((char*)&num_lines, sizeof(int));
	outfile.write((char*)points, sizeof(vec4f)*num_point);
	outfile.write((char*)velos, sizeof(vec3f)*num_point);
	outfile.write((char*)lines, sizeof(Pathline)*num_lines);
	outfile.close();
	return true;
}

bool cudaPathlineTracer::readPool( vec4f** points, vec3f** velos, Pathline** lines, 
								  int* num_lines, const char* filepath )
{
	std::ifstream file;
	if(!open_file(file, filepath, true)){
		return false;
	}
	int num_point;
	file.read((char*)&num_point, sizeof(int));
	file.read((char*)num_lines, sizeof(int));

	*points = new vec4f[num_point];
	*velos = new vec3f[num_point];
	*lines = new Pathline[*num_lines];

	file.read((char*)(*points), sizeof(vec4f)*num_point);
	file.read((char*)(*velos), sizeof(vec3f)*num_point);
	file.read((char*)(*lines), sizeof(Pathline)*(*num_lines));

	file.close();

	return true;
}

int cudaPathlineTracer::computeTotalPoints(const Pathline* ptls, const int& num_lines){
	const Pathline& last_pathline = ptls[num_lines-1];
	return (last_pathline.start+last_pathline.numPoint);
}

void cudaPathlineTracer::combinePool(vec4f** ret_points, Pathline** ret_ptls, vec3f** ret_velos,
									 const vec4f* points_1, const Pathline* ptls_1, const vec3f* velos_1, const int& num_ptl_1,
									 const vec4f* points_2, const Pathline* ptls_2, const vec3f* velos_2, const int& num_ptl_2)
{
	int num_points_1 = ptls_1[num_ptl_1-1].start+ptls_1[num_ptl_1-1].numPoint;
	int num_points_2 = ptls_2[num_ptl_2-1].start+ptls_2[num_ptl_2-1].numPoint;

	combineArray(ret_points, points_1, num_points_1, points_2, num_points_2);
	combineArray(ret_ptls, ptls_1, num_ptl_1, ptls_2, num_ptl_2);
	combineArray(ret_velos, velos_1, num_points_1, velos_2, num_points_2);

	for (int i=num_ptl_1; i<num_ptl_1+num_ptl_2; ++i) {
		(*ret_ptls)[i].id += num_ptl_1;
		(*ret_ptls)[i].start += num_points_1;
	}
}

//return total number pathlines
int cudaPathlineTracer::combinePool(vec4f** ret_points, Pathline** ret_ptls, vec3f** ret_velos,
									 const std::vector<vec4f*>& points, const std::vector<Pathline*>& ptls, 
									 const std::vector<vec3f*>& velos, const std::vector<int>& num_ptls)
{
	std::vector<int> num_points(ptls.size());
	for (int i=0; i<ptls.size(); ++i) {
		Pathline& p = ptls[i][num_ptls[i]-1];
		num_points[i] = p.start+p.numPoint;
	}

	int num_total_points = 0, num_total_lines = 0;
	combineArray(ret_points, points, num_points, num_total_points);
	combineArray(ret_velos, velos, num_points, num_total_points);
	combineArray(ret_ptls, ptls, num_ptls, num_total_lines);

	int prev_num_points = num_points[0], prev_lines = num_ptls[0];
	for (int i=1; i<num_ptls.size(); ++i) {
		for (int j=0; j<num_ptls[i]; ++j) {
			(*ret_ptls)[j+prev_lines].id += prev_lines;
			(*ret_ptls)[j+prev_lines].start += prev_num_points;
		}
		prev_num_points += num_points[i];
		prev_lines += num_ptls[i];
	}

	return num_total_lines;
}