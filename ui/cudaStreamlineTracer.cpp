#include "cudaStreamlineTracer.h"
#include "helper_cuda.h"
#include "definition.h"
#include <time.h>
#include <cmath>
#include <fstream>

extern "C" 
void cudaTrace_h(cudaArray *vec_field_d, const vec3i& dim, vec3f* seeds_h, const int& num, const StreamlineTraceParameter& param, Streamline* lines_h, vec3f* points_h);

extern "C"
void cudaTracePointCurvature_h(cudaArray *vec_field_d, const vec3i& dim, const vec3f* seeds_h, const int& num, 
	const StreamlineTraceParameter& param, Streamline* lines_h, vec3f* points_h, float* curvs_h);

extern "C" 
void cudaTraceOneDirection_h(cudaArray *vec_field_d, const vec3i& dim, vec3f* seeds_h, bool* forward_h, const int& num, const StreamlineTraceParameter& param, Streamline* lines_h, vec3f* points_h);

extern "C"
void cudaGetVelos_h(cudaArray *vec_field_d, const vec3i& dim, vec3f* points_h, vec3f* velos_h, const int& num);

cudaStreamlineTracer::cudaStreamlineTracer(vec3f* vec_field, const int& w, const int& h, const int& d){
	mFieldDim = makeVec3i(w, h, d);

	//copy as float4
	float4* vf = new float4[w*h*d];
	for (int i=0; i<w*h*d; ++i) {
		vf[i] = make_float4(vec_field[i].x, vec_field[i].y, vec_field[i].z, 1.0f);
	}

	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	checkCudaErrors(cudaMalloc3DArray(&mCudaVecField, &channelDesc, make_cudaExtent(w, h, d)));

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr(vf, w*sizeof(float4), w, h);
	copyParams.dstArray = mCudaVecField;
	copyParams.extent   = make_cudaExtent(w, h, d);
	copyParams.kind     = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams));

	delete[] vf;
}

cudaStreamlineTracer::~cudaStreamlineTracer(){
	checkCudaErrors(cudaFreeArray(mCudaVecField));
}

void cudaStreamlineTracer::trace(StreamlinePool & ret, std::vector<vec3f>& seeds, 
	const StreamlineTraceParameter & param)
{
	trace(ret, seeds.data(), seeds.size(), param);
}

void cudaStreamlineTracer::trace(StreamlinePool& ret, vec3f* seeds, const int& num_seeds, 
	const StreamlineTraceParameter& param)
{
	Streamline *lines;
	vec3f* points;
	trace(seeds, num_seeds, param, &lines, &points);

	const Streamline& last_line = lines[num_seeds - 1];
	int num_points = last_line.start + last_line.numPoint;

	ret.streamlines.assign(lines, lines + num_seeds);
	ret.points.assign(points, points + num_points);

	delete[] lines;
	delete[] points;
}

void cudaStreamlineTracer::trace(vec3f* seeds, const int& num_seeds, const StreamlineTraceParameter& param, 
								 Streamline** streamlines, vec3f** points)
{
	StreamlineTraceParameter par = param;
	par.min_point = 0;
	int ret_num_line;
	trace(seeds, num_seeds, par, streamlines, points, &ret_num_line);
}

void cudaStreamlineTracer::trace(vec3f* seeds, const int& num_seeds, const StreamlineTraceParameter& param, 
								 Streamline** streamlines, vec3f** points, int* valid_line_num)
{
	//[0,n-1]:forward [n,2n-1]:backward
	Streamline *tmp_stl = new Streamline[num_seeds*2];
	vec3f *tmp_points = new vec3f[param.max_point*num_seeds*2];

	cudaTrace_h(mCudaVecField, mFieldDim, seeds, num_seeds, param, tmp_stl, tmp_points);

	int& num_stl = *valid_line_num;
	num_stl = 0;
	int num_points = 0;
	for (int i=0; i<num_seeds; ++i) {
		tmp_stl[i].sid = tmp_stl[i].start - tmp_stl[i + num_seeds].start;
		tmp_stl[i].start = tmp_stl[i+num_seeds].start;
		tmp_stl[i].numPoint += tmp_stl[i+num_seeds].numPoint;
		if (tmp_stl[i].numPoint>=param.min_point) {
			num_points += tmp_stl[i].numPoint;
			++num_stl;
		}
	}

	int count_line=0, count_point=0;
	Streamline *ret_stls = new Streamline[num_stl];
	vec3f *ret_points = new vec3f[num_points];
	for (int i=0; i<num_seeds; ++i) if(tmp_stl[i].numPoint>=param.min_point){
		ret_stls[count_line].sid = tmp_stl[i].sid;
		ret_stls[count_line].start = count_point;
		ret_stls[count_line].numPoint = tmp_stl[i].numPoint;
		memcpy(&ret_points[ret_stls[count_line].start], &tmp_points[tmp_stl[i].start], sizeof(vec3f)*tmp_stl[i].numPoint);
		++count_line;
		count_point += tmp_stl[i].numPoint;
	}

	streamlines[0] = ret_stls;
	points[0] = ret_points;

	delete[] tmp_stl;
	delete[] tmp_points;
}

void cudaStreamlineTracer::trace(std::vector<Streamline>& streamlines, std::vector<vec3f>& points, 
	std::vector<float>& acc_curvs, const std::vector<vec3f>& seeds, const StreamlineTraceParameter& param)
{
	//[0,n-1]:forward [n,2n-1]:backward
	int n = seeds.size();
	std::vector<Streamline> tmp_stl(2 * n);
	std::vector<vec3f> tmp_points(param.max_point*n * 2);
	std::vector<float> tmp_curvs(param.max_point*n * 2);
	
	cudaTracePointCurvature_h(mCudaVecField, mFieldDim, seeds.data(), seeds.size(), param, tmp_stl.data(), 
		tmp_points.data(), tmp_curvs.data());

	streamlines.clear();
	points.clear();
	acc_curvs.clear();
	streamlines.reserve(n);
	for (int i = 0; i < n; ++i) {
		const Streamline& fline = tmp_stl[i];
		const Streamline& bline = tmp_stl[i + n];
		Streamline s = makeStreamline(fline.start - bline.start, points.size(), fline.numPoint+bline.numPoint);
		if (s.numPoint < param.min_point) continue;
		auto point_start = tmp_points.begin() + bline.start;
		points.insert(points.end(), point_start, point_start+s.numPoint);
		auto curv_start = tmp_curvs.begin() + bline.start;
		acc_curvs.insert(acc_curvs.end(), curv_start, curv_start + s.numPoint);
		streamlines.push_back(s);
	}
}

void cudaStreamlineTracer::trace(vec3f* seeds, bool* is_forward, const int& num_seeds, const StreamlineTraceParameter& param, 
								 Streamline** streamlines, vec3f** points, int* valid_line_num)
{
	//[0,n-1]:forward [n,2n-1]:backward
	Streamline *tmp_stl = new Streamline[num_seeds];
	vec3f *tmp_points = new vec3f[param.max_point*num_seeds];

	cudaTraceOneDirection_h(mCudaVecField, mFieldDim, seeds, is_forward, num_seeds, param, tmp_stl, tmp_points);

	int& num_stl = *valid_line_num;
	num_stl = 0;
	int num_points = 0;
	for (int i=0; i<num_seeds; ++i) {
		if (tmp_stl[i].numPoint>=param.min_point) {
			num_points += tmp_stl[i].numPoint;
			++num_stl;
		}
	}

	int count_line=0, count_point=0;
	Streamline *ret_stls = new Streamline[num_stl];
	vec3f *ret_points = new vec3f[num_points];
	for (int i=0; i<num_seeds; ++i) if(tmp_stl[i].numPoint>=param.min_point){
		ret_stls[count_line].sid = 0;
		ret_stls[count_line].start = count_point;
		ret_stls[count_line].numPoint = tmp_stl[i].numPoint;
		memcpy(&ret_points[ret_stls[count_line].start], &tmp_points[tmp_stl[i].start], sizeof(vec3f)*tmp_stl[i].numPoint);
		++count_line;
		count_point += tmp_stl[i].numPoint;
	}

	streamlines[0] = ret_stls;
	points[0] = ret_points;

	delete[] tmp_stl;
	delete[] tmp_points;
}

bool cudaStreamlineTracer::genRandomPool(Streamline** ret_stls, vec3f** ret_points, const StreamlineTraceParameter& param, const int& num_keep){
	//generate seeds
	vec3f* seeds = new vec3f[param.max_streamline];
	srand(time(NULL));
	for (int i=0; i<param.max_streamline; ++i) {
		seeds[i].x = (mFieldDim.x-1)*((rand()%1000)/1000.0f);
		seeds[i].y = (mFieldDim.y-1)*((rand()%1000)/1000.0f);
		seeds[i].z = (mFieldDim.z-1)*((rand()%1000)/1000.0f);
	}
	//trace
	int num_stl, num_points;
	trace(seeds, param.max_streamline*2, param, ret_stls, ret_points, &num_stl);
	num_points = (*ret_stls)[num_stl-1].start+(*ret_stls)[num_stl-1].numPoint;

	if (num_stl>=num_keep) {
		num_stl = num_keep;
		num_points = (*ret_stls)[num_stl-1].start+(*ret_stls)[num_stl-1].numPoint;
		Streamline *tmp_stls = (*ret_stls);
		vec3f *tmp_points = (*ret_points);
		*ret_stls = new Streamline[num_stl];
		*ret_points = new vec3f[num_points];
		memcpy((*ret_stls), tmp_stls, sizeof(Streamline)*num_stl);
		memcpy((*ret_points), tmp_points, sizeof(vec3f)*num_points);
	} else if (num_stl<num_keep){
		printf("Fail to trace enough streamlines: (%d/%d).\n", num_stl, num_keep);
		return false;
	}

	return true;
}

bool cudaStreamlineTracer::genAndSaveRandomPool(const int& num_gen, const int& num_keep, const char* file){
	StreamlineTraceParameter param;
	param.max_streamline = num_gen;
	param.max_point = MAX_POINT_NUM;
	param.min_point = MIN_POINT_NUM;
	param.segment_length = SEG_LEN;
	param.max_step = MAX_STEP_NUM;
	param.trace_interval = TRACE_INTERVAL;

	return genAndSaveRandomPool(param, num_keep, file);
}

bool cudaStreamlineTracer::genAndSaveRandomPool(const StreamlineTraceParameter& param, const int& num_keep, const char* file){
	//generate seeds
	Streamline* stls;
	vec3f* points;
	if(!genRandomPool(&stls, &points, param, num_keep))
		return false;

	int num_point = stls[num_keep-1].start+stls[num_keep-1].numPoint;
	//compute velocity
	vec3f* velos = new vec3f[num_point];
	cudaGetVelos_h(mCudaVecField, mFieldDim, points, velos, num_point);

	savePool(points, velos, stls, num_point, num_keep, TUBE_RADIUS, file);

	return true;
}

void cudaStreamlineTracer::allocateAndCombinePools(vec3f** ret_points, Streamline** ret_stls, 
								   vec3f* points_1, Streamline* stls_1, const int& num_stl_1,
								   vec3f* points_2, Streamline* stls_2, const int& num_stl_2)
{
	int num_points_1 = stls_1[num_stl_1-1].start+stls_1[num_stl_1-1].numPoint;
	int num_points_2 = stls_2[num_stl_2-1].start+stls_2[num_stl_2-1].numPoint;

	*ret_points = new vec3f[num_points_1+num_points_2];
	*ret_stls = new Streamline[num_stl_1+num_stl_2];

	memcpy(*ret_points, points_1, sizeof(vec3f)*num_points_1);
	memcpy(*ret_points+num_points_1, points_2, sizeof(vec3f)*num_points_2);

	memcpy(*ret_stls, stls_1, sizeof(Streamline)*num_stl_1);
	memcpy(*ret_stls+num_stl_1, stls_2, sizeof(Streamline)*num_stl_2);

	for (int i=num_stl_1; i<num_stl_1+num_stl_2; ++i) {
		(*ret_stls)[i].sid += num_stl_1;
		(*ret_stls)[i].start += num_points_1;
	}
}

void cudaStreamlineTracer::combinePools(vec3f* ret_points, Streamline* ret_stls, 
									   vec3f* points_1, Streamline* stls_1, const int& num_stl_1,
									   vec3f* points_2, Streamline* stls_2, const int& num_stl_2)
{
	int num_points_1 = stls_1[num_stl_1-1].start+stls_1[num_stl_1-1].numPoint;
	int num_points_2 = stls_2[num_stl_2-1].start+stls_2[num_stl_2-1].numPoint;

	memcpy(ret_points, points_1, sizeof(vec3f)*num_points_1);
	memcpy(ret_points+num_points_1, points_2, sizeof(vec3f)*num_points_2);

	memcpy(ret_stls, stls_1, sizeof(Streamline)*num_stl_1);
	memcpy(ret_stls+num_stl_1, stls_2, sizeof(Streamline)*num_stl_2);

	for (int i=num_stl_1; i<num_stl_1+num_stl_2; ++i) {
		ret_stls[i].sid += num_stl_1;
		ret_stls[i].start += num_points_1;
	}
}

bool cudaStreamlineTracer::savePool(vec3f* points, vec3f* velos, Streamline* stls, int num_point, int num_stl, float radius, const char* file){
	//save to file
	std::ofstream outfile;
	outfile.open(file, std::ios::binary);
	if(!outfile.is_open()){
		printf("Unable to write file: %s.", file);
		return false;
	}

	outfile.write((char*)&num_point, sizeof(int));
	outfile.write((char*)&num_stl, sizeof(int));
	outfile.write((char*)&radius, sizeof(float));
	outfile.write((char*)points, sizeof(vec3f)*num_point);
	outfile.write((char*)velos, sizeof(vec3f)*num_point);
	outfile.write((char*)stls, sizeof(Streamline)*num_stl);
	outfile.close();

	return true;
}

void cudaStreamlineTracer::getVelos(vec3f* points, vec3f** velos, int num_point){
	*velos = new vec3f[num_point];
	cudaGetVelos_h(mCudaVecField, mFieldDim, points, *velos, num_point);
}

void cudaStreamlineTracer::getAccCurvature(std::vector<float>& ret, const std::vector<vec3f>& seeds, 
	const StreamlineTraceParameter& pars) 
{
	std::vector<Streamline> streamlines;
	std::vector<vec3f> points;
	trace(streamlines, points, ret, seeds, pars);
}