#include "StreamlineResample.h"

//indices is within each streamline rather than entire streamline pool
void resampleStreamlineDiscreteCurvature(vec3f* points, float* curvature, const int& numPoint, float thresh, vec3f *retPoints, int *retIndices, const int& minNum, int& retNum){
	float acc=0.0f;
	retPoints[0] = points[0];
	retIndices[0] = 0;
	retNum = 1;
	int last;
	int count = 0;
	while(retNum<minNum && count<MAX_RESAMPLE_ITER){
		retNum = 1; acc=0.0f;
		for (int i=1; i<numPoint; ++i){
			acc += curvature[i];
			if(acc>thresh){
				retPoints[retNum] = points[i];
				retIndices[retNum] = i;
				last = i;
				retNum++;
				acc = 0.0f;
			}
		}
		if (last!=numPoint-1){
			retPoints[retNum] = points[numPoint-1];
			retIndices[retNum] = numPoint-1;
			retNum++;
		}
		thresh*=0.8f;
		count++;
	}
	if(retNum<minNum){//evenly resample if still not enough point
		int interval = ((numPoint-2.0f)/(minNum-1)+0.5f);
		for (int i=1; i<minNum-1; ++i){
			retPoints[i] = points[i*interval];
			retIndices[i] = i*interval;
		}
		retPoints[minNum-1] = points[numPoint-1];
		retIndices[minNum-1] = numPoint-1;
		retNum = minNum;
	}
}

using namespace StreamlineResample;

template<typename index_type>
template<ResampleType resample_type>
void MultiPool<index_type>::resample(const float& thresh)
{
	//采样流线池初始化
	Pool<index_type>& resample_pool = resample_pools[(int)resample_type][thresh];
	resample_pool.org_pool = pool;
	resample_pool.streamlines.reserve(pool->streamlines.size());
	//根据采样类型进行采样
	for (int i = 0; i < pool->streamlines.size(); ++i) {
		const Streamline& s = pool->streamlines[i];
		//初始化单条流线，sid=0,start=resample_pool.points.size(),numPoint=0
		Streamline resample_line = makeStreamline(0, resample_pool.points.size(), 0);
		//曲率采样
		if (resample_type==acc_attrib){
			//流线第一个点的指针
			float* line_attribs = attribs.data() + s.start;
			//开始曲率采样
			resample(resample_pool.points, line_attribs, s.numPoint, thresh);
		} else if (resample_type==arc_length) {	//弧长采样
			//流线第一个点的指针
			vec3f* line_points = pool->points.data() + s.start;
			//开始弧长采样
			resample(resample_pool.points, line_points, s.numPoint, thresh);
		}
		//计算当前流线上的点的个数
		resample_line.numPoint = resample_pool.points.size() - resample_line.start;
		//保存流线上的点的line_ids
		resample_pool.line_ids.insert(resample_pool.line_ids.end(), resample_line.numPoint, i);
		//将流线加入采样流线池
		resample_pool.streamlines.push_back(resample_line);
	}
}
template void MultiPool<float>::resample<arc_length>(const float& thresh);
template void MultiPool<float>::resample<acc_attrib>(const float& thresh);
//弧长采样
template<>
template<>
void MultiPool<float>::resample<vec3f>(std::vector<float>& ret,
	vec3f* points, const int& num, const float& thresh)
{
	std::vector<float> line_resample_indices;
	line_resample_indices.reserve(num);
	line_resample_indices.push_back(0.0f);
	vec3f p = points[0];
	float di1 = 0.0f, di, fac;
	for (int i = 1; i < num; ++i) {
		di = length(p - points[i]);
		if (di > thresh) {
			fac = interpolate(di1, di, thresh, 0.0f, 1.0f);
			p = interpolate(points[i - 1], points[i], fac);
			line_resample_indices.push_back(i - 1 + fac);
			di1 = 0.0f;
		} else {
			di1 = di;
		}
	}
	ret.insert(ret.end(), line_resample_indices.begin(), line_resample_indices.end());
}
//曲率采样
template<>
template<>
void MultiPool<float>::resample<float>(std::vector<float>& ret,
	float* points, const int& num, const float& thresh)
{
	std::vector<float> line_resample_indices;
	line_resample_indices.reserve(num);
	line_resample_indices.push_back(0.0f);

	float remain = thresh, to_next;
	for (int i = 0; i < num-1; ++i) {
		to_next = points[i + 1];
		float f = 0.0f;
		while (to_next > remain) {
			f = interpolate(0.0f, to_next, remain, f, 1.0f);
			line_resample_indices.push_back(i+f);
			to_next -= remain;
			remain = thresh;
		}
		remain -= to_next;
	}
	if (std::abs(line_resample_indices.back() - (num - 1)) > 0.2f) {
		line_resample_indices.push_back(num - 1);
	}
	ret.insert(ret.end(), line_resample_indices.begin(), line_resample_indices.end());
}

template <>
void MultiPool<float>::getResampleLine(std::vector<vec3f>& ret, const int& streamline_id,
	const float& thresh, const ResampleType& resample_type)
{
	ret.clear();
	Pool<float>& resample_pool = *(getResamplePool(thresh, resample_type));
	Streamline& rs = resample_pool.streamlines[streamline_id];
	Streamline& s = pool->streamlines[streamline_id];
	vec3f* points = &pool->points[s.start];
	for (int i = rs.start; i < rs.start + rs.numPoint; ++i) {
		float resample_index = resample_pool.points[i];
		int int_index = (int)resample_index;
		if (int_index == s.numPoint - 1) {
			ret.push_back(points[s.numPoint - 1]);
		} else {
			float fac = resample_index - int_index;
			ret.push_back(interpolate(points[int_index], points[int_index + 1], fac));
		}
	}
}
