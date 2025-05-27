#ifndef TYPE_OPERATION_H
#define TYPE_OPERATION_H

#define USE_CUDA

#include <cmath>
#include <fstream>
#include <sstream>
#include <map>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cctype>
#include <GL/glew.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "helper_cuda.h"
#define CUDA_HOST_DEVICE	__host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

#define CUDA_WARP_SIZE	32

typedef struct{
	float x, y;
} vec2f;

typedef struct{
	union{
		struct{
			union{
				vec2f xy;
				struct {
					float x, y;
				};
			};
			float z;
		};
		struct{
			float r, g, b;
		};
	};
} vec3f;

typedef struct{
	float s[3];
} vec3fa;

typedef struct{
	double s[3];
} vec3da;

typedef struct{
	union{
		vec3f xyz;
		struct{
			union{
				vec2f xy;
				struct {
					float x, y;
				};
			};
			float z;
		};
		vec3f rgb;
		struct{
			float r, g, b;
		};
	};
	union{
		float w;
		float a;
	};
} vec4f;

typedef struct{
	float s[4];
} vec4fa;

typedef struct{
	int x, y;
} vec2i;

typedef struct{
	int x, y, z;
} vec3i;

typedef struct{
	int s[3];
} vec3ia;

typedef struct{
	int x, y, z, w;
} vec4i;

typedef struct{
	int s[4];
} vec4ia;

typedef struct{
	vec4f m[3];
} vec3x4f;


typedef struct{
	union{
		vec2f pos;
		struct {
			float x, y;
		};
	};
	union{
		vec2f size;
		struct{
			float w, h;
		};
	};
} Area;

enum NormScheme {
	MINMAX = 0,
	MAXMIN=1,
	UNIFORM,
	ACC,
	PERCENT95，
};

typedef struct{
	int lower, upper;
} IndexRange;

struct Range {
	Range() {
		lower = upper = 0.0f;
	}
	Range(const float& low, const float& up) {
		lower = low;
		upper = up;
	}
	float lower, upper;
};

typedef struct {
	float lower = 0.0f;
	float upper = 0.0f;
} coordinateRange;

typedef struct{
	int sid, start, numPoint;
} Streamline;

typedef struct{
	int id, start, numPoint;
} Pathline;

typedef struct{
	int sid;
	int pid;
} StreamlineClickInfo;

class StreamlinePool {
public:
	StreamlinePool(){}

	typedef struct {
		int streamline_id;
		int point_local_id;
	} PointInfo;


	inline int getStreamlineIdOfPoint(const int& point_global_id) {
		return line_ids[point_global_id];
	}

	inline Streamline getStreamlineOfPoint(const int& point_global_id) {
		return streamlines[line_ids[point_global_id]];
	}

	inline PointInfo getPointInfo(const int& point_global_id) {
		PointInfo ret;
		ret.streamline_id = line_ids[point_global_id];
		ret.point_local_id = point_global_id - streamlines[ret.streamline_id].start;
		return ret;
	}

	inline void fillLineIds() {
		line_ids.clear();
		line_ids.reserve(points.size());
		for (int i = 0; i < streamlines.size(); ++i) {
			line_ids.insert(line_ids.end(), streamlines[i].numPoint, i);
			streamlines[i].sid = i;
		}
	}

	inline int getNumPoints() {
		return points.size();
	}

	inline void getSeeds(std::vector<vec3f>& ret) {
		ret.reserve(streamlines.size());
		for (auto s : streamlines) {
			ret.push_back(points[s.start + s.sid]);
		}
	}

	std::vector<vec3f> points;
	std::vector<Streamline> streamlines;
	std::vector<int> line_ids;
};

typedef struct {
	int streamline_id;
	IndexRange segment;
} StreamlineSegment;

typedef struct{
	int vertex[2];
	int nbrGrid[4];
	int direction;
} EdgeInfoItem;

typedef struct{
	float val;
	int idx;
} sortElemInc;

typedef struct{
	int timeline_start, timeline_num;
	int vertex_start, vertex_num;
	int prim_start, prim_num;
} Surface;

inline CUDA_HOST_DEVICE Surface makeSurface(const int& timeline_start, const int& timeline_num, 
											const int& vertex_start, const int& vertex_num,
											const int& prim_start, const int& prim_num)
{
	Surface ret = {timeline_start, timeline_num, vertex_start, vertex_num, prim_start, prim_num};
	return ret;
}


inline CUDA_HOST_DEVICE sortElemInc makeSortElemInc(const float& val, const int& idx){
	sortElemInc ret = {val, idx};
	return ret;
}

inline CUDA_HOST_DEVICE bool operator < (const sortElemInc& a, const sortElemInc& b){
	return (a.val<b.val);
}

typedef struct{
	float val;
	int idx;
} sortElemDec;

inline CUDA_HOST_DEVICE sortElemDec makeSortElemDec(const float& val, const int& idx){
	sortElemDec ret = {val, idx};
	return ret;
}

inline CUDA_HOST_DEVICE bool operator < (const sortElemDec& a, const sortElemDec& b){
	return (a.val>b.val);
}

inline CUDA_HOST_DEVICE Streamline makeStreamline(const int& sid, const int& start, const int& num_points) {
	Streamline ret = { sid, start, num_points };
	return ret;
}

inline CUDA_HOST_DEVICE StreamlineSegment makeStreamlineSegment(const int& stl_id, const int& lower, const int& upper) {
	StreamlineSegment ret = { stl_id, lower, upper };
	return ret;
}

static CUDA_HOST_DEVICE void mergeSegments(std::vector<StreamlineSegment>& segs) {
	if (segs.empty()) return;

	std::vector<StreamlineSegment> sorted_segs;
	std::swap(sorted_segs, segs);
	std::sort(sorted_segs.begin(), sorted_segs.end());
	//merge segments on the same streamline
	StreamlineSegment seg = sorted_segs[0];
	for (int i = 1; i < sorted_segs.size(); ++i) {
		if (seg.streamline_id == sorted_segs[i].streamline_id && seg.segment.upper > sorted_segs[i].segment.lower) {//overlapped
			seg.segment.upper = std::max(seg.segment.upper, sorted_segs[i].segment.upper);
		} else {
			segs.push_back(seg);
			seg = sorted_segs[i];
		}
	}
	segs.push_back(seg);
}

inline bool operator<(const StreamlineSegment& a, const StreamlineSegment& b) {
	if (a.streamline_id == b.streamline_id) {
		if (a.segment.lower == b.segment.lower) {
			return (a.segment.upper < b.segment.upper);
		}
		return (a.segment.lower < b.segment.lower);
	}
	return (a.streamline_id < b.streamline_id);
}

inline CUDA_HOST_DEVICE IndexRange makeIndexRange(const int& lower, const int& upper){
	IndexRange ret = {lower, upper};
	return ret;
}

inline CUDA_HOST_DEVICE IndexRange outerRangeBound(const IndexRange& a, const IndexRange& b){
	if (a.lower<0) {
		return b;
	} else if (b.lower<0){
		return a;
	}
	IndexRange ret;
	ret.lower = (a.lower<b.lower)?a.lower:b.lower;
	ret.upper = (a.upper>b.upper)?a.upper:b.upper;
	return ret;
}

inline CUDA_HOST_DEVICE Range makeRange(const float& lower, const float& upper) {
	Range ret = { lower, upper };
	return ret;
}

inline CUDA_HOST_DEVICE Range outerRangeBound(const Range& a, const Range& b) {
	Range ret;
	ret.lower = std::min(a.lower, b.lower);
	ret.upper = std::max(a.upper, b.upper);
	return ret;
}

inline CUDA_HOST_DEVICE bool inRange(const Range& r, const float& val) {
	return (val >= r.lower && val <= r.upper);
}

inline CUDA_HOST_DEVICE bool inRange(const IndexRange& r, const int& val) {
	return (val >= r.lower && val <= r.upper);
}


inline CUDA_HOST_DEVICE IndexRange moveRangeInBound(const IndexRange& range, const IndexRange& bound) {
	if (range.upper - range.lower > bound.upper - bound.lower) {
		return bound;
	} else if (range.lower<bound.lower) {
		int diff = bound.lower - range.lower;
		return makeIndexRange(range.lower + diff, range.upper + diff);
	} else if (range.upper > bound.upper) {
		int diff = bound.upper - range.upper;
		return makeIndexRange(range.lower + diff, range.upper + diff);
	}
	return range;
}

inline CUDA_HOST_DEVICE Range moveRangeInBound(const Range& range, const Range& bound) {
	if (range.upper - range.lower > bound.upper - bound.lower) {
		return bound;
	} else if (range.lower < bound.lower) {
		float diff = bound.lower - range.lower;
		return makeRange(range.lower + diff, range.upper + diff);
	} else if (range.upper > bound.upper) {
		float diff = bound.upper - range.upper;
		return makeRange(range.lower + diff, range.upper + diff);
	}
	return range;
}

typedef enum {
	STREAMLINE_TRACE_VELOCITY = 0,
	STREAMLINE_TRACE_NORMAL,
	STREAMLINE_TRACE_BINORMAL
} StreamlineTraceDirectionType;

typedef struct _StreamlineTraceParameter {
	int max_streamline;
	int max_point;
	int min_point;
	int max_step;
	float segment_length;
	float trace_interval;
	int trace_type;
	int store_gap;

	_StreamlineTraceParameter() {
		max_streamline = 200;
		max_point = 500;
		min_point = 50;
		max_step = 5000;
		segment_length = 0.5f;
		trace_interval = 0.1f;
		trace_type = (int)STREAMLINE_TRACE_VELOCITY;
		store_gap = 1;
	}
} StreamlineTraceParameter;

typedef struct _PathlineTraceParameter{
	int max_pathline;
	int max_point;
	int min_point;
	int max_step;
	float segment_length;
	float trace_interval;
	float time_speed;

	_PathlineTraceParameter(){
		max_pathline = 200;
		max_point = 500;
		min_point = 50;
		max_step = 5000;
		segment_length = 0.5f;
		trace_interval = 0.1f;
		time_speed = 1.0f;
	}
} PathlineTraceParameter;

//conversions
inline vec3f makeVec3f(const vec3i& v){
	vec3f ret = {(float)v.x, (float)v.y, (float)v.z};
	return ret;
}

inline vec3i makeVec3i(const vec3f& v){
	vec3i ret = {(int)v.x, (int)v.y, (int)v.z};
	return ret;
}



//-----------------------------------------------------------------
//------------------    float type    -----------------------------
//-----------------------------------------------------------------
inline CUDA_HOST_DEVICE vec2f makeVec2f(const float &x, const float &y){
	vec2f ret = {x, y};
	return ret;
}

inline CUDA_HOST_DEVICE vec2f makeVec2f(const float& v) {
	vec2f ret = { v, v };
	return ret;
}

inline CUDA_HOST_DEVICE vec2f makeVec2f(const vec2i& v){
	vec2f ret = {v.x, v.y};
	return ret;
}

inline CUDA_HOST_DEVICE vec3f makeVec3f(const vec2f &xy, const float &z){
	vec3f ret = {xy.x, xy.y, z};
	return ret;
}

inline CUDA_HOST_DEVICE vec3f makeVec3f(const float &x, const float &y, const float &z){
	vec3f ret = {x, y, z};
	return ret;
}

inline CUDA_HOST_DEVICE vec3f makeVec3f(const float &v){
	vec3f ret = {v, v, v};
	return ret;
}

inline CUDA_HOST_DEVICE vec3f makeVec3f(const vec4f &v){
	vec3f ret = {v.x, v.y, v.z};
	return ret;
}

inline CUDA_HOST_DEVICE vec4f makeVec4f(const float &x, const float &y, const float &z, const float &w){
	vec4f ret = {x, y, z, w};
	return ret;
}

inline CUDA_HOST_DEVICE vec4f makeVec4f(const vec3f& xyz, const float &w){
	vec4f ret = {xyz.x, xyz.y, xyz.z, w};
	return ret;
}

inline CUDA_HOST_DEVICE vec4f makeVec4f(const float &v){
	vec4f ret = {v, v, v, v};
	return ret;
}

inline CUDA_HOST_DEVICE vec2f operator * (const float &scale, const vec2f &vec){
	vec2f ret = {scale*vec.x, scale*vec.y};
	return ret;
}

inline CUDA_HOST_DEVICE vec3f operator * (const float &scale, const vec3f &vec){
	vec3f ret = {scale*vec.x, scale*vec.y, scale*vec.z};
	return ret;
}

inline CUDA_HOST_DEVICE vec4f operator * (const float &scale, const vec4f &vec){
	vec4f ret = {scale*vec.x, scale*vec.y, scale*vec.z, scale*vec.w};
	return ret;
}

inline CUDA_HOST_DEVICE vec2f operator * (const vec2f &vec, const float &scale){
	vec2f ret = {scale*vec.x, scale*vec.y};
	return ret;
}

inline CUDA_HOST_DEVICE vec3f operator * (const vec3f &vec, const float &scale){
	vec3f ret = {scale*vec.x, scale*vec.y, scale*vec.z};
	return ret;
}

inline CUDA_HOST_DEVICE vec4f operator * (const vec4f &vec, const float &scale){
	vec4f ret = {scale*vec.x, scale*vec.y, scale*vec.z, scale*vec.w};
	return ret;
}
inline CUDA_HOST_DEVICE vec2f operator + (const vec2f &vec1, const vec2f &vec2){
	vec2f ret = {vec1.x+vec2.x, vec1.y+vec2.y};
	return ret;
}

inline CUDA_HOST_DEVICE vec3f operator + (const vec3f &vec1, const vec3f &vec2){
	vec3f ret = {vec1.x+vec2.x, vec1.y+vec2.y, vec1.z+vec2.z};
	return ret;
}

inline CUDA_HOST_DEVICE vec4f operator + (const vec4f &vec1, const vec4f &vec2){
	vec4f ret = {vec1.x+vec2.x, vec1.y+vec2.y, vec1.z+vec2.z, vec1.w+vec2.w};
	return ret;
}

inline CUDA_HOST_DEVICE vec2f operator - (const vec2f &vec1, const vec2f &vec2){
	vec2f ret = {vec1.x-vec2.x, vec1.y-vec2.y};
	return ret;
}

inline CUDA_HOST_DEVICE vec3f operator - (const vec3f &vec1, const vec3f &vec2){
	vec3f ret = {vec1.x-vec2.x, vec1.y-vec2.y, vec1.z-vec2.z};
	return ret;
}

inline CUDA_HOST_DEVICE vec4f operator - (const vec4f &vec1, const vec4f &vec2){
	vec4f ret = {vec1.x-vec2.x, vec1.y-vec2.y, vec1.z-vec2.z, vec1.w-vec2.w};
	return ret;
}

inline CUDA_HOST_DEVICE vec2f operator - (const vec2f &vec){
	vec2f ret = {-vec.x, -vec.y};
	return ret;
}

inline CUDA_HOST_DEVICE vec3f operator - (const vec3f &vec){
	vec3f ret = {-vec.x, -vec.y, -vec.z};
	return ret;
}

inline CUDA_HOST_DEVICE vec4f operator - (const vec4f &vec){
	vec4f ret = {-vec.x, -vec.y, -vec.z, -vec.w};
	return ret;
}

inline CUDA_HOST_DEVICE float operator * (const vec2f &vec1, const vec2f &vec2){
	return (vec1.x*vec2.x+vec1.y*vec2.y);
}

inline CUDA_HOST_DEVICE float operator * (const vec3f &vec1, const vec3f &vec2){
	return (vec1.x*vec2.x+vec1.y*vec2.y+vec1.z*vec2.z);
}

inline CUDA_HOST_DEVICE float operator * (const vec4f &vec1, const vec4f &vec2){
	return (vec1.x*vec2.x+vec1.y*vec2.y+vec1.z*vec2.z+vec1.w*vec2.w);
}


inline CUDA_HOST_DEVICE vec2f operator / (const vec2f &vec1, const vec2f &vec2){
	return makeVec2f(vec1.x/vec2.x, vec1.y/vec2.y);
}

inline CUDA_HOST_DEVICE vec3f operator / (const vec3f &vec1, const vec3f &vec2){
	return makeVec3f(vec1.x/vec2.x,vec1.y/vec2.y,vec1.z/vec2.z);
}

inline CUDA_HOST_DEVICE vec4f operator / (const vec4f &vec1, const vec4f &vec2){
	return makeVec4f(vec1.x/vec2.x,vec1.y/vec2.y,vec1.z/vec2.z,vec1.w/vec2.w);
}

inline CUDA_HOST_DEVICE vec2f operator / (const float &val, const vec2f &vec){
	return makeVec2f(val/vec.x,val/vec.y);
}

inline CUDA_HOST_DEVICE vec3f operator / (const float &val, const vec3f &vec){
	return makeVec3f(val/vec.x,val/vec.y,val/vec.z);
}

inline CUDA_HOST_DEVICE vec4f operator / (const float &val, const vec4f &vec){
	return makeVec4f(val/vec.x,val/vec.y,val/vec.z,val/vec.w);
}

inline CUDA_HOST_DEVICE vec2f operator / (const vec2f &vec, const float &val){
	return makeVec2f(vec.x/val,vec.y/val);
}

inline CUDA_HOST_DEVICE vec3f operator / (const vec3f &vec, const float &val){
	return makeVec3f(vec.x/val,vec.y/val,vec.z/val);
}

inline CUDA_HOST_DEVICE vec4f operator / (const vec4f &vec, const float &val){
	return makeVec4f(vec.x/val,vec.y/val,vec.z/val,vec.w/val);
}

inline CUDA_HOST_DEVICE bool operator < (const vec2f &vec1, const vec2f &vec2){
	if (vec1.x==vec2.x)
		return (vec1.y<vec2.y);
	return (vec1.x<vec2.x);
}

inline CUDA_HOST_DEVICE bool operator > (const vec2f &vec1, const vec2f &vec2){
	if (vec1.x==vec2.x)
		return (vec1.y>vec2.y);
	return (vec1.x>vec2.x);
}

inline CUDA_HOST_DEVICE bool operator < (const vec3f &vec1, const vec3f &vec2){
	if (vec1.x==vec2.x){
		if (vec1.y==vec2.y){
			return (vec1.z<vec2.z);
		}
		return (vec1.y<vec2.y);
	}
	return (vec1.x<vec2.x);
}

inline CUDA_HOST_DEVICE bool operator > (const vec3f &vec1, const vec3f &vec2){
	if (vec1.x==vec2.x){
		if (vec1.y==vec2.y){
			return (vec1.z>vec2.z);
		}
		return (vec1.y>vec2.y);
	}
	return (vec1.x>vec2.x);
}

inline CUDA_HOST_DEVICE void operator += (vec2f &vec1, const vec2f &vec2){
	vec1.x += vec2.x;
	vec1.y += vec2.y;
}

inline CUDA_HOST_DEVICE void operator += (vec3f &vec1, const vec3f &vec2){
	vec1.x += vec2.x;
	vec1.y += vec2.y;
	vec1.z += vec2.z;
}

inline CUDA_HOST_DEVICE void operator += (vec4f &vec1, const vec4f &vec2){
	vec1.x += vec2.x;
	vec1.y += vec2.y;
	vec1.z += vec2.z;
	vec1.w += vec2.w;
}

inline CUDA_HOST_DEVICE void operator -= (vec2f &vec1, const vec2f &vec2){
	vec1.x -= vec2.x;
	vec1.y -= vec2.y;
}

inline CUDA_HOST_DEVICE void operator -= (vec3f &vec1, const vec3f &vec2){
	vec1.x -= vec2.x;
	vec1.y -= vec2.y;
	vec1.z -= vec2.z;
}

inline CUDA_HOST_DEVICE void operator -= (vec4f &vec1, const vec4f &vec2){
	vec1.x -= vec2.x;
	vec1.y -= vec2.y;
	vec1.z -= vec2.z;
	vec1.w -= vec2.w;
}

inline CUDA_HOST_DEVICE void operator *= (vec2f &vec1, const float& val){
	vec1.x *= val;
	vec1.y *= val;
}

inline CUDA_HOST_DEVICE void operator *= (vec3f &vec1, const float& val){
	vec1.x *= val;
	vec1.y *= val;
	vec1.z *= val;
}

inline CUDA_HOST_DEVICE void operator *= (vec4f &vec1, const float& val){
	vec1.x *= val;
	vec1.y *= val;
	vec1.z *= val;
	vec1.w *= val;
}

inline CUDA_HOST_DEVICE void normalize(vec2f &vec){
	float length = sqrt(vec*vec);
	vec.x /= length;
	vec.y /= length;
}

inline CUDA_HOST_DEVICE void normalize(vec3f &vec){
	float length = sqrt(vec*vec);
	vec.x /= length;
	vec.y /= length;
	vec.z /= length;
}

inline CUDA_HOST_DEVICE void normalize(vec4f &vec){
	float length = sqrt(vec*vec);
	vec.x /= length;
	vec.y /= length;
	vec.z /= length;
	vec.w /= length;
}

template<typename T>
inline CUDA_HOST_DEVICE T unit_vec(const T& vec) {
	float fac = 1.0f / sqrtf(vec * vec);
	return (fac * vec);
}

//project a on b
template<typename T>
inline CUDA_HOST_DEVICE T project_vec(const T& a, const T& b) {
	float len = a * b;
	return len * b;
}

inline CUDA_HOST_DEVICE float length(vec2f &vec){
	return sqrt(vec*vec);
}

inline CUDA_HOST_DEVICE float length(vec3f &vec){
	return sqrt(vec*vec);
}

inline CUDA_HOST_DEVICE float length(vec4f &vec){
	return sqrt(vec*vec);
}
inline CUDA_HOST_DEVICE float length(const vec2f& vec) {
	return sqrt(vec * vec);
}

inline CUDA_HOST_DEVICE float length(const vec3f& vec) {
	return sqrt(vec * vec);
}

inline CUDA_HOST_DEVICE float length(const vec4f& vec) {
	return sqrt(vec * vec);
}


inline CUDA_HOST_DEVICE float dist(const vec2f &vec1, const vec2f &vec2){
	return length(vec1-vec2);
}

inline CUDA_HOST_DEVICE float dist(const vec3f &vec1, const vec3f &vec2){
	return length(vec1-vec2);
}


inline CUDA_HOST_DEVICE float dist(const vec4f &vec1, const vec4f &vec2){
	return length(vec1-vec2);
}

inline CUDA_HOST_DEVICE bool isZero(vec3f & vec){
	return ((vec.x<0.000001f && vec.x>-0.000001f) 
		&& (vec.y<0.000001f && vec.y>-0.000001f) && (vec.z<0.000001f && vec.z>-0.000001f));
}

inline CUDA_HOST_DEVICE vec3f cross(const vec3f &vec1, const vec3f &vec2){
	vec3f ret = {vec1.y*vec2.z-vec1.z*vec2.y, vec1.z*vec2.x-vec1.x*vec2.z, vec1.x*vec2.y-vec1.y*vec2.x};
	return ret;
}

inline CUDA_HOST_DEVICE float det(const vec3f &vec1, const vec3f &vec2, const vec3f &vec3){
	return (vec1.x*vec2.y*vec3.z+vec1.y*vec2.z*vec3.x+vec1.z*vec2.x*vec3.y
		-vec1.x*vec2.z*vec3.y-vec1.y*vec2.x*vec3.z-vec1.z*vec2.y*vec3.x);
}

inline CUDA_HOST_DEVICE bool operator == (const vec2f &vec1, const vec2f &vec2){
	return ((vec1.x==vec2.x)&&(vec1.y==vec2.y));
}

inline CUDA_HOST_DEVICE bool operator == (const vec3f &vec1, const vec3f &vec2){
	return ((vec1.x==vec2.x)&&(vec1.y==vec2.y)&&(vec1.z==vec2.z));
}

inline CUDA_HOST_DEVICE bool operator == (const vec4f &vec1, const vec4f &vec2){
	return ((vec1.x==vec2.x)&&(vec1.y==vec2.y)&&(vec1.z==vec2.z)&&(vec1.w==vec2.w));
}

inline CUDA_HOST_DEVICE bool operator != (const vec2f &vec1, const vec2f &vec2){
	return ((vec1.x!=vec2.x)||(vec1.y!=vec2.y));
}

inline CUDA_HOST_DEVICE bool operator != (const vec3f &vec1, const vec3f &vec2){
	return ((vec1.x!=vec2.x)||(vec1.y!=vec2.y)||(vec1.z!=vec2.z));
}

inline CUDA_HOST_DEVICE bool operator != (const vec4f &vec1, const vec4f &vec2){
	return ((vec1.x!=vec2.x)||(vec1.y!=vec2.y)||(vec1.z!=vec2.z)||(vec1.w!=vec2.w));
}

inline CUDA_HOST_DEVICE float clamp(float v, const float &low, const float &up){
	v = (v>low)?v:low;
	v = (v<up)?v:up;
	return v;
}

inline CUDA_HOST_DEVICE vec2f clamp(vec2f v, const vec2f &low, const vec2f &up){
	v.x = clamp(v.x, low.x, up.x);
	v.y = clamp(v.y, low.y, up.y);
	return v;
}

inline CUDA_HOST_DEVICE vec3f clamp(vec3f v, const vec3f &low, const vec3f &up){
	v.x = clamp(v.x, low.x, up.x);
	v.y = clamp(v.y, low.y, up.y);
	v.z = clamp(v.z, low.z, up.z);
	return v;
}

inline CUDA_HOST_DEVICE vec4f clamp(vec4f v, const vec4f &low, const vec4f &up){
	v.x = clamp(v.x, low.x, up.x);
	v.y = clamp(v.y, low.y, up.y);
	v.z = clamp(v.z, low.z, up.z);
	v.w = clamp(v.w, low.w, up.w);
	return v;
}

inline CUDA_HOST_DEVICE bool inBound(const float& v, const float& low, const float& up){
	return (v>=low && v<=up);
}

inline CUDA_HOST_DEVICE bool inBound(const vec2f& v, const vec2f& low, const vec2f& up){
	return ((v.x>=low.x)&&(v.y>=low.y)&&(v.x<=up.x)&&(v.y<=up.y));
}

inline CUDA_HOST_DEVICE bool inBound(const vec3f& v, const vec3f& low, const vec3f& up){
	return ((v.x>=low.x)&&(v.y>=low.y)&&(v.z>=low.z)&&(v.x<=up.x)&&(v.y<=up.y)&&(v.z<=up.z));
}

inline CUDA_HOST_DEVICE bool inBound(const vec4f& v, const vec4f& low, const vec4f& up){
	return ((v.x>=low.x)&&(v.y>=low.y)&&(v.z>=low.z)&&(v.w>=low.w)&&(v.x<=up.x)&&(v.y<=up.y)&&(v.z<=up.z)&&(v.w<=up.w));
}


inline CUDA_HOST_DEVICE Area makeArea(const float& x, const float& y, const float& w, const float& h){
	Area ret = {x, y, w, h};
	return ret;
}

struct RectDisplayArea {
	vec2f origin;
	vec2f row_axis;
	vec2f col_axis;

	inline vec2f display2norm(const vec2f& p) const {
		vec2f rp = p - origin;
		vec2f ret;
		ret.x = rp * row_axis / (row_axis * row_axis);
		ret.y = rp * col_axis / (col_axis * col_axis);
		return ret;
	}

	inline vec2f norm2display(const vec2f& n) const {
		return origin + n.x * row_axis + n.y * col_axis;
	}

	inline bool is_in(const vec2f& p) const {
		vec2f n = display2norm(p);
		return (n.x >= 0.0f && n.x <= 1.0f && n.y >= 0.0f && n.y <= 1.0f);
	}

	inline void scale(const float& s) {
		row_axis *= s;
		col_axis *= s;
	}
};

inline RectDisplayArea makeRectDisplayArea(const vec2f& orig, const vec2f& row, const vec2f& col) {
	RectDisplayArea ret = { orig, row, col };
	return ret;
}

inline RectDisplayArea makeRectDisplayArea(const float& org_x, const float& org_y, 
	const float& row_x, const float& row_y, const float& col_x, const float& col_y)
{
	RectDisplayArea ret = {org_x, org_y, row_x, row_y, col_x, col_y};
	return ret;
}

inline vec2f getNormalizeCoordInRect(const vec2f& p, const RectDisplayArea& area) {
	vec2f rp = p - area.origin;
	vec2f ret;
	ret.x = rp*area.row_axis / (area.row_axis*area.row_axis);
	ret.y = rp*area.col_axis / (area.col_axis*area.col_axis);
	return ret;
}

inline bool inRectDisplayArea(const vec2f& p, const RectDisplayArea& area) {
	vec2f n = getNormalizeCoordInRect(p, area);
	return (n.x >= 0.0f && n.x <= 1.0f && n.y >= 0.0f && n.y <= 1.0f);
}


inline CUDA_HOST_DEVICE bool isOverlapped(const Area& a, const Area& b){
	if ((a.x>b.x+b.w) || (b.x>a.x+a.w) || (a.y>b.y+b.h) || (b.y>a.y+a.h))
		return false;
	return true;
}

inline CUDA_HOST_DEVICE bool isOverlapped(const Area& a, const Area& b, const float& margin){
	if ((a.x-margin>b.x+b.w) || (b.x-margin>a.x+a.w) || (a.y-margin>b.y+b.h) || (b.y-margin>a.y+a.h))
		return false;
	return true;
}

inline CUDA_HOST_DEVICE bool isContained(const Area& a, const Area& b){
	if ((a.x>b.x) && (a.y>b.y) && (a.x+a.w<b.x+b.w) && (a.y+a.h<b.y+b.h))
		return true;
	return false;
}

inline CUDA_HOST_DEVICE bool isContained(const Area& a, const vec2f& p){
	if (a.x<=p.x && a.x+a.w>=p.x && a.y<=p.y && a.y+a.h>=p.y)
		return true;
	return false;
}


inline CUDA_HOST_DEVICE bool isContained(const vec2f& lb, const vec2f& rt, const vec2f& p){
	if (lb.x<=p.x && lb.y<=p.y && rt.x>=p.x && rt.y>=p.y)
		return true;
	return false;
}

//-----------------------------------------------------------------
//------------------     int  type    -----------------------------
//-----------------------------------------------------------------
inline CUDA_HOST_DEVICE vec3i makeVec3i(const int &x, const int &y, const int &z){
	vec3i ret = {x, y, z};
	return ret;
}

inline CUDA_HOST_DEVICE vec3i operator * (const int &scale, const vec3i &vec){
	vec3i ret = {scale*vec.x, scale*vec.y, scale*vec.z};
	return ret;
}

inline CUDA_HOST_DEVICE vec3i operator * (const vec3i &vec, const int &scale){
	vec3i ret = {scale*vec.x, scale*vec.y, scale*vec.z};
	return ret;
}

inline CUDA_HOST_DEVICE vec3i operator + (const vec3i &vec1, const vec3i &vec2){
	vec3i ret = {vec1.x+vec2.x, vec1.y+vec2.y, vec1.z+vec2.z};
	return ret;
}

inline CUDA_HOST_DEVICE vec3i operator - (const vec3i &vec1, const vec3i &vec2){
	vec3i ret = {vec1.x-vec2.x, vec1.y-vec2.y, vec1.z-vec2.z};
	return ret;
}

inline CUDA_HOST_DEVICE vec3i operator - (const vec3i &vec){
	vec3i ret = {-vec.x, -vec.y, -vec.z};
	return ret;
}

inline CUDA_HOST_DEVICE int operator * (const vec3i &vec1, const vec3i &vec2){
	return (vec1.x*vec2.x+vec1.y*vec2.y+vec1.z*vec2.z);
}

inline CUDA_HOST_DEVICE bool operator == (const vec3i &vec1, const vec3i &vec2){
	return ((vec1.x==vec2.x)&&(vec1.y==vec2.y)&&(vec1.z==vec2.z));
}

inline CUDA_HOST_DEVICE bool operator != (const vec3i &vec1, const vec3i &vec2){
	return ((vec1.x!=vec2.x)||(vec1.y!=vec2.y)||(vec1.z!=vec2.z));
}

inline CUDA_HOST_DEVICE bool operator < (const vec3i &vec1, const vec3i &vec2){
	if (vec1.x==vec2.x){
		if (vec1.y==vec2.y){
			return (vec1.z<vec2.z);
		}
		return (vec1.y<vec2.y);
	}
	return (vec1.x<vec2.x);
}

inline CUDA_HOST_DEVICE bool operator > (const vec3i &vec1, const vec3i &vec2){
	if (vec1.x==vec2.x){
		if (vec1.y==vec2.y){
			return (vec1.z>vec2.z);
		}
		return (vec1.y>vec2.y);
	}
	return (vec1.x>vec2.x);
}

//2i
inline CUDA_HOST_DEVICE vec2i makeVec2i(const int &x, const int &y){
	vec2i ret = {x, y};
	return ret;
}

inline CUDA_HOST_DEVICE vec2i operator * (const int &scale, const vec2i &vec){
	vec2i ret = {scale*vec.x, scale*vec.y};
	return ret;
}

inline CUDA_HOST_DEVICE vec2i operator * (const vec2i &vec, const int &scale){
	vec2i ret = {scale*vec.x, scale*vec.y};
	return ret;
}

inline CUDA_HOST_DEVICE vec2i operator + (const vec2i &vec1, const vec2i &vec2){
	vec2i ret = {vec1.x+vec2.x, vec1.y+vec2.y};
	return ret;
}

inline CUDA_HOST_DEVICE vec2i operator - (const vec2i &vec1, const vec2i &vec2){
	vec2i ret = {vec1.x-vec2.x, vec1.y-vec2.y};
	return ret;
}

inline CUDA_HOST_DEVICE vec2i operator - (const vec2i &vec){
	vec2i ret = {-vec.x, -vec.y};
	return ret;
}

inline CUDA_HOST_DEVICE int operator * (const vec2i &vec1, const vec2i &vec2){
	return (vec1.x*vec2.x+vec1.y*vec2.y);
}

inline CUDA_HOST_DEVICE bool operator == (const vec2i &vec1, const vec2i &vec2){
	return ((vec1.x==vec2.x)&&(vec1.y==vec2.y));
}

inline CUDA_HOST_DEVICE bool operator != (const vec2i &vec1, const vec2i &vec2){
	return ((vec1.x!=vec2.x)||(vec1.y!=vec2.y));
}

inline CUDA_HOST_DEVICE bool operator < (const vec2i &vec1, const vec2i &vec2){
	if (vec1.x==vec2.x)
		return (vec1.y<vec2.y);
	return (vec1.x<vec2.x);
}

inline CUDA_HOST_DEVICE bool operator > (const vec2i &vec1, const vec2i &vec2){
	if (vec1.x==vec2.x)
		return (vec1.y>vec2.y);
	return (vec1.x>vec2.x);
}

//4i
inline CUDA_HOST_DEVICE vec4i makeVec4i(const int &x, const int &y, const int &z, const int &w){
	vec4i ret = {x, y, z, w};
	return ret;
}

inline CUDA_HOST_DEVICE vec4i operator * (const int &scale, const vec4i &vec){
	vec4i ret = {scale*vec.x, scale*vec.y, scale*vec.z, scale*vec.w};
	return ret;
}

inline CUDA_HOST_DEVICE vec4i operator * (const vec4i &vec, const int &scale){
	vec4i ret = {scale*vec.x, scale*vec.y, scale*vec.z, scale*vec.w};
	return ret;
}

inline CUDA_HOST_DEVICE vec4i operator + (const vec4i &vec1, const vec4i &vec2){
	vec4i ret = {vec1.x+vec2.x, vec1.y+vec2.y, vec1.z+vec2.z, vec1.w+vec2.w};
	return ret;
}

inline CUDA_HOST_DEVICE vec4i operator - (const vec4i &vec1, const vec4i &vec2){
	vec4i ret = {vec1.x-vec2.x, vec1.y-vec2.y, vec1.z-vec2.z, vec1.w-vec2.w};
	return ret;
}

inline CUDA_HOST_DEVICE vec4i operator - (const vec4i &vec){
	vec4i ret = {-vec.x, -vec.y, -vec.z, -vec.w};
	return ret;
}

inline CUDA_HOST_DEVICE int operator * (const vec4i &vec1, const vec4i &vec2){
	return (vec1.x*vec2.x+vec1.y*vec2.y+vec1.z*vec2.z+vec1.w*vec2.w);
}

inline CUDA_HOST_DEVICE bool operator == (const vec4i &vec1, const vec4i &vec2){
	return ((vec1.x==vec2.x)&&(vec1.y==vec2.y)&&(vec1.z==vec2.z)&&(vec1.w==vec2.w));
}

inline CUDA_HOST_DEVICE bool operator != (const vec4i &vec1, const vec4i &vec2){
	return ((vec1.x!=vec2.x)||(vec1.y!=vec2.y)||(vec1.z!=vec2.z)||(vec1.w!=vec2.w));
}

inline CUDA_HOST_DEVICE bool operator < (const vec4i &vec1, const vec4i &vec2){
	if (vec1.x==vec2.x){
		if (vec1.y==vec2.y){
			if (vec1.z==vec2.z) {
				return (vec1.w<vec2.w);
			}
			return (vec1.z<vec2.z);
		}
		return (vec1.y<vec2.y);
	}
	return (vec1.x<vec2.x);
}

inline CUDA_HOST_DEVICE bool operator > (const vec4i &vec1, const vec4i &vec2){
	if (vec1.x==vec2.x){
		if (vec1.y==vec2.y){
			if (vec1.z==vec2.z) {
				return (vec1.w>vec2.w);
			}
			return (vec1.z>vec2.z);
		}
		return (vec1.y>vec2.y);
	}
	return (vec1.x>vec2.x);
}

inline CUDA_HOST_DEVICE void operator += (vec2i &vec1, const vec2i &vec2){
	vec1.x += vec2.x;
	vec1.y += vec2.y;
}

inline CUDA_HOST_DEVICE void operator += (vec3i &vec1, const vec3i &vec2){
	vec1.x += vec2.x;
	vec1.y += vec2.y;
	vec1.z += vec2.z;
}

inline CUDA_HOST_DEVICE void operator += (vec4i &vec1, const vec4i &vec2){
	vec1.x += vec2.x;
	vec1.y += vec2.y;
	vec1.z += vec2.z;
	vec1.w += vec2.w;
}

inline CUDA_HOST_DEVICE void operator -= (vec2i &vec1, const vec2i &vec2){
	vec1.x -= vec2.x;
	vec1.y -= vec2.y;
}

inline CUDA_HOST_DEVICE void operator -= (vec3i &vec1, const vec3i &vec2){
	vec1.x -= vec2.x;
	vec1.y -= vec2.y;
	vec1.z -= vec2.z;
}

inline CUDA_HOST_DEVICE void operator -= (vec4i &vec1, const vec4i &vec2){
	vec1.x -= vec2.x;
	vec1.y -= vec2.y;
	vec1.z -= vec2.z;
	vec1.w -= vec2.w;
}

inline CUDA_HOST_DEVICE float random_float(const float& up) {//in the range of [0,up]
	return (rand() / (RAND_MAX / up));
}

inline CUDA_HOST_DEVICE int clamp(int v, const int &low, const int &up){
	v = (v>low)?v:low;
	v = (v<up)?v:up;
	return v;
}

inline CUDA_HOST_DEVICE vec2i clamp(vec2i v, const vec2i &low, const vec2i &up){
	v.x = clamp(v.x, low.x, up.x);
	v.y = clamp(v.y, low.y, up.y);
	return v;
}

inline CUDA_HOST_DEVICE vec3i clamp(vec3i v, const vec3i &low, const vec3i &up){
	v.x = clamp(v.x, low.x, up.x);
	v.y = clamp(v.y, low.y, up.y);
	v.z = clamp(v.z, low.z, up.z);
	return v;
}

inline CUDA_HOST_DEVICE vec4i clamp(vec4i v, const vec4i &low, const vec4i &up){
	v.x = clamp(v.x, low.x, up.x);
	v.y = clamp(v.y, low.y, up.y);
	v.z = clamp(v.z, low.z, up.z);
	v.w = clamp(v.w, low.w, up.w);
	return v;
}

inline CUDA_HOST_DEVICE bool inBound(const int& v, const int& low, const int& up){
	return (v>=low && v<=up);
}

inline CUDA_HOST_DEVICE bool inBound(const vec2i& v, const vec2i& low, const vec2i& up){
	return ((v.x>=low.x)&&(v.y>=low.y)&&(v.x<=up.x)&&(v.y<=up.y));
}

inline CUDA_HOST_DEVICE bool inBound(const vec3i& v, const vec3i& low, const vec3i& up){
	return ((v.x>=low.x)&&(v.y>=low.y)&&(v.z>=low.z)&&(v.x<=up.x)&&(v.y<=up.y)&&(v.z<=up.z));
}

inline CUDA_HOST_DEVICE bool inBound(const vec4i& v, const vec4i& low, const vec4i& up){
	return ((v.x>=low.x)&&(v.y>=low.y)&&(v.z>=low.z)&&(v.w>=low.w)&&(v.x<=up.x)&&(v.y<=up.y)&&(v.z<=up.z)&&(v.w<=up.w));
}

//other vector functions
#define vec2dDot(v1, v2) ((v1).x*(v2).x+(v1).y*(v2).y)
#define vec3fDot(v1, v2) ((v1).x*(v2).x+(v1).y*(v2).y+(v1).z*(v2).z)
#define vec2dLen(v)	sqrtf((v).x*(v).x+(v).y*(v).y)
#define vec3fLen(v) sqrtf((v).x*(v).x+(v).y*(v).y+(v).z*(v).z)
#define dist2d(v1, v2) sqrtf((v1-v2)*(v1-v2))
#define dist3d(v1, v2) sqrtf((v1-v2)*(v1-v2))

//other functions
#define maxOf(a, b) (((a)>(b))?(a):(b))
#define minOf(a, b) (((a)<(b))?(a):(b))

inline void glColor(const vec4f& color) {
	glColor4f(color.x, color.y, color.z, color.w);
}

inline void glVertex(const vec2f& p) {
	glVertex2f(p.x, p.y);
}

inline void glVertex(const vec3f& p) {
	glVertex3f(p.x, p.y, p.z);
}

inline void glTexCoord(const vec2f& p) {
	glTexCoord2f(p.x, p.y);
}

inline void glNormal(const vec3f& n) {
	glNormal3f(n.x, n.y, n.z);
}

inline void glLookAt(const vec3f& eye, const vec3f& target, const vec3f& up) {
	gluLookAt(eye.x, eye.y, eye.z, target.x, target.y, target.z, up.x, up.y, up.z);
}

inline void glTranslate(const vec3f& t) {
	glTranslatef(t.x, t.y, t.z);
}

inline void glTranslate(const vec2f& t) {
	glTranslatef(t.x, t.y, 0.0f);
}



inline CUDA_HOST_DEVICE int iDivUp(int a, int b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

template <class T>
inline CUDA_HOST_DEVICE void computeMinMax(T* data, int num, T& minv, T& maxv){
	minv = data[0];
	maxv = data[0];
	for (int i=1; i<num; ++i){
		if (minv>data[i]) minv = data[i];
		if (maxv<data[i]) maxv = data[i];
	}
}



static CUDA_HOST_DEVICE bool getInvertMatrix(const float m[16], float invOut[16])
{
	float inv[16], det;
	int i;

	inv[0] = m[5]  * m[10] * m[15] - 
		m[5]  * m[11] * m[14] - 
		m[9]  * m[6]  * m[15] + 
		m[9]  * m[7]  * m[14] +
		m[13] * m[6]  * m[11] - 
		m[13] * m[7]  * m[10];

	inv[4] = -m[4]  * m[10] * m[15] + 
		m[4]  * m[11] * m[14] + 
		m[8]  * m[6]  * m[15] - 
		m[8]  * m[7]  * m[14] - 
		m[12] * m[6]  * m[11] + 
		m[12] * m[7]  * m[10];

	inv[8] = m[4]  * m[9] * m[15] - 
		m[4]  * m[11] * m[13] - 
		m[8]  * m[5] * m[15] + 
		m[8]  * m[7] * m[13] + 
		m[12] * m[5] * m[11] - 
		m[12] * m[7] * m[9];

	inv[12] = -m[4]  * m[9] * m[14] + 
		m[4]  * m[10] * m[13] +
		m[8]  * m[5] * m[14] - 
		m[8]  * m[6] * m[13] - 
		m[12] * m[5] * m[10] + 
		m[12] * m[6] * m[9];

	inv[1] = -m[1]  * m[10] * m[15] + 
		m[1]  * m[11] * m[14] + 
		m[9]  * m[2] * m[15] - 
		m[9]  * m[3] * m[14] - 
		m[13] * m[2] * m[11] + 
		m[13] * m[3] * m[10];

	inv[5] = m[0]  * m[10] * m[15] - 
		m[0]  * m[11] * m[14] - 
		m[8]  * m[2] * m[15] + 
		m[8]  * m[3] * m[14] + 
		m[12] * m[2] * m[11] - 
		m[12] * m[3] * m[10];

	inv[9] = -m[0]  * m[9] * m[15] + 
		m[0]  * m[11] * m[13] + 
		m[8]  * m[1] * m[15] - 
		m[8]  * m[3] * m[13] - 
		m[12] * m[1] * m[11] + 
		m[12] * m[3] * m[9];

	inv[13] = m[0]  * m[9] * m[14] - 
		m[0]  * m[10] * m[13] - 
		m[8]  * m[1] * m[14] + 
		m[8]  * m[2] * m[13] + 
		m[12] * m[1] * m[10] - 
		m[12] * m[2] * m[9];

	inv[2] = m[1]  * m[6] * m[15] - 
		m[1]  * m[7] * m[14] - 
		m[5]  * m[2] * m[15] + 
		m[5]  * m[3] * m[14] + 
		m[13] * m[2] * m[7] - 
		m[13] * m[3] * m[6];

	inv[6] = -m[0]  * m[6] * m[15] + 
		m[0]  * m[7] * m[14] + 
		m[4]  * m[2] * m[15] - 
		m[4]  * m[3] * m[14] - 
		m[12] * m[2] * m[7] + 
		m[12] * m[3] * m[6];

	inv[10] = m[0]  * m[5] * m[15] - 
		m[0]  * m[7] * m[13] - 
		m[4]  * m[1] * m[15] + 
		m[4]  * m[3] * m[13] + 
		m[12] * m[1] * m[7] - 
		m[12] * m[3] * m[5];

	inv[14] = -m[0]  * m[5] * m[14] + 
		m[0]  * m[6] * m[13] + 
		m[4]  * m[1] * m[14] - 
		m[4]  * m[2] * m[13] - 
		m[12] * m[1] * m[6] + 
		m[12] * m[2] * m[5];

	inv[3] = -m[1] * m[6] * m[11] + 
		m[1] * m[7] * m[10] + 
		m[5] * m[2] * m[11] - 
		m[5] * m[3] * m[10] - 
		m[9] * m[2] * m[7] + 
		m[9] * m[3] * m[6];

	inv[7] = m[0] * m[6] * m[11] - 
		m[0] * m[7] * m[10] - 
		m[4] * m[2] * m[11] + 
		m[4] * m[3] * m[10] + 
		m[8] * m[2] * m[7] - 
		m[8] * m[3] * m[6];

	inv[11] = -m[0] * m[5] * m[11] + 
		m[0] * m[7] * m[9] + 
		m[4] * m[1] * m[11] - 
		m[4] * m[3] * m[9] - 
		m[8] * m[1] * m[7] + 
		m[8] * m[3] * m[5];

	inv[15] = m[0] * m[5] * m[10] - 
		m[0] * m[6] * m[9] - 
		m[4] * m[1] * m[10] + 
		m[4] * m[2] * m[9] + 
		m[8] * m[1] * m[6] - 
		m[8] * m[2] * m[5];

	det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

	if (det == 0)
		return false;

	det = 1.0f / det;

	for (i = 0; i < 16; i++)
		invOut[i] = inv[i] * det;

	return true;
}

template <class T>
inline CUDA_HOST_DEVICE void mat_mult(T** m, T* v, T* ret, const int& n){
	for (int i=0; i<n; ++i) {
		ret[i] = 0;
		for (int j=0; j<n; ++j) {
			ret[i] += m[i][j]*v[j];
		}
	}
}

template <class T>
inline CUDA_HOST_DEVICE T inner(T* v1, T* v2, const int& n){
	T ret= 0.0f;
	for (int i=0; i<n; ++i) {
		ret += v1[i]*v2[i];
	}
	return ret;
}

template<typename T>
inline CUDA_HOST_DEVICE T interpolate(const T& lower, const T& upper, float fac){
	fac = clamp(fac, 0.0f, 1.0f);
	return ((1.0f-fac)*lower+fac*upper);
}

template<typename T>
inline CUDA_HOST_DEVICE T interpolate(const float& a1, const float& b1, const float& c1, const T& a2, const T& b2){
	if (a1==b1) {
		return a2;
	}
	float fac = (c1 - a1) / (b1 - a1);
	return ((1.0f-fac)*a2+fac*b2);
}

inline bool onSegment(const vec2f& p, const vec2f& q, const vec2f& r){
	if (q.x <= maxOf(p.x, r.x) && q.x >= minOf(p.x, r.x) &&
		q.y <= maxOf(p.y, r.y) && q.y >= minOf(p.y, r.y))
		return true;

	return false;
}

inline int orientation(const vec2f& p, const vec2f& q, const vec2f& r){
	int val = (q.y - p.y) * (r.x - q.x) -
		(q.x - p.x) * (r.y - q.y);

	if (val == 0) return 0;  // colinear

	return (val > 0)? 1: 2; // clock or counterclock wise
}

static bool segmentIntersected(const vec2f& p1, const vec2f& p2, const vec2f& q1, const vec2f& q2){

	int o1 = orientation(p1, q1, p2);
	int o2 = orientation(p1, q1, q2);
	int o3 = orientation(p2, q2, p1);
	int o4 = orientation(p2, q2, q1);

	if (o1 != o2 && o3 != o4)
		return true;

	if (o1 == 0 && onSegment(p1, p2, q1)) return true;
	if (o2 == 0 && onSegment(p1, q2, q1)) return true;
	if (o3 == 0 && onSegment(p2, p1, q2)) return true;
	if (o4 == 0 && onSegment(p2, q1, q2)) return true;

	return false;
}

static CUDA_HOST_DEVICE float determinantOfColumnMatrix(const vec3f mat[3]){
	return (mat[0].x*mat[1].y*mat[2].z+mat[1].x*mat[2].y*mat[0].z+mat[2].x*mat[0].y*mat[1].z
		-mat[2].x*mat[1].y*mat[0].z-mat[1].x*mat[0].y*mat[2].z-mat[0].x*mat[2].y*mat[1].z);
}

static CUDA_HOST_DEVICE float determinantOfColumnMatrix(const vec3f& v1, const vec3f& v2, const vec3f& v3){
	return (v1.x*v2.y*v3.z+v2.x*v3.y*v1.z+v3.x*v1.y*v2.z
		-v3.x*v2.y*v1.z-v2.x*v1.y*v3.z-v1.x*v3.y*v2.z);
}


static CUDA_HOST_DEVICE void transformVector(const float m[16], const float v[4], float ret[4]){
	ret[0] = m[0]*v[0]+m[4]*v[1]+m[ 8]*v[2]+m[12]*v[3];
	ret[1] = m[1]*v[0]+m[5]*v[1]+m[ 9]*v[2]+m[12]*v[3];
	ret[2] = m[2]*v[0]+m[6]*v[1]+m[10]*v[2]+m[12]*v[3];
	ret[3] = m[3]*v[0]+m[7]*v[1]+m[11]*v[2]+m[12]*v[3];
}

template<typename T>
inline int getItemIndex(std::map<T,int>& id_map, const T& item){
	typename std::map<T,int>::iterator it = id_map.find(item);
	if (it!=id_map.end()) {
		return it->second;
	}
	int ret = id_map.size();
	id_map.insert(std::pair<T,int>(item, ret));
	return ret;
}

template <class T>
inline void allocateMatrix(T*& data, T**& mat, const int& w, const int& h){
	int wh = w*h;

	data = new T[wh];
	mat = new T*[h];

	for (int i=0; i<h; ++i) {
		mat[i] = &data[i*w];
	}
}

template <class T>
inline void allocateMatrixAccess(T*& data, T**& mat, const int& w, const int& h){
	mat = new T*[h];

	for (int i=0; i<h; ++i) {
		mat[i] = &data[i*w];
	}
}


template <class T>
inline void allocateVolume(T*& data, T**& rows, T***& vol, const int& w, const int& h, const int& d){
	int wh = w*h;

	data = new T[wh*d];
	rows = new T*[h*d];
	vol = new T**[d];

	for (int i=0; i<d; ++i) {
		vol[i] = &rows[i*h];
		for (int j=0; j<h; ++j) {
			rows[i*h+j] = &data[i*wh+j*w];
		}
	}
}

template <class T>
inline void allocateVolumeAccess(T* data, T**& rows, T***& vol, const int& w, const int& h, const int& d){
	int wh = w*h;

	rows = new T*[h*d];
	vol = new T**[d];
	for (int i=0; i<d; ++i) {
		vol[i] = &rows[i*h];
		for (int j=0; j<h; ++j) {
			rows[i*h+j] = &data[i*wh+j*w];
		}
	}
}

#ifdef USE_CUDA
//TODO: try other implementations later
// 1. copy the float3 vector field to gpu first and then use a global function to copy each element
// 2. copy with stride
inline cudaArray* allocateCudaVectorField(vec3f* vec_field, const vec3i& dim, const float& t_speed=1.0f){
	//copy as float4
	float4* vf = new float4[dim.x*dim.y*dim.z];
	for (int i=0; i<dim.x*dim.y*dim.z; ++i) {
		vf[i] = make_float4(vec_field[i].x, vec_field[i].y, vec_field[i].z, t_speed);
	}

	cudaArray* ret;
	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	checkCudaErrors(cudaMalloc3DArray(&ret, &channelDesc, make_cudaExtent(dim.x, dim.y, dim.z)));

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr(vf, dim.x*sizeof(float4), dim.x, dim.y);
	copyParams.dstArray = ret;
	copyParams.extent   = make_cudaExtent(dim.x, dim.y, dim.z);
	copyParams.kind     = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams));

	delete[] vf;
	return ret;
}

inline cudaArray* allocateCudaVectorField(vec4f* vec_field, const vec3i& dim) {
	cudaArray* ret;
	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	checkCudaErrors(cudaMalloc3DArray(&ret, &channelDesc, make_cudaExtent(dim.x, dim.y, dim.z)));

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr(vec_field, dim.x * sizeof(float4), dim.x, dim.y);
	copyParams.dstArray = ret;
	copyParams.extent = make_cudaExtent(dim.x, dim.y, dim.z);
	copyParams.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams));

	return ret;
}

inline cudaArray* allocateCudaVectorField(const vec3i& dim) {
	cudaArray* ret;
	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	checkCudaErrors(cudaMalloc3DArray(&ret, &channelDesc, make_cudaExtent(dim.x, dim.y, dim.z)));
	return ret;
}

inline void updateCudaVectorField(cudaArray* vec_field_d, vec4f* vec_field_h, const vec3i& dim){
	// copy data to 3D array
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr(vec_field_h, dim.x*sizeof(float4), dim.x, dim.y);
	copyParams.dstArray = vec_field_d;
	copyParams.extent   = make_cudaExtent(dim.x, dim.y, dim.z);
	copyParams.kind     = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams));
}
#endif

inline bool open_file(std::ifstream& fin, const char* fname, const bool& binary=false){
	if (binary) {
		fin.open(fname, std::ios_base::in|std::ios::binary);
	} else {
		fin.open(fname);
	}
	if(!fin.is_open()){
		printf("Fail to read file: %s.\n", fname);
		return false;
	}
	return true;
}

inline bool open_file(std::ofstream& fout, const char* fname, const bool& binary=false){
	if (binary) {
		fout.open(fname, std::ios_base::out|std::ios::binary);
	} else {
		fout.open(fname);
	}
	if(!fout.is_open()){
		printf("Fail to read file: %s.\n", fname);
		return false;
	}
	return true;
}

inline bool file_exist(const char* file_path) {
	std::ifstream in_file;
	bool ret = open_file(in_file, file_path);
	in_file.close();
	return ret;
}

static bool readStreamlinePool(StreamlinePool& ret_pool, const char* file_path) {
	std::ifstream input_file;
	if (!open_file(input_file, file_path, true)) {
		return false;
	}

	float radius;
	int num_points, num_streamlines;
	input_file.read((char*)&num_points, sizeof(int));
	input_file.read((char*)&num_streamlines, sizeof(int));
	input_file.read((char*)&radius, sizeof(float));

	ret_pool.points.resize(num_points);
	ret_pool.streamlines.resize(num_streamlines);

	input_file.read((char*)ret_pool.points.data(), sizeof(vec3f)*num_points);
	input_file.seekg(sizeof(vec3f)*num_points, input_file.cur);
	input_file.read((char*)ret_pool.streamlines.data(), sizeof(Streamline)*num_streamlines);
	input_file.close();

	ret_pool.fillLineIds();

	return true;
}

static bool readBrainStreamlinePool(StreamlinePool& ret_pool, const char* file_path) {
	std::ifstream input_file;
	if (!open_file(input_file, file_path, true)) {
		return false;
	}

	int num_points, num_stls;
	input_file.read((char*)&num_points, sizeof(int));
	input_file.read((char*)&num_stls, sizeof(int));

	ret_pool.points.resize(num_points);
	ret_pool.streamlines.resize(num_stls);

	input_file.read((char*)ret_pool.points.data(), sizeof(vec3f) * num_points);
	input_file.read((char*)ret_pool.streamlines.data(), sizeof(Streamline) * num_stls);
	input_file.close();

	ret_pool.fillLineIds();

	return true;
}

static bool writeStreamlinePool(StreamlinePool& pool, const char* file_path) {
	std::ofstream output_file;
	if (!open_file(output_file, file_path, true)) {
		return false;
	}

	float radius = 1.0f;
	int num_points = pool.points.size(), num_streamlines = pool.streamlines.size();
	output_file.write((char*)&num_points, sizeof(int));
	output_file.write((char*)&num_streamlines, sizeof(int));
	output_file.write((char*)&radius, sizeof(float));

	output_file.write((char*)pool.points.data(), sizeof(vec3f)*num_points);
	// output_file.write((char*)pool.points.data(), sizeof(vec3f)*num_points);
	output_file.write((char*)pool.streamlines.data(), sizeof(Streamline)*num_streamlines);
	output_file.close();

	return true;
}

template <typename T>
static bool read_array(T* ret, const int& n, const char* file_path) {
	std::ifstream input_file;
	if (!open_file(input_file, file_path, true)) {
		return false;
	}

	input_file.read((char*)ret, sizeof(T)*n);
	input_file.close();

	return true;
}

template <typename T>
static bool read_array(std::vector<T>& ret, const char* file_path) {
	std::ifstream input_file;
	if (!open_file(input_file, file_path, true)) {
		return false;
	}
	input_file.seekg(0, input_file.end);
	size_t length = input_file.tellg();
	input_file.seekg(0, input_file.beg);
	size_t n = length / sizeof(T);
	ret.resize(n);

	input_file.read((char*)ret.data(), sizeof(T)*n);
	input_file.close();

	return true;
}

template <typename T>
static bool write_array(T* ret, const int& n, const char* file_path) {
	std::ofstream output_file;
	if (!open_file(output_file, file_path, true)) {
		return false;
	}

	output_file.write((char*)ret, sizeof(T)*n);
	output_file.close();

	return true;
}

static vec3i read_header(const char* filename){
	std::ifstream fin;
	if (!open_file(fin, filename, false)) {
		return makeVec3i(-1,-1,-1);
	}
	vec3i ret;
	fin>>ret.x>>ret.y>>ret.z;
	fin.close();
	return ret;
}

static Range compute_bound(float* val, int num, float cutoff_ratio = 0.05f) {
	cutoff_ratio = clamp(cutoff_ratio, 0.0f, 1.0f);
	if (cutoff_ratio > 0.5f) cutoff_ratio = 1.0f - 0.5f;
	Range ret(0, 0);
	if (num == 1) {
		ret = Range(val[0], val[0]);
	}
	else if (cutoff_ratio < 0.0001f) {
		ret.upper = ret.lower = val[0];
		for (int i = 1; i < num; ++i) {
			if (val[i] > ret.upper) ret.upper = val[i];
			if (val[i] < ret.lower) ret.lower = val[i];
		}
	}
	else {
		std::vector<float> cp_vals(val, val + num);
		int low_idx = num * cutoff_ratio;
		int up_idx = num - low_idx;
		if (low_idx > up_idx) std::swap(low_idx, up_idx);
		std::nth_element(cp_vals.begin(), cp_vals.begin() + low_idx, cp_vals.end());
		std::nth_element(cp_vals.begin(), cp_vals.begin() + up_idx, cp_vals.end());
		ret.lower = cp_vals[low_idx];
		ret.upper = cp_vals[up_idx];
	}
	return ret;
}

template <typename T>
static void normalizeArray(T* val, const int& num, const float& cutoff_percentage=0.9f){
	float upper, lower;
	if (cutoff_percentage>=0.9999999f) {
		upper = lower = val[0];
		for (int i=1; i<num; ++i) {
			if (val[i]>upper) upper = val[i];
			if (val[i]<lower) lower = val[i];
		}
	} else {
		std::vector<T> sorted_array;
		sorted_array.assign(val, val+num);
		std::sort(sorted_array.begin(), sorted_array.end());
		int upper_idx = num*cutoff_percentage, lower_idx = num*(1-cutoff_percentage);
		upper = sorted_array[upper_idx];
		lower = sorted_array[lower_idx];
	}


	float fac = 1.0f/(upper-lower);

	for (int i=0; i<num; ++i) {
		val[i] = (val[i]-lower)*fac;
		val[i] = clamp(val[i], 0.0f, 1.0f);
	}
}

static void discretize(int* ret, float* vals, const int& n, const int& num_bin, const float& cutoff_ratio = 0.05f){
	std::vector<float> cp_vals(vals, vals+n);
	int low_idx = n*cutoff_ratio;
	int up_idx = n-low_idx;
	if (low_idx>up_idx) std::swap(low_idx, up_idx);
	std::nth_element(cp_vals.begin(), cp_vals.begin()+low_idx, cp_vals.end());
	std::nth_element(cp_vals.begin(), cp_vals.begin()+up_idx, cp_vals.end());
	float low = cp_vals[low_idx], up = cp_vals[up_idx];
	float fac = num_bin/(up-low);
	for (int i=0; i<n; ++i) {
		ret[i] = clamp((int)((vals[i]-low)*fac), 0, num_bin-1);
	}
}

template<typename T>
static void combineArray(T** ret, const T* a, const int& num_a, const T* b, const int& num_b){
	int num = num_a+num_b;
	*ret = new T[num];
	memcpy(*ret, a, sizeof(T)*num_a);
	memcpy(*ret+num_a, b, sizeof(T)*num_b);
}

template<typename T>
static void combineArray(T** ret, const std::vector<T*> arrays, const std::vector<int>& nums, int& total_nums){
	if (total_nums==0) {
		for (int i=0; i<nums.size(); ++i) {
			total_nums += nums[i];
		}
	}
	*ret = new T[total_nums];

	int cur = 0;
	for (int i=0; i<nums.size(); ++i) {
		memcpy(*ret+cur, arrays[i], sizeof(T)*nums[i]);
		cur += nums[i];
	}
}

static void histogram(std::vector<int>& hist, float* val, const int& n, const int& num_bin,
	const Range& bound, const bool& count_out_of_bound)
{
	hist.assign(num_bin, 0);
	float fac = (float)num_bin / (bound.upper - bound.lower);
	int max_bin = num_bin - 1;

	if (count_out_of_bound) {
		for (int i = 0; i < n; ++i) {
			++hist[clamp((int)((val[i] - bound.lower) * fac), 0, max_bin)];
		}
	}
	else {
		for (int i = 0; i < n; ++i) if (inBound(val[i], bound.lower, bound.upper)) {
			++hist[clamp((int)((val[i] - bound.lower) * fac), 0, max_bin)];
		}
	}
}

template<typename T>
static void normailize_array(std::vector<T>& hist)
{
	if (hist.empty()) {
		// 数据为空，不进行操作
		return;
	}

	// 寻找最小值和最大值
	T min_val = *std::min_element(hist.begin(), hist.end());
	T max_val = *std::max_element(hist.begin(), hist.end());

	// 遍历并进行归一化
	for (T& value : hist) {
		value = (value - min_val) / (max_val - min_val);
	}
}

#endif