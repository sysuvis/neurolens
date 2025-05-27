#ifndef STREAMLINE_RESAMPLE_H
#define STREAMLINE_RESAMPLE_H

#include "typeOperation.h"

#define MAX_RESAMPLE_ITER 5

void resampleStreamlineDiscreteCurvature(vec3f* points, float* curvature, const int& numPoint, float thresh, vec3f *retPoints, int *retIndices, const int& minNum, int& retNum);

namespace StreamlineResample {
	typedef enum ResampleType{
		arc_length = 0,
		acc_attrib
	};

	template<typename index_type>
	class Pool {
	public:
		Pool(){}

		int getNumPoints() {
			return points.size();
		}

		StreamlinePool::PointInfo getPointInfo(const int& point_global_id) {
			StreamlinePool::PointInfo ret;
			ret.streamline_id = line_ids[point_global_id];
			ret.point_local_id = point_global_id - streamlines[ret.streamline_id].start;
			return ret;
		}

		int getPointGlobalId(const int& point_local_id, const int& streamline_id) {
			return point_local_id + streamlines[streamline_id].start;
		}

		int getPointGlobalId(const StreamlinePool::PointInfo& pinfo) {
			return getPointGlobalId(pinfo.point_local_id, pinfo.streamline_id);
		}

		index_type* getPoints(const int& streamline_id) {
			return &points[streamlines[streamline_id].start];
		}

		index_type getPoint(const int& point_local_id, const int& streamline_id) {
			return points[getPointGlobalId(point_local_id, streamline_id)];
		}

		//get the segment centered at center, each side covers index_radius points
		StreamlineSegment getSegmentClamp(const int& point_global_id, const int& index_radius) {
			getSegmentClamp(getPointInfo(point_global_id), index_radius);
		}

		StreamlineSegment getSegmentClamp(const StreamlinePool::PointInfo& center, const int& index_radius) {
			const int& num_points = streamlines[center.streamline_id].numPoint;
			StreamlineSegment ret;
			ret.streamline_id = center.streamline_id;
			ret.segment.lower = clamp(center.point_local_id - index_radius, 0, num_points - 1);
			ret.segment.upper = clamp(center.point_local_id + index_radius, 0, num_points - 1);
			return ret;
		}

		StreamlineSegment getSegmentShift(const int& point_global_id, const int& index_radius) {
			getSegmentShift(getPointInfo(point_global_id), index_radius);
		}

		StreamlineSegment getSegmentShift(const StreamlinePool::PointInfo& center, const int& index_radius) {
			const int& num_points = streamlines[center.streamline_id].numPoint;
			StreamlineSegment ret;
			ret.streamline_id = center.streamline_id;
			ret.segment.lower = center.point_local_id - index_radius;
			ret.segment.upper = center.point_local_id + index_radius;
			moveRangeInBound(ret.segment, makeIndexRange(0, num_points));
			return ret;
		}

		StreamlineSegment mapToResample(const StreamlineSegment& seg) {
			StreamlineSegment ret;
			ret.streamline_id = seg.streamline_id;

			//resample indices of the queried streamline
			const Streamline& s = streamlines[seg.streamline_id];
			index_type* start = &points[s.start];
			index_type* end = start + s.numPoint;

			ret.segment.lower = std::lower_bound(start, end, seg.segment.lower) - start;
			ret.segment.upper = std::upper_bound(start, end, seg.segment.upper) - start;
			return ret;
		}

		StreamlineSegment mapToOriginal(const StreamlineSegment& resample_seg) {
			StreamlineSegment ret;
			ret.streamline_id = resample_seg.streamline_id;
			ret.segment.upper = ceil(getPoint(resample_seg.segment.upper, resample_seg.streamline_id));
			ret.segment.lower = floor(getPoint(resample_seg.segment.lower, resample_seg.streamline_id));
			return ret;
		}

		void mapToResample(std::vector<int>& ret_point_id, const int& streamline_id) {
			const Streamline& rs = streamlines[streamline_id];
			const Streamline& s = org_pool->streamlines[streamline_id];

			index_type* resample_points = getPoints(streamline_id);
			ret_point_id.resize(s.numPoint);
			ret_point_id[0];
			for (int i = 1, p = 1; i < s.numPoint; ++i) {
				//find the first p larger than i
				while (resample_points[p] < i && p < rs.numPoint) ++p;
				if (p == rs.numPoint ) {//if p already reach the end
					while (i<s.numPoint) {
						ret_point_id[i] = p-1;
						++i;
					}
				} else {
					ret_point_id[i] = (i - resample_points[p - 1] > resample_points[p] - i) ? p : (p - 1);
				}
			}
		}

		void mapToResample(std::vector<int>& ret_point_id) {
			ret_point_id.reserve(org_pool->getNumPoints());
			for (int i = 0; i < org_pool->streamlines.size(); ++i) {
				std::vector<int> point_id_of_line;
				mapToResample(point_id_of_line, i);
				int start = streamlines[i].start;
				for (auto& point_id : point_id_of_line) point_id += start;
				ret_point_id.insert(ret_point_id.end(), point_id_of_line.begin(), point_id_of_line.end());
			}
		}

		int mapToResample(const int& streamline_id, const int& point_id) {
			const Streamline& s = streamlines[streamline_id];
			index_type* start = &points[s.start];
			index_type* end = start + s.numPoint;

			int ret = std::lower_bound(start, end, point_id)-start;
			if (ret == s.numPoint - 1) return ret;
			if (point_id - start[ret] < start[ret + 1] - point_id) {
				return ret;
			}
			return ret + 1;
		}

		StreamlinePool* org_pool;
		std::vector<index_type> points;
		std::vector<int> line_ids;
		std::vector<Streamline> streamlines;
	};

	template <typename index_type>
	class MultiPool {
	public:
		MultiPool() {
			pool = NULL;
			resample_pools.resize(2);
		}

		MultiPool(StreamlinePool* _pool) {
			pool = _pool;
		}

		void setStreamlinePool(StreamlinePool* _pool) { pool = _pool; }

		template<ResampleType resample_type>
		void resample(const float& thresh);

		template<typename T>
		static void resample(std::vector<index_type>& ret, T* points, const int& num, const float& thresh);

		void getResampleLine(std::vector<vec3f>& ret, const int& streamline_id, 
			const float& thresh, const ResampleType& resample_type);

		Pool<index_type>* getResamplePool(const float& thresh, const ResampleType& resample_type) {
			auto it = resample_pools[(int)resample_type].find(thresh);
			if (it != resample_pools[(int)resample_type].end()) {
				return &(it->second);
			}
			return NULL;
		}

		StreamlineSegment mapToResample(const StreamlineSegment& seg, const float& thresh) {
			StreamlineSegment ret;
			Pool<index_type>* resample_pool = getResamplePool(thresh);
			if (resample_pool != NULL) {
				ret = resample_pool->mapToResample(seg);
			}
			return ret;
		}

		StreamlineSegment mapToOriginal(const StreamlineSegment& resample_seg, const float& thresh) {
			StreamlineSegment ret;
			Pool<index_type>* resample_pool = getResamplePool(thresh);
			if (resample_pool != NULL) {
				ret = resample_pool->mapToOriginal(seg);
			}
			return ret;
		}

		void mapToResample(std::vector<int>& ret_point_id, const int& streamline_id, const float& thresh,
			const ResampleType& resample_type) 
		{
			Pool<index_type>& resample_pool = *(getResamplePool(thresh, resample_type)); 
			resample_pool.mapToResample(ret_point_id, streamline_id);
		}

		void mapToResample(std::vector<int>& ret_point_id, const float& thresh, const ResampleType& resample_type) {
			Pool<index_type>& resample_pool = *(getResamplePool(thresh, resample_type));
			resample_pool.mapToResample(ret_point_id);
		}

		void setAttribs(const std::vector<float>& _attribs) {
			if (_attribs.size() != pool->getNumPoints()) {
				printf("Err: size of attributes does not match with the number of points.\n");
				return;
			}
			attribs.assign(_attribs.begin(), _attribs.end());
		}

		std::vector<float> attribs;//for each line, the acc_attribs should start from 0
		StreamlinePool* pool;
		std::vector<std::map<float, Pool<index_type>>> resample_pools;
	};
}

#endif //STREAMLINE_RESAMPLE_H