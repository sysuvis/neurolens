#ifndef STREAMLINE_CHARACTERISTICS_H
#define STREAMLINE_CHARACTERISTICS_H

#include "typeOperation.h"

//curvature range: [0, pi]
inline __host__ __device__ float computeCurvature(const vec3f& v1, const vec3f& v2){
	float cosv = (v1*v2)/vec3fLen(v1)/vec3fLen(v2);
	cosv = clamp(cosv, -1.0f, 1.0f);
	return (acosf(cosv));
}

//torsion range: [-pi, pi]
inline __host__ __device__ float computeTorsion(const vec3f& v1, const vec3f& v2, const vec3f& v3){
	float len1 = vec3fLen(v1);
	float len2 = vec3fLen(v2);
	float len3 = vec3fLen(v3);
	float tmp1 = (v1*v2)/len1/len2;
	float tmp2 = (v2*v3)/len2/len3;
	if(tmp1>0.99f || tmp1<-0.99f || tmp2>0.99f || tmp2<-0.99f){
		return 0.0f;
	} else {
		float cosv = (tmp1*tmp2-((v1*v3)/len1/len3))/sqrtf(1.0f-tmp1*tmp1)/sqrtf(1.0f-tmp2*tmp2);
		cosv = clamp(cosv, -1.0f, 1.0f);
		if (cross(v1, v2)*v3<0) return (-acosf(cosv));
		return (acosf(cosv));
	}
}

//compute average curvature and torsion for one streamline
void averageDiscreteCurvatureTorsion( vec3f *points, const int& numPoint, float& curvature, float& torsion );
void discreteCurvatureTorsion(vec3f *points, Streamline *pool, const int& numStl, float *curvature, float *torsion);
void discreteCurvatureTorsion(vec3f *points, const int& numPoint, float *curvature, float *torsion);

#endif