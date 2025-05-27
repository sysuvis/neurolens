#include "StreamlineCharacteristics.h"

void discreteCurvatureTorsion(vec3f *points, const int& numPoint, float *curvature, float *torsion){
	vec3f v1, v2, v3;
	v1 = points[1]-points[0];
	v2 = points[2]-points[1];
	v3 = points[3]-points[2];

	float kappa = computeCurvature(v1, v2);
	float tau;
	float ptau = computeTorsion(v1, v2, v3);

	curvature[0] = kappa;
	curvature[1] = kappa;
	torsion[0] = ptau;
	torsion[1] = ptau;

	for(int i=2; i<numPoint-2; i++){
		v1 = v2;
		v2 = v3;
		v3 = points[i+2]-points[i+1];

		curvature[i] = computeCurvature(v1, v2);
		tau = computeTorsion(v1, v2, v3);
		
		torsion[i] = (tau+ptau)*0.5f;
		ptau = tau;
	}

	kappa = computeCurvature(v2, v3);

	curvature[numPoint-2] = kappa;
	curvature[numPoint-1] = kappa;
	torsion[numPoint-2] = tau;
	torsion[numPoint-1] = tau;
}

void discreteCurvatureTorsion(vec3f *points, Streamline *pool, const int& numStl, float *curvature, float *torsion){
	for (int i=0; i<numStl; ++i) {
		Streamline s = pool[i];
		discreteCurvatureTorsion(&points[s.start], s.numPoint, &curvature[s.start], &torsion[s.start]);
	}
}

void averageDiscreteCurvatureTorsion( vec3f *points, const int& numPoint, float& curvature, float& torsion ){
	vec3f v1, v2, v3;

	v1 = points[1] - points[0];
	v2 = points[2] - points[1];
	v3 = points[3] - points[2];

	curvature = computeCurvature(v1, v2)+computeCurvature(v2, v3);
	torsion = computeTorsion(v1, v2, v3);

	for (int i=3; i<numPoint-1; ++i) {
		v1 = v2;
		v2 = v3;
		v3 = points[i+1]-points[i];

		curvature += computeCurvature(v2, v3);
		torsion += computeTorsion(v1, v2, v3);
	}

	curvature /= (numPoint-1);
	torsion /= (numPoint-2);
}
