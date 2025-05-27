#ifndef SPHERE_H
#define SPHERE_H

#include "typeOperation.h"
#include <vector>

static void generateSphere(std::vector<vec3f>& ret_points, const int& n, const float& radius){
	float golden_angle = 3.1415926f*(3.0f-sqrtf(5.0f));
	float theta, r;
	float z = 1.0f-(1.0f/n);
	float z_inc = (z-(1.0f/n)+1.0f)/(1-n);
	vec3f p;

	for (int i=0; i<n; ++i, z+=z_inc) {
		theta = golden_angle*i;
		r = radius*sqrtf(1.0f-z*z);
		p = makeVec3f(r*cosf(theta), r*sinf(theta), z);
		normalize(p);
		ret_points.push_back(p);
	}
}

#define PI 3.1415926f

typedef struct{
	float theta1, theta2;
	short startIdx;
	float avgAngle;
} RegInfoCollarItem;

typedef struct{
	RegInfoCollarItem *collarInfo;
	int numCollar;
	int numRegion;
} RegInfoItem;

static void partitionUnitSphere(RegInfoItem &regInfo){
	float spharea = 4*PI*1*1;	//r == 1
	float vr = spharea/(float)regInfo.numRegion;
	float thetac;
	float deltai;
	float deltaf;
	float ni;
	int n;

	if(regInfo.numRegion == 1){	//only single region
		regInfo.numCollar = 1;
		regInfo.collarInfo = new RegInfoCollarItem[1];
		regInfo.collarInfo[0].theta1 = 0;
		regInfo.collarInfo[0].theta2 = PI;
		regInfo.collarInfo[0].startIdx = 0;
		regInfo.collarInfo[0].avgAngle = 2*PI;
	} else {
		//step 1:determine the colatitude of cap
		thetac = asin(sqrt(vr/(4*PI)))*2;

		//step2: determine ideal collar angle
		deltai = sqrt(vr);

		//step3: determine the ideal collar number
		ni = (PI - 2*thetac)/deltai;

		//step4: determine the actual collar number
		int n2 = floor(ni + 0.5);
		if(n2 > 1)
			n = n2;
		else
			n = 1;
		regInfo.numCollar = n+2;
		regInfo.collarInfo = new RegInfoCollarItem[regInfo.numCollar];

		//step5: creat list of ideal number of regions of each collar
		deltaf = ni/(float)n*deltai;
		float *thetaF;
		thetaF = new float[n + 1];
		for(int i = 0; i <n+1; i++){
			thetaF[i] = thetac + i*deltaf;
		}

		float *y;
		y = new float[n+1];
		for(int i = 1; i <= n; i++){
			y[i] = (4*PI*sin(thetaF[i]/2)*sin(thetaF[i]/2) - 4*PI*sin(thetaF[i-1]/2)*sin(thetaF[i-1]/2))/vr;
		}

		//step6: creat list of actual number of regions of each collar
		float *m;
		m = new float[n+1];
		float *a;
		a = new float[n+1];
		a[0] = 0;
		int max= -1;	// this is used to store the largest number of regions of the collar
		for(int i = 1; i <= n; i++){
			m[i] = floor(y[i] + a[i-1] + 0.5);
			float sum = 0 ;
			for(int j = 1; j <= i; j++)
				sum += y[j] - m[j];
			a[i] = sum;
			if(m[i]>=max)
				max = m[i];
		}

		//step7: creat list of colatitudes for each zone
		float *theta;
		theta = new float[n+3];
		theta[0] = 0;
		theta[n+2] = PI;
		for(int i = 1; i < n+2; i++){
			float sum = 0;
			for(int j = 1; j <= i-1; j++){
				sum += m[j];
			}
			sum += 1;
			float sumarea = sum*vr;
			theta[i] = asin(sqrt(sumarea/(4*PI)))*2;
		}

		float sum = 2 ;
		for(int i = 1; i <= n; i++)
			sum += m[i];

		//step8: partition each collar and store the final region information into the array ra;
		regInfo.collarInfo[0].theta1 = theta[0];
		regInfo.collarInfo[0].theta2 = theta[1];
		regInfo.collarInfo[0].startIdx = 0;
		regInfo.collarInfo[0].avgAngle = 2.0f*PI;
		int currIdx = 1;
		for(int i = 1; i <= n; i++){	//for each collar, we partition it into m[i] regions
			regInfo.collarInfo[i].theta1 = theta[i];
			regInfo.collarInfo[i].theta2 = theta[i+1];
			regInfo.collarInfo[i].startIdx = currIdx;
			regInfo.collarInfo[i].avgAngle = (2.0f*PI)/m[i];

			currIdx += m[i];
		}

		regInfo.collarInfo[n+1].theta1 = theta[n+1];
		regInfo.collarInfo[n+1].theta2 = theta[n+2];
		regInfo.collarInfo[n+1].startIdx = currIdx;
		regInfo.collarInfo[n+1].avgAngle = 2.0*PI;

		delete[] m;
		delete[] a;
		delete[] theta;
	}
}

static void generateSphereUsingPartition(std::vector<vec3f>& ret_points, const int& n, const float& radius){
	RegInfoItem reg_info = {NULL, 0, n};
	partitionUnitSphere(reg_info);

	for (int i=0; i<reg_info.numCollar; ++i) {
		const RegInfoCollarItem& collar = reg_info.collarInfo[i];
		float theta = 0.5f*(collar.theta1+collar.theta2);
		vec3f p;
		p.z = radius*sinf(theta);
		float r = radius*cosf(theta);
		float angle = (i&1)?(0.5f*collar.avgAngle):(0.0f);
		for (;angle<PI*2.0f; angle += collar.avgAngle) {
			p.x = r*sinf(angle);
			p.y = r*cosf(angle);
			ret_points.push_back(p);
		}
	}

	delete reg_info.collarInfo;
}

namespace icosahedron
{
	const float X=.525731112119133606f;
	const float Z=.850650808352039932f;
	const float N=0.f;

	static const vec3f vertices[12]=
	{
		{-X,N,Z}, {X,N,Z}, {-X,N,-Z}, {X,N,-Z},
		{N,Z,X}, {N,Z,-X}, {N,-Z,X}, {N,-Z,-X},
		{Z,X,N}, {-Z,X, N}, {Z,-X,N}, {-Z,-X, N}
	};

	static const vec3i triangles[20]=
	{
		{0,4,1},{0,9,4},{9,5,4},{4,5,8},{4,8,1},
		{8,10,1},{8,3,10},{5,3,8},{5,2,3},{2,7,3},
		{7,10,3},{7,6,10},{7,11,6},{11,0,6},{0,1,6},
		{6,1,10},{9,0,11},{9,11,2},{9,2,5},{7,2,11}
	};
}

static void generateSphereSubdivision(std::vector<vec3f>& vertices, std::vector<vec3i>& triangles,
	const int& num_iterations, const float& radius) 
{
	vertices.assign(icosahedron::vertices, icosahedron::vertices+12);
	triangles.assign(icosahedron::triangles, icosahedron::triangles+20);

	for (int iter = 0; iter<num_iterations; ++iter) {
		std::vector<vec3i> new_triangles;
		for (int i = 0; i<triangles.size(); ++i) {
			const vec3i& t = triangles[i];
			int xy = vertices.size(), xz = xy+1, yz = xz+1;

			vec3f v = 0.5f*(vertices[t.x]+vertices[t.y]);
			normalize(v);
			vertices.push_back(v);

			v = 0.5f*(vertices[t.x]+vertices[t.z]);
			normalize(v);
			vertices.push_back(v);

			v = 0.5f*(vertices[t.y]+vertices[t.z]);
			normalize(v);
			vertices.push_back(v);

			new_triangles.push_back(makeVec3i(t.x, xy, xz));
			new_triangles.push_back(makeVec3i(t.y, yz, xy));
			new_triangles.push_back(makeVec3i(t.z, xz, yz));
			new_triangles.push_back(makeVec3i(xy, yz, xz));
		}
		triangles.assign(new_triangles.begin(), new_triangles.end());
	}

	for (int i = 0; i<vertices.size(); ++i) {
		vertices[i] *= radius;
	}
}

static void generateSphereSubdivision(std::vector<vec3f>& ret_points, const int& num_iterations, const float& radius){
	std::vector<vec3f> vertices(icosahedron::vertices, icosahedron::vertices+12);
	std::vector<vec3i> triangles(icosahedron::triangles, icosahedron::triangles+20);

	generateSphereSubdivision(vertices, triangles, num_iterations, radius);

	ret_points.clear();

	for (int i=0; i<triangles.size(); ++i) {
		const vec3i& t = triangles[i];
		ret_points.push_back(vertices[t.x]+vertices[t.y]+vertices[t.z]);
	}
}
#endif //SHERE_H