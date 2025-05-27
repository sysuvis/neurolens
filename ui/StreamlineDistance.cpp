#include "StreamlineDistance.h"
#include "StreamlineCharacteristics.h"
#include "InfoTheory.h"
#include <cstdio>
#include <string>
#include <vector>

float* genClosestPointDistanceMatrix(Streamline* stls, vec3f* points, vec2i* pairs, const int& numStls, const int& numPairs){
	int i, u, v;
	float* ret = new float[numStls*numStls];
	float** mat = new float*[numStls];
	float dist;
	for (i=0; i<numStls*numStls; ++i) ret[i] = 1e30;
	for (i=0; i<numStls; ++i) mat[i] = &ret[i*numStls];
	for (i=0; i<numPairs; ++i) {
		u = pairs[i].x;
		v = pairs[i].y;
		dist = computeClosestPointDistance(&points[stls[u].start], &points[stls[v].start], stls[u].numPoint, stls[v].numPoint);
		dist += computeClosestPointDistance(&points[stls[v].start], &points[stls[u].start], stls[v].numPoint, stls[u].numPoint);
		mat[u][v] = mat[v][u] = dist;
	}
	delete[] mat;

	return ret;
}

float* genClosestPointDistanceMatrix(Streamline* stls, vec3f* points, const int& numStls){
	int i, u, v;
	float* ret = new float[numStls*numStls];
	float** mat = new float*[numStls];
	float dist;
	for (i=0; i<numStls*numStls; ++i) ret[i] = 1e30;
	for (i=0; i<numStls; ++i) mat[i] = &ret[i*numStls];
	//printf("Distance Matrix Computation: 0.0000%%");
	float one_streamline_percentage = 200.0f/((numStls-2)*(numStls-1));
	float percentage = 0.0f;
	for (u=0; u<numStls; ++u){
		mat[u][u] = 0.0f;
		for (v=u+1; v<numStls; ++v) {
			dist = computeClosestPointDistance(&points[stls[u].start], &points[stls[v].start], stls[u].numPoint, stls[v].numPoint);
			dist += computeClosestPointDistance(&points[stls[v].start], &points[stls[u].start], stls[v].numPoint, stls[u].numPoint);
			mat[u][v] = mat[v][u] = dist;
			percentage += one_streamline_percentage;
			//printf("\rDistance Matrix Computation: %6.4f%%", percentage);
		}
	}
	//printf("\n");
	delete[] mat;

	return ret;
}

void computeClosestPointDistanceMatrix(Streamline* stls, vec3f* points, const int& numStls,
									   float* avg, float* minv, float* maxv)
{
	int i, u, v;
	float dist;

	float tmp_avg, tmp_minv, tmp_maxv;
	printf("Distance Matrix Computation: 0.0000%%");
	float one_streamline_percentage = 100.0f/(numStls*(numStls-1));
	float percentage = 0.0f;
	for (u=0; u<numStls; ++u){
		avg[u*numStls+u] = 0.0f;
		minv[u*numStls+u] = 0.0f;
		maxv[u*numStls+u] = 0.0f;
		for (v=0; v<numStls; ++v) if(u!=v) {
			computeClosestPointDistance(&points[stls[u].start], &points[stls[v].start], stls[u].numPoint, stls[v].numPoint,
				tmp_avg, tmp_minv, tmp_maxv);
			
			avg[u*numStls+v] = tmp_avg;
			minv[u*numStls+v] = tmp_minv;
			maxv[u*numStls+v] = tmp_maxv;

			percentage += one_streamline_percentage;
			printf("\rDistance Matrix Computation: %6.4f%%", percentage);
		}
	}
	printf("\n");
}

void computeClosestPointDistanceMatrix(Streamline* stls, vec3f* points, const int& numStls,
									   const float& bin_width, const int& num_bin,
									   float* avg, float* minv, float* maxv, short* hist)
{
	int i, u, v;
	float dist;

	memset(hist, 0, sizeof(short)*numStls*numStls*num_bin);

	float tmp_avg, tmp_minv, tmp_maxv;
	printf("Distance Matrix Computation: 0.0000%%");
	float one_streamline_percentage = 100.0f/(numStls*(numStls-1));
	float percentage = 0.0f;
	for (u=0; u<numStls; ++u){
		avg[u*numStls+u] = 0.0f;
		minv[u*numStls+u] = 0.0f;
		maxv[u*numStls+u] = 0.0f;
		for (v=0; v<numStls; ++v) if(u!=v) {
			computeClosestPointDisanceDistribution(&points[stls[u].start], &points[stls[v].start], stls[u].numPoint, stls[v].numPoint,
				bin_width, num_bin,
				&hist[(u*numStls+v)*num_bin],
				tmp_avg, tmp_minv, tmp_maxv);

			avg[u*numStls+v] = tmp_avg;
			minv[u*numStls+v] = tmp_minv;
			maxv[u*numStls+v] = tmp_maxv;

			percentage += one_streamline_percentage;
			printf("\rDistance Matrix Computation: %6.4f%%", percentage);
		}
	}
	printf("\n");
}

void computeCurvatureTorsionJSDMatrix(Streamline* stls, vec3f* points, const int& numStls, const int& numBin,
										float* curvatureJSDMat, float* torsionJSDMat)
{
	int numPoint = stls[numStls-1].start+stls[numStls-1].numPoint;
	float* curvature = new float[numPoint];
	float* torision = new float[numPoint];

	discreteCurvatureTorsion(points, stls, numStls, curvature, torision);

	int* curvatureHist = new int[numStls*numBin];
	int* torisionHist = new int[numStls*numBin];
	memset(curvatureHist, 0, sizeof(int)*numStls*numBin);
	memset(torisionHist, 0, sizeof(int)*numStls*numBin);
	
	////compute from actual data
	//float cmax = curvature[0], cmin = curvature[0];
	//float tmax = torision[0], tmin = torision[0];
	//for(int i=1; i<numPoint; ++i){
	//	if (cmax < curvature[i]) cmax = curvature[i];
	//	if (cmin > curvature[i]) cmin = curvature[i];
	//	if (tmax < torision[i]) tmax = torision[i];
	//	if (tmin > torision[i]) tmin = torision[i];
	//}

	Streamline s;
	float maxv = 3.0f;
	for (int i=0; i<numStls; ++i) {
		s = stls[i];
		computeArrayHist(&curvature[s.start], s.numPoint, 0, maxv, numBin, &curvatureHist[i*numBin]);
		computeArrayHist(&torision[s.start], s.numPoint, -maxv, maxv, numBin, &torisionHist[i*numBin]);
	}

	delete[] curvature;
	delete[] torision;

	float one_streamline_percentage = 200.0f/(numStls*(numStls-1));
	float percentage = 0.0f;
	int numi, numj;
	float cJSD, tJSD;
	for (int i=0; i<numStls; ++i) {
		numi = stls[i].numPoint;
		
		curvatureJSDMat[i*numStls+i] = 0.0f;
		torsionJSDMat[i*numStls+i] = 0.0f;
		
		for (int j=i+1; j<numStls; ++j) {
			numj = stls[j].numPoint;

			cJSD = computeJSD(&curvatureHist[i*numBin], &curvatureHist[j*numBin], numi, numj, numBin);
			tJSD = computeJSD(&torisionHist[i*numBin], &torisionHist[j*numBin], numi, numj, numBin);

			curvatureJSDMat[i*numStls+j] = curvatureJSDMat[j*numStls+i] = cJSD;
			torsionJSDMat[i*numStls+j] = torsionJSDMat[j*numStls+i] = tJSD;

			percentage += one_streamline_percentage;
			printf("\rDistance Matrix Computation: %6.4f%%", percentage);
		}
	}
	printf("\n");
}

bool isConnected(vec3f* p1, vec3f* p2, const int& n1, const int&n2, const float& d){
	for (int i=0; i<n1; ++i) {
		for (int j=0; j<n2; ++j) {
			if (dist3d(p1[i], p2[j])<=d) {
				return true;
			}
		}
	}
	return false;
}

int* genConnectionDistance(Streamline* stls, Streamline* connects, vec3f* stls_points, vec3f* connect_points,
							 const int& numStls, const int& numConnects, const float& dist_thresh)
{
	int* ret = new int[numStls*numStls];
	memset(ret, 0, sizeof(int)*numStls*numStls);

	std::vector<int> intersected;
	
	vec3f *p1, *p2;
	Streamline s1, s2;
	int sid1, sid2;
	
	float percentage = 0.0f;
	float one_streamline_percentage = 100.0f/numStls;
	for (int i=0; i<numConnects; ++i) {
		s1 = connects[i];
		p1 = &connect_points[s1.start];
		intersected.clear();
		for (int j=0; j<numStls; ++j) {
			s2 = stls[j];
			p2 = &stls_points[s2.start];
			if (isConnected(p1, p2, s1.numPoint, s2.numPoint, dist_thresh)) {
				intersected.push_back(j);
			}
		}

		for (int j=0; j<intersected.size(); ++j) {
			sid1 = intersected[j];
			++ret[sid1*numStls+sid1];
			for (int k=j+1; k<intersected.size(); ++k) {
				sid2 = intersected[k];
				++ret[sid1*numStls+sid2];
				++ret[sid2*numStls+sid1];
			}
		}

		percentage += one_streamline_percentage;
		printf("\rDistance Matrix Computation: %6.4f%%", percentage);
	}
	printf("\n");
	return ret;
}

float* genLineToLineSmallestDistance(Streamline* stls1, Streamline* stls2, vec3f* points1, vec3f* points2,
								  const int& numStls1, const int& numStls2)
{
	float* ret = new float[numStls1*numStls2];

	float percentage = 0.0f;
	float one_streamline_percentage = 100.0f/numStls1;
	float dist;
	for (int i=0; i<numStls1; ++i) {
		for (int j=0; j<numStls2; ++j) {
			ret[j*numStls1+i] = computeClosestPointDistanceMin(&points1[stls1[i].start], &points2[stls2[j].start], stls1[i].numPoint, stls2[j].numPoint);
		}

		percentage += one_streamline_percentage;
		printf("\rDistance Matrix Computation: %6.4f%%", percentage);
	}
	printf("\n");

	return ret;
}

int* genConnectionDistance(float* dist_mat, const int& n, const int& m, const float& dist_thresh){
	int* ret = new int[m*m];

	std::vector<int> intersected;

	int u, v, p=0;
	for (int i=0; i<n; ++i) {
		intersected.clear();
		for (int j=0; j<m; ++j, ++p) {
			if (dist_mat[p]<dist_thresh) {
				intersected.push_back(j);
			}
		}

		for (int j=0; j<intersected.size(); ++j) {
			u = intersected[j];
			++ret[u*m+u];
			for (int k=j+1; k<intersected.size(); ++k) {
				v = intersected[k];
				++ret[u*m+v];
				++ret[v*m+u];
			}
		}
	}

	return ret;
}

//wrappers
extern "C" float *genMinMeanLineToGroupDistance_h(Streamline* stl1, Streamline* stl2, vec3f* points1, vec3f* points2, const int& num_stl1, const int& num_stl2, int* group_member, const int& group_size);
float* cudaGenMinMeanLineToGroupDistance(Streamline* stl1, Streamline* stl2, vec3f* points1, vec3f* points2, const int& num_stl1, const int& num_stl2, int* group_member, const int& group_size){
	return genMinMeanLineToGroupDistance_h(stl1, stl2, points1, points2, num_stl1, num_stl2, group_member, group_size);
}

extern "C"
void computeClosestPointDistanceWithHist_h(Streamline* stls, vec3f* points, const int& num_stls, const float& bin_width, const int& num_bin, float* avg, float* minv, float* maxv, short* hist);
void cudaComputeClosestPointDistanceMatrix(Streamline* stls, vec3f* points, const int& num_stls, const float& bin_width, const int& num_bin, float* avg, float* minv, float* maxv, short* hist){
	computeClosestPointDistanceWithHist_h(stls, points, num_stls, bin_width, num_bin, avg, minv, maxv, hist);
}

extern "C"
float* computeMinClosestPointDistanceMatrix_h(Streamline* stl1, Streamline* stl2, vec3f* points1, vec3f* points2, int num_stl1, int num_stl2);
float* cudaGenLineToLineSmallestDistance(Streamline* stls1, Streamline* stls2, vec3f* points1, vec3f* points2, const int& numStls1, const int& numStls2){
	return computeMinClosestPointDistanceMatrix_h(stls1, stls2, points1, points2, numStls1, numStls2);
}

extern "C" 
float* genClosestPointDistanceMatrixInSmallPools_h(Streamline* stls, vec3f* points, const int& num_stls);
float* cudaGenClosestPointDistanceMatrixInSmallPools(Streamline* stls, vec3f* points, const int& num_stls){
	return genClosestPointDistanceMatrixInSmallPools_h(stls, points, num_stls);
}

extern "C"
float* genClosestPointDistanceFromLineToSmallPools_h(Streamline* stls, vec3f* points, const int& num_stls, vec3f* line_points, const int& line_point_num);
float* cudaGenClosestPointDistanceFromLineToSmallPools(Streamline* stls, vec3f* points, const int& num_stls, vec3f* line_points, const int& line_point_num){
	return genClosestPointDistanceFromLineToSmallPools_h(stls, points, num_stls, line_points, line_point_num);
}

extern "C"
float computeMCPOneWay_h(vec3f* from, vec3f* to, int from_num, int to_num);
float cudaComputeMCPOneWay(vec3f* from, vec3f* to, int from_num, int to_num){
	return computeMCPOneWay_h(from, to, from_num, to_num);
}
extern "C"
float computeMCPOneWayWithDeviceMemoryInput_h(vec4f* from_d, vec4f* to_d, int from_num, int to_num);
float cudaComputeMCPOneWayWithDeviceMemoryInput(vec4f* from_d, vec4f* to_d, int from_num, int to_num){
	return computeMCPOneWayWithDeviceMemoryInput_h(from_d, to_d, from_num, to_num);
}

//******************************************************************
// discrete Frechet distance
//******************************************************************
#define FRECHET_NOT_INIT	-1.0f
#define FRECHET_INIT_THRESH	-0.9f
#define FRECHET_INFINITY	1e30

float discreteFrechetGetCa(vec3f* p1, vec3f* p2, const int& i, const int& j, float** ca){
	if (ca[i][j]>FRECHET_INIT_THRESH) {
		return ca[i][j];
	} else if (i==0 && j==0) {
		ca[i][j] = dist3d(p1[i],p2[j]);
	} else if (i>0 && j==0) {
		float c = discreteFrechetGetCa(p1, p2, i-1, 0, ca);
		float d = dist3d(p1[i],p2[j]);
		ca[i][j] = (c>d)?c:d;
	} else if (i==0 && j>0) {
		float c = discreteFrechetGetCa(p1, p2, 0, j-1, ca);
		float d = dist3d(p1[i],p2[j]);
		ca[i][j] = (c>d)?c:d;
	} else if (i>0 && j>0) {
		float c1 = discreteFrechetGetCa(p1, p2, i-1, j, ca);
		float c2 = discreteFrechetGetCa(p1, p2, i, j-1, ca);
		float c3 = discreteFrechetGetCa(p1, p2, i-1, j-1, ca);
		float d = dist3d(p1[i],p2[j]);
		if (c1>d) d = c1;
		if (c2>d) d = c2;
		if (c3>d) d = c3;
		ca[i][j] = d;
	} else {
		ca[i][j] = FRECHET_INFINITY;
	}
	return ca[i][j];
}

float discreteFrechetDistance(vec3f* p1, vec3f* p2, const int& n1, const int& n2, float** ca){
	for (int i=0; i<n1; ++i) {
		for (int j=0; j<n2; ++j) {
			ca[i][j] = FRECHET_NOT_INIT;
		}
	}

	return discreteFrechetGetCa(p1, p2, n1-1, n2-1, ca);
}

float* genDiscreteFrechetDistanceMatrix(Streamline* stls, const int& num_stls, vec3f* points){
	int num_point = stls[num_stls-1].start+stls[num_stls-1].numPoint;
	int max_point_stl = stls[0].numPoint;
	for (int i=1; i<num_stls; ++i) {
		if (stls[i].numPoint>max_point_stl) {
			max_point_stl = stls[i].numPoint;
		}
	}

	float *ret_mat_data, **ret_mat;
	float *ca_data, **ca;
	allocateMatrix(ret_mat_data, ret_mat, num_stls, num_stls);
	allocateMatrix(ca_data, ca, max_point_stl, max_point_stl);

	int num_pair=(num_stls)*(num_stls-1)>>1, count=0;

	vec3f *p1, *p2;
	int n1, n2;

	for (int i=0; i<num_stls; ++i) {
		p1 = &points[stls[i].start];
		n1 = stls[i].numPoint;
		ret_mat[i][i] = 0.0f;

		for (int j=i+1; j<num_stls; ++j) {
			p2 = &points[stls[j].start];
			n2 = stls[j].numPoint;

			ret_mat[i][j] = ret_mat[j][i] = discreteFrechetDistance(p1, p2, n1, n2, ca);

			++count;
			printf("\rDistance Matrix Computation: %d/%d", count, num_pair);
		}
	}
	printf("\n");

	delete[] ret_mat;
	delete[] ca;
	delete[] ca_data;

	return ret_mat_data;
}