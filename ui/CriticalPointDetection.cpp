#include "CriticalPointDetection.h"
#include <string>
#include <fstream>
// #include "D:/Tools/Eigen/eigen3/Eigen/Eigen"
#include "Eigen"
#include "Core"
#include "Eigenvalues"
#include "Dense"

//classify critical points

void getJacobian(const vec3f& pos, vec3f*** vf, const vec3i& dim, vec3f ret[3]){
	vec3i ipos = makeVec3i(pos);
	vec3f w[2];
	w[1]= pos-makeVec3f(ipos);
	w[0] = makeVec3f(1.0f,1.0f,1.0f)-w[1];
	
	memset(ret, 0, sizeof(vec3f)*3);

	float weight;
	vec3f jacob[3];
	for (int i=0; i<8; ++i) {
		getJacobianAtGridPoint(ipos+makeVec3i(i&1,(i&2)>>1,(i&4)>>2), vf, dim, jacob);
		weight = w[i&1].x*w[(i&2)>>1].y*w[(i&4)>>2].z;
		ret[0] = ret[0] + weight*jacob[0];
		ret[1] = ret[1] + weight*jacob[1];
		ret[2] = ret[2] + weight*jacob[2];
	}
}

void getJacobianAverage(const vec3f& pos, const int& range, vec3f*** vf, const vec3i& dim, vec3f ret[3]){
	vec3f jacob[3];
	int count=0;
	memset(ret, 0, sizeof(vec3f)*3);
	for (int i=-range; i<=range; ++i) {
		for (int j=-range; j<=range; ++j) {
			for (int k=-range; k<=range; ++k) {
				if (!(pos.x+k<0.0000001f || pos.x+k>=dim.x-1.0000001f 
					|| pos.y+j<0.0000001f || pos.y+j>=dim.y-1.0000001f 
					|| pos.z+i<0.0000001f || pos.z+i>=dim.z-1.0000001f))
				{
					getJacobian(pos, vf, dim, jacob);
					ret[0] = ret[0]+jacob[0];
					ret[1] = ret[1]+jacob[1];
					ret[2] = ret[2]+jacob[2];
					++count;
				}
			}
		}
	}

	ret[0] = (1.0f/count)*ret[0];
	ret[1] = (1.0f/count)*ret[1];
	ret[2] = (1.0f/count)*ret[2];
}

template<typename T>
bool readCriticalPoints(std::vector<T>& ret_points, std::vector<int>& types, std::vector<float>& scales, const char* file_name){
	std::ifstream file;
	file.open(file_name, std::ios_base::in|std::ios::binary);
	if (!file.is_open()) {
		printf("Fail to read file: %s \n", file_name);
		return false;
	}

	int num;
	file.read((char*)&num, sizeof(int));
	if (num==0) {
		ret_points.clear();
		types.clear();
		scales.clear();
		return true;
	}


	ret_points.resize(num);
	types.resize(num);
	scales.resize(num);
	file.read((char*)&ret_points[0], sizeof(T)*num);
	file.read((char*)&types[0], sizeof(int)*num);
	file.read((char*)&scales[0], sizeof(float)*num);
	file.close();

	return true;
}

template<typename T>
bool saveCriticalPoints(std::vector<T>& points, std::vector<int>& types, std::vector<float>& scales, const char* file_name){
	std::ofstream file;
	file.open(file_name, std::ios_base::out|std::ios::binary);
	if (!file.is_open()) {
		printf("Fail to read file: %s \n", file_name);
		return false;
	}

	int num = points.size();
	file.write((char*)&num, sizeof(int));

	if (num!=0) {
		file.write((char*)&points[0], sizeof(T)*num);
		file.write((char*)&types[0], sizeof(int)*num);
		file.write((char*)&scales[0], sizeof(float)*num);
	}

	file.close();

	return true;
}

template bool readCriticalPoints<vec3f>(std::vector<vec3f>& ret_points, std::vector<int>& types, std::vector<float>& scales, const char* file_name);
template bool saveCriticalPoints<vec3f>(std::vector<vec3f>& points, std::vector<int>& types, std::vector<float>& scales, const char* file_name);
template bool readCriticalPoints<vec4f>(std::vector<vec4f>& ret_points, std::vector<int>& types, std::vector<float>& scales, const char* file_name);
template bool saveCriticalPoints<vec4f>(std::vector<vec4f>& points, std::vector<int>& types, std::vector<float>& scales, const char* file_name);

#define CP_EQUAL_ZERO_THREASH 0.00001f
int classifyCriticalPoint(const vec3f& pos, vec3f*** vf, const vec3i& dim){
	vec3f jacob[3];
	getJacobianAverage(pos, 2, vf, dim, jacob);
	
	Eigen::Matrix3f m = Eigen::Matrix3f::Zero(3, 3);
	m(0,0) = jacob[0].x;m(0,1) = jacob[0].y;m(0,2) = jacob[0].z;
	m(1,0) = jacob[1].x;m(1,1) = jacob[1].y;m(1,2) = jacob[1].z;
	m(2,0) = jacob[2].x;m(2,1) = jacob[2].y;m(2,2) = jacob[2].z;
	Eigen::EigenSolver<Eigen::MatrixXf> es(m);

	vec2f eigen_value[3] = {{es.eigenvalues()[0].real(),es.eigenvalues()[0].imag()},
		{es.eigenvalues()[1].real(),es.eigenvalues()[1].imag()},
	{es.eigenvalues()[2].real(),es.eigenvalues()[2].imag()}};

	int real_pos=0, real_neg=0, real_zero=0;
	int imag_non_zero = 0;
	for (int i=0; i<3; ++i) {
		if (eigen_value[i].x<-CP_EQUAL_ZERO_THREASH) {++real_neg;}
		else if (eigen_value[i].x>CP_EQUAL_ZERO_THREASH) {++real_pos;}
		else {++real_zero;}

		if (eigen_value[i].y<-CP_EQUAL_ZERO_THREASH || eigen_value[i].y>CP_EQUAL_ZERO_THREASH) {
			++imag_non_zero;
		}
	}

	int type = CP_CENTER;
	if (real_pos==3) {
		type = CP_REPEL_NODE;
	} else if (real_neg==3) {
		type = CP_ATTRACT_NODE;
	} else if (real_pos==2 && real_neg==1){
		type = CP_REPEL_NODE_SADDLE;
	} else if (real_pos==1 && real_neg==2){
		type = CP_ATTRACT_NODE_SADDLE;
	}
	if (imag_non_zero!=0) {
		type |= CP_FOCUS;
	}

	return type;
}

#define GREENE_INTVL                2
#define GREENE_DEGREE_THRESH        0.00001
#define GREENE_SCALE_THRESH         0.0001f
#define GRID_INTVL                  0.05f

double computeSolidAngle(vec3f* v){
	float l1 = vec3fLen(v[0]);
	float l2 = vec3fLen(v[1]);
	float l3 = vec3fLen(v[2]);
	double t1 = acos((v[1]*v[2])/(l2*l3));
	double t2 = acos((v[0]*v[2])/(l1*l3));
	double t3 = acos((v[0]*v[1])/(l1*l2));

	double t = tan(0.25*(t1+t2+t3));
	t *= tan(0.25*(t1+t2-t3));
	t *= tan(0.25*(t2+t3-t1));
	t *= tan(0.25*(t3+t1-t2));

	double a = atan(sqrt(t))*4.0;
	if((v[0]*cross(v[1],v[2]))<0) a=-a;

	return a;
}

//front index     back index
//    2 3            6 7
//    0 1            4 5
int compute_degree_idx[12][3] = {
	{0,1,2},{1,3,2},//front
	{1,5,3},{5,7,3},//right
	{5,4,7},{4,6,7},//back
	{4,0,6},{0,2,6},//left
	{3,7,2},{7,6,2},//top
	{4,5,0},{5,1,0}//bottom
};

double computeDegree(vec3f* v){
	if((v[0].x>=-GRID_INTVL||v[1].x>=-GRID_INTVL||v[2].x>=-GRID_INTVL||v[3].x>=-GRID_INTVL||v[4].x>=-GRID_INTVL||v[5].x>=-GRID_INTVL||v[6].x>=-GRID_INTVL||v[7].x>=-GRID_INTVL) &&
		(v[0].x<=GRID_INTVL||v[1].x<=GRID_INTVL||v[2].x<=GRID_INTVL||v[3].x<=GRID_INTVL||v[4].x<=GRID_INTVL||v[5].x<=GRID_INTVL||v[6].x<=GRID_INTVL||v[7].x<=GRID_INTVL) &&
		(v[0].y>=-GRID_INTVL||v[1].y>=-GRID_INTVL||v[2].y>=-GRID_INTVL||v[3].y>=-GRID_INTVL||v[4].y>=-GRID_INTVL||v[5].y>=-GRID_INTVL||v[6].y>=-GRID_INTVL||v[7].y>=-GRID_INTVL) &&
		(v[0].y<=GRID_INTVL||v[1].y<=GRID_INTVL||v[2].y<=GRID_INTVL||v[3].y<=GRID_INTVL||v[4].y<=GRID_INTVL||v[5].y<=GRID_INTVL||v[6].y<=GRID_INTVL||v[7].y<=GRID_INTVL) &&
		(v[0].z>=-GRID_INTVL||v[1].z>=-GRID_INTVL||v[2].z>=-GRID_INTVL||v[3].z>=-GRID_INTVL||v[4].z>=-GRID_INTVL||v[5].z>=-GRID_INTVL||v[6].z>=-GRID_INTVL||v[7].z>=-GRID_INTVL) &&
		(v[0].z<=GRID_INTVL||v[1].z<=GRID_INTVL||v[2].z<=GRID_INTVL||v[3].z<=GRID_INTVL||v[4].z<=GRID_INTVL||v[5].z<=GRID_INTVL||v[6].z<=GRID_INTVL||v[7].z<=GRID_INTVL))
	{
		double a = 0.0;

		vec3f tri[3];
		for (int i=0; i<12; ++i) {
			tri[0] = v[compute_degree_idx[i][0]];
			tri[1] = v[compute_degree_idx[i][1]];
			tri[2] = v[compute_degree_idx[i][2]];
			a += computeSolidAngle(tri);
		}
		a /= 12.56637061;//4*pi
		return a;
	}
	return 0.0;
}

//bisect cell index
//   front      middle       back
//  2 12  3    19 20 21     6 26  7
//  9 10 11    16 17 18    23 24 25
//  0  8  1    13 14 15     4 22  5
int locate_point_interpolate[19][3] = {
	{8,0,1},{9,0,2},{11,1,3},{10,9,11},{12,2,3},//front
	{22,4,5},{23,4,6},{25,5,7},{24,23,25},{26,6,7},//back
	{13,0,4},{14,8,22},{15,1,5},{16,9,23},{17,10,24},{18,11,25},{19,2,6},{20,12,26},{21,3,7}//middle
};

int subcell_idx[8][8] = {
	{0,8,9,10,13,14,16,17},//front-left-bottom
	{9,10,2,12,16,17,19,20},//front-left-top
	{8,1,10,11,14,15,17,18},//front-right-bottom
	{10,11,12,3,17,18,20,21},//front-right-top
	{13,14,16,17,4,22,23,24},//back-left-bottom
	{16,17,19,20,23,24,6,26},//back-left-top
	{14,15,17,18,22,5,24,25},//back-right-bottom
	{17,18,20,21,24,25,26,7}//back-right-top
};

vec3f subcell_pos[8] = {
	{-0.25f, -0.25f, -0.25f},//front-left-bottom
	{-0.25f,  0.25f, -0.25f},//front-left-top
	{ 0.25f, -0.25f, -0.25f},//front-right-bottom
	{ 0.25f,  0.25f, -0.25f},//front-right-top
	{-0.25f, -0.25f,  0.25f},//back-left-bottom
	{-0.25f,  0.25f,  0.25f},//back-left-top
	{ 0.25f, -0.25f,  0.25f},//back-right-bottom
	{ 0.25f,  0.25f,  0.25f}//back-right-top
};

bool locateCriticalPoint(vec3f* v, const vec3f& pos, const float& scale, std::vector<vec3f>& ret){
	if (scale<=GREENE_SCALE_THRESH) {
		ret.push_back(pos);
		return true;
	}

	int i, j;

	float subScale = scale*0.5f;

	vec3f c[27];//bisection cell
	for (i=0; i<8; ++i)//copy
		c[i]=v[i];
	for (i=0; i<19; ++i)//interpolate
		c[locate_point_interpolate[i][0]] = 0.5f*(c[locate_point_interpolate[i][1]]+c[locate_point_interpolate[i][2]]);

	double degree;
	vec3f sub[8];//subcell
	for(i=0; i<8; ++i){
		for (j=0; j<8; ++j) {
			sub[j] = c[subcell_idx[i][j]];
		}
		degree = computeDegree(sub);
		if (abs(degree)>GREENE_DEGREE_THRESH){
			if(locateCriticalPoint(sub, pos+scale*subcell_pos[i], subScale, ret)){
				return true;
			}
		}
	}
	return false;
}

void locateAllCriticalPoints(vec3f* vecField, const vec3i& dim, std::vector<vec3f>& ret, std::vector<int>& ret_type){
	vec3f v[8];
	vec3f pos;
	double degree;

	vec3f** vf_rows;
	vec3f*** vf;
	allocateVolumeAccess(vecField, vf_rows, vf, dim.x, dim.y, dim.z);

	float init_scale = GREENE_INTVL, init_offset=0.5f*GREENE_INTVL;
	for (int i=0; i<dim.z-GREENE_INTVL; ++i){
		for (int j=0; j<dim.y-GREENE_INTVL; ++j){
			for(int k=0; k<dim.x-GREENE_INTVL; ++k){
				v[0] = vf[i][j][k];
				v[1] = vf[i][j][k+GREENE_INTVL];
				v[2] = vf[i][j+GREENE_INTVL][k];
				v[3] = vf[i][j+GREENE_INTVL][k+GREENE_INTVL];
				v[4] = vf[i+GREENE_INTVL][j][k];
				v[5] = vf[i+GREENE_INTVL][j][k+GREENE_INTVL];
				v[6] = vf[i+GREENE_INTVL][j+GREENE_INTVL][k];
				v[7] = vf[i+GREENE_INTVL][j+GREENE_INTVL][k+GREENE_INTVL];

				//greene's method
				degree = computeDegree(v);
				if(abs(degree)>GREENE_DEGREE_THRESH){
					pos = makeVec3f(k+init_offset, j+init_offset, i+init_offset);
					locateCriticalPoint(v, pos, init_scale, ret);
				}
			}
		}
	}

	for (int i=0; i<ret.size(); ++i) {
		ret_type.push_back(classifyCriticalPoint(ret[i], vf, dim));
	}

	delete[] vf_rows;
	delete[] vf;
}

//void locateAllNearByCriticalPoints(vec3f* vecField, const vec3i& dim, std::vector<vec3f>& pos, std::vector<vec3f>& ret, std::vector<int>& ret_type){
//	vec3f v[8];
//	vec3f pos;
//	double degree;
//
//	vec3f** vf_rows;
//	vec3f*** vf;
//	allocateVolumeAccess(vecField, vf_rows, vf, dim.x, dim.y, dim.z);
//
//	float init_scale = GREENE_INTVL, init_offset=0.5f*GREENE_INTVL;
//	for(int i=0; i<pos.size(); ++i){
//	}
//	for (int i=0; i<dim.z-GREENE_INTVL; ++i){
//		for (int j=0; j<255; ++j){
//			for(int k=0; k<dim.x-GREENE_INTVL; ++k){
//				v[0] = vf[i][j][k];
//				v[1] = vf[i][j][k+GREENE_INTVL];
//				v[2] = vf[i][j+GREENE_INTVL][k];
//				v[3] = vf[i][j+GREENE_INTVL][k+GREENE_INTVL];
//				v[4] = vf[i+GREENE_INTVL][j][k];
//				v[5] = vf[i+GREENE_INTVL][j][k+GREENE_INTVL];
//				v[6] = vf[i+GREENE_INTVL][j+GREENE_INTVL][k];
//				v[7] = vf[i+GREENE_INTVL][j+GREENE_INTVL][k+GREENE_INTVL];
//
//				//greene's method
//				degree = computeDegree(v);
//				if(abs(degree)>GREENE_DEGREE_THRESH){
//					pos = makeVec3f(k+init_offset, j+init_offset, i+init_offset);
//					locateCriticalPoint(v, pos, init_scale, ret);
//				}
//			}
//		}
//	}
//
//	for (int i=0; i<ret.size(); ++i) {
//		ret_type.push_back(classifyCriticalPoint(ret[i], vf, dim));
//	}
//
//	delete[] vf_rows;
//	delete[] vf;
//}

void groupCriticalPoints(std::vector<vec3f>& cp, std::vector<int>& type, const float& dist_thresh){
	std::vector<bool> remove_mark(cp.size(), false);
	for (int i=0; i<cp.size(); ++i) if(!remove_mark[i]){
		for (int j=i+1; j<cp.size(); ++j) if(!remove_mark[j]){
			if (dist3d(cp[i],cp[j])<dist_thresh /*&& type[i]==type[j]*/) {
				remove_mark[j] = true;
			}
		}
	}
	std::vector<vec3f> cp_copy;
	std::vector<int> type_copy;
	cp_copy.swap(cp);
	type_copy.swap(type);
	for (int i=0; i<remove_mark.size(); ++i) if (!remove_mark[i]) {
		cp.push_back(cp_copy[i]);
		type.push_back(type_copy[i]);
	}
}

void groupCriticalPoints(std::vector<vec3f>& cp, std::vector<int>& type, std::vector<float>& scales, const float& dist_thresh){
	if (cp.empty()) return;

	std::vector<bool> remove_mark(cp.size(), false);
	for (int i=0; i<cp.size(); ++i) if(!remove_mark[i]){
		for (int j=i+1; j<cp.size(); ++j) if(!remove_mark[j]){
			if (dist3d(cp[i],cp[j])<dist_thresh /*&& type[i]==type[j]*/) {
				remove_mark[j] = true;
			}
		}
	}
	std::vector<vec3f> cp_copy;
	std::vector<int> type_copy;
	std::vector<float> scale_copy;
	cp_copy.swap(cp);
	type_copy.swap(type);
	scale_copy.swap(scales);
	for (int i=0; i<remove_mark.size(); ++i) if (!remove_mark[i]) {
		cp.push_back(cp_copy[i]);
		type.push_back(type_copy[i]);
		scales.push_back(scale_copy[i]);

	}
}

vec3f getVector(const vec3f& p, vec3f*** vf){
	vec3i p_int = makeVec3i(p); 
	vec3f w[2];
	w[1] = p-makeVec3f(p_int);
	w[0] = makeVec3f(1,1,1)-w[1];

	vec3f ret = makeVec3f(0,0,0);
	for (int i=0; i<8; ++i) {
		vec3i inc = makeVec3i(i&1, i&2, i&4);
		ret = ret + (w[inc.x].x*w[inc.y].y*w[inc.z].z)*vf[p_int.z+inc.z][p_int.y+inc.y][p_int.x+inc.x];
	}

	return ret;
}


vec3f ga_index_face[6][3] = {
	{{-1,-1,-1},{ 1, 0, 0},{ 0, 1, 0}},//front
	{{ 1,-1, 1},{-1, 0, 0},{ 0, 1, 0}},//back
	{{-1,-1,-1},{ 0, 1, 0},{ 0, 0, 1}},//left
	{{ 1,-1,-1},{ 0, 0, 1},{ 0, 1, 0}},//right
	{{ 1,-1, 1},{ 0, 0,-1},{-1, 0, 0}},//bottom
	{{ 1, 1, 1},{-1, 0,-1},{ 0, 0,-1}} //top
};

float computeTriVectorMagnitude(const vec3f& a, const vec3f& b, const vec3f& c){
	return (a.x*b.y*c.z-a.x*b.z*c.y+a.y*b.x*c.z-a.y*b.z*c.x+a.z*b.x*c.y-a.z*b.y*c.x);
}

float computeIndexGA(const vec3f& p, vec3f*** vf, const float& cube_size, const int& sample_num_per_edge){
	int n = (sample_num_per_edge<2)?2:sample_num_per_edge;
	float half = cube_size*0.5f;
	float intv = cube_size/(n-1);

	float index = 0.0f;

	vec3f** face = new vec3f*[n];
	vec3f* face_data = new vec3f[n*n];
	for (int i=0; i<n; ++i) face[i] = &face_data[i*n];
	vec3f anchor, cur_pos, cur_vec;
	vec3f e1, e2;
	for (int i=0; i<6; ++i) {//for each face
		anchor = p+(half*ga_index_face[i][0]);
		e1 = intv*ga_index_face[i][1];
		e2 = intv*ga_index_face[i][2];
		for (int j=0; j<n; ++j) {
			for (int k=0; k<n; ++k) {
				cur_pos = anchor+j*e1+k*e2;
				cur_vec = getVector(cur_pos, vf);
				normalize(cur_vec);
				face[j][k] = cur_vec;
			}
		}

		for (int j=0; j<n-1; ++j) {
			for (int k=0; k<n-1; ++k) {
				index += computeTriVectorMagnitude(face[j][k],face[j][k+1],face[j+1][k]);
				index += computeTriVectorMagnitude(face[j+1][k+1],face[j+1][k],face[j][k+1]);
			}
		}
	}

	delete[] face;
	delete[] face_data;

	index *= 0.039788736f; 
	return index;
}