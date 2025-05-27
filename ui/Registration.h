#ifndef REGISTRATION_H
#define REGISTRATION_H

#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include "typeOperation.h"
#include "definition.h"

#define PI 3.1415926f

//compute rotation matrix for every grid
inline __host__ __device__ void tridiagonal4 (float mat[4][4], float* diag, float* subd)
// input: mat = 4x4 real, symmetric A
// output: mat = orthogonal matrix Q
// diag = diagonal entries of T, diag[0,1,2,3]
// subd = subdiagonal entry of T, subd[0,1,2]
{
	// save matrix M
	float a = mat[0][0], b = mat[0][1], c = mat[0][2], d = mat[0][3],
					     e = mat[1][1], f = mat[1][2], g = mat[1][3],
		                                h = mat[2][2], i = mat[2][3],
		                                               j = mat[3][3];

	diag[0] = a;
	subd[3] = 0;
	mat[0][0] = 1; mat[0][1] = 0; mat[0][2] = 0; mat[0][3] = 0;
	mat[1][0] = 0;
	mat[2][0] = 0;
	mat[3][0] = 0;
	if ( c != 0 || d != 0 ) {
		float q11, q12, q13;
		float q21, q22, q23;
		float q31, q32, q33;
		// build column Q1
		float len = sqrt(b*b+c*c+d*d);
		q11 = b/len;
		q21 = c/len;
		q31 = d/len;
		subd[0] = len;
		// compute S*Q1
		float v0 = e*q11+f*q21+g*q31;
		float v1 = f*q11+h*q21+i*q31;
		float v2 = g*q11+i*q21+j*q31;
		diag[1] = q11*v0+q21*v1+q31*v2;
		// build column Q3 = Q1x(S*Q1)
		q13 = q21*v2-q31*v1;
		q23 = q31*v0-q11*v2;
		q33 = q11*v1-q21*v0;
		len = sqrt(q13*q13+q23*q23+q33*q33);
		if ( len > 0 ) {
			q13 /= len;
			q23 /= len;
			q33 /= len;
			// build column Q2 = Q3xQ1
			q12 = q23*q31-q33*q21;
			q22 = q33*q11-q13*q31;
			q32 = q13*q21-q23*q11;
			v0 = q12*e+q22*f+q32*g;
			v1 = q12*f+q22*h+q32*i;
			v2 = q12*g+q22*i+q32*j;
			subd[1] = q11*v0+q21*v1+q31*v2;
			diag[2] = q12*v0+q22*v1+q32*v2;
			subd[2] = q13*v0+q23*v1+q33*v2;
			v0 = q13*e+q23*f+q33*g;
			v1 = q13*f+q23*h+q33*i;
			v2 = q13*g+q23*i+q33*j;
			diag[3] = q13*v0+q23*v1+q33*v2;
		}
		else { // S*Q1 parallel to Q1, choose any valid Q2 and Q3
			subd[1] = 0;
			len = q21*q21+q31*q31;
			if ( len > 0 ) {
				float tmp = q11-1;
				q12 = -q21;
				q22 = 1+tmp*q21*q21/len;
				q32 = tmp*q21*q31/len;
				q13 = -q31;
				q23 = q32;
				q33 = 1+tmp*q31*q31/len;
				v0 = q12*e+q22*f+q32*g;
				v1 = q12*f+q22*h+q32*i;
				v2 = q12*g+q22*i+q32*j;
				diag[2] = q12*v0+q22*v1+q32*v2;
				subd[2] = q13*v0+q23*v1+q33*v2;
				v0 = q13*e+q23*f+q33*g;
				v1 = q13*f+q23*h+q33*i;
				v2 = q13*g+q23*i+q33*j;
				diag[3] = q13*v0+q23*v1+q33*v2;
			}
			else { // Q1 = (+-1,0,0)
				q12 = 0; q22 = 1; q32 = 0;
				q13 = 0; q23 = 0; q33 = 1;
				diag[2] = h;
				diag[3] = j;
				subd[2] = i;
			}
		}
		mat[1][1] = q11; mat[1][2] = q12; mat[1][3] = q13;
		mat[2][1] = q21; mat[2][2] = q22; mat[2][3] = q23;
		mat[3][1] = q31; mat[3][2] = q32; mat[3][3] = q33;
	}
	else {
		diag[1] = e;
		subd[0] = b;
		mat[1][1] = 1;
		mat[2][1] = 0;
		mat[3][1] = 0;
		if ( g != 0 ) {
			float ell = sqrt(f*f+g*g);
			f /= ell;
			g /= ell;
			float Q = 2*f*i+g*(j-h);
			diag[2] = h+g*Q;
			diag[3] = j-g*Q;
			subd[1] = ell;
			subd[2] = i-f*Q;
			mat[1][2] = 0; mat[1][3] = 0;
			mat[2][2] = f; mat[2][3] = g;
			mat[3][2] = g; mat[3][3] = -f;
		}
		else {
			diag[2] = h;
			diag[3] = j;
			subd[1] = f;
			subd[2] = i;
			mat[1][2] = 0; mat[1][3] = 0;
			mat[2][2] = 1; mat[2][3] = 0;
			mat[3][2] = 0; mat[3][3] = 1;
		}
	}
}

inline __host__ __device__ bool QLAlgorithm4 (float* m_afDiag, float* m_afSubd,
							float m_aafMat[4][4])
{
	const int iMaxIter = 32;

	for (int i0 = 0; i0 < 4; i0++)
	{
		int i1;
		for (i1 = 0; i1 < iMaxIter; i1++)
		{
			int i2;
			for (i2 = i0; i2 <= 2; i2++)
			{
				float fTmp = abs(m_afDiag[i2])+abs(m_afDiag[i2+1]);
				if ( abs(m_afSubd[i2]) + fTmp == fTmp )
					break;
			}
			if ( i2 == i0 )
				break;

			float fG = (m_afDiag[i0+1]-m_afDiag[i0])/(2.0f*m_afSubd[i0]);
			float fR = sqrtf(fG*fG+1.0f);
			if ( fG < 0.0f )
				fG = m_afDiag[i2]-m_afDiag[i0]+m_afSubd[i0]/(fG-fR);
			else
				fG = m_afDiag[i2]-m_afDiag[i0]+m_afSubd[i0]/(fG+fR);
			float fSin = 1.0f, fCos = 1.0f, fP = 0.0f;
			for (int i3 = i2-1; i3 >= i0; i3--)
			{
				float fF = fSin*m_afSubd[i3];
				float fB = fCos*m_afSubd[i3];
				if ( abs(fF) >= abs(fG) )
				{
					fCos = fG/fF;
					fR = sqrtf(fCos*fCos+1.0f);
					m_afSubd[i3+1] = fF*fR;
					fSin = 1.0f/fR;
					fCos *= fSin;
				}
				else
				{
					fSin = fF/fG;
					fR = sqrtf(fSin*fSin+1.0f);
					m_afSubd[i3+1] = fG*fR;
					fCos = 1.0f/fR;
					fSin *= fCos;
				}
				fG = m_afDiag[i3+1]-fP;
				fR = (m_afDiag[i3]-fG)*fSin+2.0f*fB*fCos;
				fP = fSin*fR;
				m_afDiag[i3+1] = fG+fP;
				fG = fCos*fR-fB;

				for (int i4 = 0; i4 < 4; i4++)
				{
					fF = m_aafMat[i4][i3+1];
					m_aafMat[i4][i3+1] = fSin*m_aafMat[i4][i3]+fCos*fF;
					m_aafMat[i4][i3] = fCos*m_aafMat[i4][i3]-fSin*fF;
				}
			}
			m_afDiag[i0] -= fP;
			m_afSubd[i0] = fG;
			m_afSubd[i2] = 0.0f;
		}
		if ( i1 == iMaxIter )
			return false;
	}

	return true;
}

//x^4+c2x^2+c1x+c0=0
inline __host__ __device__ float ferrariMethod(float c2, float c1, float c0){
	float alpha = c2, beta = c1, gamma = c0;
	if (abs(beta)<0.00000001f){
		return sqrtf(0.5f*(-alpha+sqrtf(alpha*alpha-4.0f*gamma)));
	}

	float p = -alpha*alpha/12.0f-gamma;
	float q = -alpha*alpha*alpha/108.0f+alpha*gamma/3.0f-beta*beta/8.0f;
	float r = -0.5f*q+sqrtf(0.25f*q*q+p*p*p/27.0f);
	float u = pow(r, 0.3333333333f);

	float y;
	if (abs(u)<0.00000001f){
		y = -5.0f/6.0f*alpha+u-pow(q, 0.3333333333f);
	} else {
		y = -5.0f/6.0f*alpha+u-p/(3.0f*u);
	}

	float w = sqrtf(alpha+2.0f*y);

	float t1 = -(3.0f*alpha+2.0f*y+2.0f*beta/w);
	float t2 = -(3.0f*alpha+2.0f*y-2.0f*beta/w);

	if(t1 < 0.0f){
		return (0.5f*(-w+sqrtf(t2)));
	} else if (t2 < 0.0f){
		return (0.5f*(w+sqrtf(t1)));
	}

	float x1 = 0.5f*(-w+sqrtf(t2));
	float x2 = 0.5f*(w+sqrtf(t1));

	return ((x1>x2)?x1:x2);
}

inline __host__ __device__ void computeEigenVectorDevice(float mat[4][4], float x[4]){
	x[0]=x[1]=x[2]=x[3]=1.0f;
	float w[4];
	float maxv;
	bool isfinished = false;

	for(int i=0; i<3000; i++){
		w[0] = mat[0][0]*x[0]+mat[0][1]*x[1]+mat[0][2]*x[2]+mat[0][3]*x[3];
		w[1] = mat[1][0]*x[0]+mat[1][1]*x[1]+mat[1][2]*x[2]+mat[1][3]*x[3];
		w[2] = mat[2][0]*x[0]+mat[2][1]*x[1]+mat[2][2]*x[2]+mat[2][3]*x[3];
		w[3] = mat[3][0]*x[0]+mat[3][1]*x[1]+mat[3][2]*x[2]+mat[3][3]*x[3];

		maxv = w[0];
		for(int j=1; j<4; j++){
			if(abs(w[j])>abs(maxv)) maxv=w[j];
		}
		w[0] /= maxv;
		w[1] /= maxv;
		w[2] /= maxv;
		w[3] /= maxv;

		if(abs(w[0]-x[0])<0.00001f && abs(w[1]-x[1])<0.00001f && abs(w[2]-x[2])<0.00001f && abs(w[3]-x[3])<0.0001f) isfinished=true;

		x[0] = w[0];
		x[1] = w[1];
		x[2] = w[2];
		x[3] = w[3];

		if(isfinished){
			//printf("Converged in %3d iterations. x=(%8.5f, %8.5f, %8.5f, %8.5f)\n\n", i, x[0], x[1], x[2], x[3]);
			break;
		}
	}
	//if(!isfinished) printf("Not converged. x=(%8.5f, %8.5f, %8.5f, %8.5f)\n\n", x[0], x[1], x[2], x[3]);
	//normalize
	float len = sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]+x[3]*x[3]);
	x[0] /= len;
	x[1] /= len;
	x[2] /= len;
	x[3] /= len;
}

inline __host__ __device__ void rotMatrixFromQuaternion(float *mat, float quat[4]){
	mat[0] = 1.0f - 2.0f * (quat[2] * quat[2] + quat[3] * quat[3]);
	mat[1] = 2.0f * (quat[1] * quat[2] - quat[0] * quat[3]);
	mat[2] = 2.0f * (quat[1] * quat[3] + quat[0] * quat[2]);

	mat[3] = 2.0f * (quat[1] * quat[2] + quat[0] * quat[3]);
	mat[4] = 1.0f - 2.0f * (quat[1] * quat[1] + quat[3] * quat[3]);
	mat[5] = 2.0f * (quat[2] * quat[3] - quat[0] * quat[1]);

	mat[6] = 2.0f * (quat[1] * quat[3] - quat[0] * quat[2]);
	mat[7] = 2.0f * (quat[2] * quat[3] + quat[0] * quat[1]);
	mat[8] = 1.0f - 2.0f * (quat[1] * quat[1] + quat[2] * quat[2]);
}

inline __host__ __device__ float normalizePointSet(vec3f* pts, const int& num, vec3f* retPts, vec3f& center){
	center = makeVec3f(0.0f, 0.0f, 0.0f);
	for (int i=0; i<num; ++i){
		center = center+pts[i];
	}
	center = (1.0f/num)*center;

	float s;
	s= 0.0f;
	for (int i=0; i<num; ++i){
		retPts[i] = pts[i]-center;
		s += retPts[i]*retPts[i];
	}
	s = sqrtf(s);

	for (int i=0; i<num; ++i){
		retPts[i] = (10.0f/s)*retPts[i];
	}

	return (s/10.0f);
}

inline __host__ __device__ void normalizePointSet(vec3f* pts, const int& num, vec3f* retPts){
	vec3f c; //center

	c = makeVec3f(0.0f, 0.0f, 0.0f);
	for (int i=0; i<num; ++i){
		c = c+pts[i];
	}
	c = (1.0f/num)*c;

	float s;
	s= 0.0f;
	for (int i=0; i<num; ++i){
		retPts[i] = pts[i]-c;
		s += retPts[i]*retPts[i];
	}
	s = sqrtf(s);

	for (int i=0; i<num; ++i){
		retPts[i] = (10.0f/s)*retPts[i];
	}
}

inline __host__ __device__ void normalizePointSet(vec3f* pts1, vec3f* pts2, const int& num, vec3f* retPts1, vec3f* retPts2){
	vec3f c1, c2; //centers

	c1 = makeVec3f(0.0f, 0.0f, 0.0f);
	c2 = makeVec3f(0.0f, 0.0f, 0.0f);
	for (int i=0; i<num; ++i){
		c1 = c1+pts1[i];
		c2 = c2+pts2[i];
	}
	c1 = (1.0f/num)*c1;
	c2 = (1.0f/num)*c2;

	float s1, s2;
	s1 = s2 = 0.0f;
	for (int i=0; i<num; ++i){
		retPts1[i] = pts1[i]-c1;
		retPts2[i] = pts2[i]-c2;
		s1 += retPts1[i]*retPts1[i];
		s2 += retPts2[i]*retPts2[i];
	}
	s1 = sqrtf(s1); s2 = sqrtf(s2);

	for (int i=0; i<num; ++i){
		retPts1[i] = (10.0f/s1)*retPts1[i];
		retPts2[i] = (10.0f/s2)*retPts2[i];
	}
}

inline __host__ __device__ void computeRotationQuat(vec3f* pts1, vec3f* pts2, const int& num, float retQuat[4]){
	float matrix[4][4];
	float diag[4], subd[4];
	float sxx, sxy, sxz, syx, syy, syz, szx, szy, szz;
	sxx=sxy=sxz=syx=syy=syz=szx=szy=szz=0.0f;
	for(int i=0; i<num; ++i){
		sxx += pts1[i].x*pts2[i].x;
		sxy += pts1[i].x*pts2[i].y;
		sxz += pts1[i].x*pts2[i].z;
		syx += pts1[i].y*pts2[i].x;
		syy += pts1[i].y*pts2[i].y;
		syz += pts1[i].y*pts2[i].z;
		szx += pts1[i].z*pts2[i].x;
		szy += pts1[i].z*pts2[i].y;
		szz += pts1[i].z*pts2[i].z;
	}

	matrix[0][0] = sxx+syy+szz;
	matrix[0][1] = syz-szy;
	matrix[0][2] = szx-sxz;
	matrix[0][3] = sxy-syx;
	matrix[1][0] = syz-szy;
	matrix[1][1] = sxx-syy-szz;
	matrix[1][2] = sxy+syx;
	matrix[1][3] = szx+sxz;
	matrix[2][0] = szx-sxz;
	matrix[2][1] = sxy+syx;
	matrix[2][2] = -sxx+syy-szz;
	matrix[2][3] = syz+szy;
	matrix[3][0] = sxy-syx;
	matrix[3][1] = szx+sxz;
	matrix[3][2] = syz+szy;
	matrix[3][3] = -sxx-syy+szz;

	tridiagonal4(matrix, diag, subd);
	QLAlgorithm4(diag, subd, matrix);
	int max_eigen = 0;
	for (int i=1; i<4; ++i){
		if (diag[i]>diag[max_eigen]){
			max_eigen = i;
		}
	}
	retQuat[0] = matrix[0][max_eigen];
	retQuat[1] = matrix[1][max_eigen];
	retQuat[2] = matrix[2][max_eigen];
	retQuat[3] = matrix[3][max_eigen];
}

inline __host__ __device__ vec3f rotateVector(const vec3f& vec, float rot[9]){
	vec3f ret;
	ret.x = vec.x*rot[0]+vec.y*rot[1]+vec.z*rot[2];
	ret.y = vec.x*rot[3]+vec.y*rot[4]+vec.z*rot[5];
	ret.z = vec.x*rot[6]+vec.y*rot[7]+vec.z*rot[8];
	return ret;
}

inline __host__ __device__ float computeWeightedPointWiseDistance(vec3f* pts1, vec3f* pts2, const int& num, float rot[9]){
	float diff = 0.0f;
	vec3f tmp;
	float *d1 = new float[num+1];
	float *d2 = new float[num+1];
	float d1t = 0.0f, d2t = 0.0f;
	for(int i=0; i<num-1; ++i){
		d1[i+1] = dist3d(pts1[i], pts1[i+1]);
		d2[i+1] = dist3d(pts2[i], pts2[i+1]);
		d1t += d1[i+1];
		d2t += d2[i+1];
	}
	d1[0] = d1[1]; d2[0]=d2[1];
	d1[num] = d1[num-1]; d2[num] = d2[num-1];

	for (int i=0; i<num; ++i){
		tmp = rotateVector(pts1[i], rot);
		diff += (d1t/(d1[i]+d1[i+1])+d2t/(d2[i]+d2[i+1]))*dist3d(tmp, pts2[i]);
	}
	delete[] d1;
	delete[] d2;
	return diff;
}

inline __host__ __device__ float computePointWiseDistance(vec3f* pts1, vec3f* pts2, const int& num, float rot[9]){
	float diff = 0.0f;
	vec3f tmp;
	std::vector<float>r;
	for (int i=0; i<num; ++i){
		tmp = rotateVector(pts1[i], rot);
		diff += dist3d(tmp, pts2[i]);
	}
	return diff;
}

inline __host__ __device__ float computeProcrustesDistanceWithoutOrder(vec3f* pts1, vec3f* pts2, const int& num, const bool& to_normalize=true){
	
	vec3f* tmp1 = new vec3f[num];
	vec3f* tmp2 = new vec3f[num];
	for (int i = 0; i < num; ++i) {
		tmp1[i] = pts1[i];
		tmp2[i] = pts2[i];
	}
	vec3f tmp;
	float quat[4];
	float rot[9];

	if (to_normalize) normalizePointSet(pts1, pts2, num, tmp1, tmp2);

	float diff, prevDiff;
	
	for(int order=0; order<2; order++){
		if(order==1){
			prevDiff = diff;
			for (int i=0; i<(num/2); ++i){
				tmp = tmp2[i];
				tmp2[i] = tmp2[num-1-i];
				tmp2[num-1-i] = tmp;
			}
		}
		//compute rotation
		computeRotationQuat(tmp1, tmp2, num, quat);
		rotMatrixFromQuaternion(rot, quat);

		diff = computePointWiseDistance(tmp1, tmp2, num, rot);

		if(diff<0.01f) break;
		if(order==1 && diff>prevDiff) diff = prevDiff;
	}

	delete[] tmp1;
	delete[] tmp2;
	return diff;
}

inline __host__ __device__ float computeProcrustesDistanceWithOrder(vec3f* pts1, vec3f* pts2, const int& num, const bool& to_normalize=true){

	vec3f* tmp1 = new vec3f[num];
	vec3f* tmp2 = new vec3f[num];
	float quat[4];
	float rot[9];

	if (to_normalize) normalizePointSet(pts1, pts2, num, tmp1, tmp2);
	computeRotationQuat(tmp1, tmp2, num, quat);
	rotMatrixFromQuaternion(rot, quat);
	float diff = computePointWiseDistance(tmp1, tmp2, num, rot);

	delete[] tmp1;
	delete[] tmp2;
	return diff;
}

__host__ float* genProcrustesDistanceMatrixHost(Streamline* stls, const int& numStl, vec3f* pts, const int& numPoint, const int& winsize);

#endif //REGISTRATION_H