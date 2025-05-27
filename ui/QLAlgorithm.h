#ifndef _QL_ALGORITHM_H
#define _QL_ALGORITHM_H

#include <cuda.h>

inline __host__ __device__ void tridiagonal3 (float mat[3][3], float* diag, float* subd)
// input:   mat = 3x3 real, symmetric A
// output:  mat = orthogonal matrix Q
//          diag = diagonal entries of T, diag[0,1,2]
//          subd = subdiagonal entry of T, subd[0,1]
{
	float a = mat[0][0], b = mat[0][1], c = mat[0][2],
		d = mat[1][1], e = mat[1][2],
		f = mat[2][2];
	diag[0] = a;
	subd[2] = 0;
	if ( c != 0 ) {
		float ell = sqrt(b*b+c*c);
		b /= ell;
		c /= ell;
		float q = 2*b*e+c*(f-d);
		diag[1] = d+c*q;
		diag[2] = f-c*q;
		subd[0] = ell;
		subd[1] = e-b*q;
		mat[0][0] = 1; mat[0][1] = 0; mat[0][2] = 0;
		mat[1][0] = 0; mat[1][1] = b; mat[1][2] = c;
		mat[2][0] = 0; mat[2][1] = c; mat[2][2] = -b;
	} else {
		diag[1] = d;
		diag[2] = f;
		subd[0] = b;
		subd[1] = e;
		mat[0][0] = 1; mat[0][1] = 0; mat[0][2] = 0;
		mat[1][0] = 0; mat[1][1] = 1; mat[1][2] = 0;
		mat[2][0] = 0; mat[2][1] = 0; mat[2][2] = 1;
	}
}

inline __host__ __device__ bool QLAlgorithm3 (float* m_afDiag, float* m_afSubd,
											  float m_aafMat[3][3])
{
	// QL iteration with implicit shifting to reduce matrix from tridiagonal
	// to diagonal

	for (int i0 = 0; i0 < 3; i0++)
	{
		const int iMaxIter = 32;
		int iIter;
		for (iIter = 0; iIter < iMaxIter; iIter++)
		{
			int i1;
			for (i1 = i0; i1 <= 1; i1++)
			{
				float fSum = abs(m_afDiag[i1]) + abs(m_afDiag[i1+1]);
				if ( abs(m_afSubd[i1]) + fSum == fSum ) {
					break;
				}
			}
			if ( i1 == i0 ) {
				break;
			}

			float fTmp0 = (m_afDiag[i0+1]-m_afDiag[i0])/(2.0f*m_afSubd[i0]);
			float fTmp1 = sqrtf(fTmp0*fTmp0+1.0f);
			if ( fTmp0 < 0.0f ) {
				fTmp0 = m_afDiag[i1]-m_afDiag[i0]+m_afSubd[i0]/(fTmp0-fTmp1);
			}
			else {
				fTmp0 = m_afDiag[i1]-m_afDiag[i0]+m_afSubd[i0]/(fTmp0+fTmp1);
			}
			float fSin = 1.0f;
			float fCos = 1.0f;
			float fTmp2 = 0.0f;
			for (int i2 = i1-1; i2 >= i0; i2--)
			{
				float fTmp3 = fSin*m_afSubd[i2];
				float fTmp4 = fCos*m_afSubd[i2];
				if ( abs(fTmp3) >= abs(fTmp0) )
				{
					fCos = fTmp0/fTmp3;
					fTmp1 = sqrtf(fCos*fCos+1.0f);
					m_afSubd[i2+1] = fTmp3*fTmp1;
					fSin = 1.0f/fTmp1;
					fCos *= fSin;
				}
				else
				{
					fSin = fTmp3/fTmp0;
					fTmp1 = sqrtf(fSin*fSin+1.0f);
					m_afSubd[i2+1] = fTmp0*fTmp1;
					fCos = 1.0f/fTmp1;
					fSin *= fCos;
				}
				fTmp0 = m_afDiag[i2+1]-fTmp2;
				fTmp1 = (m_afDiag[i2]-fTmp0)*fSin+2.0f*fTmp4*fCos;
				fTmp2 = fSin*fTmp1;
				m_afDiag[i2+1] = fTmp0+fTmp2;
				fTmp0 = fCos*fTmp1-fTmp4;

				for (int i3 = 0; i3 < 3; i3++)
				{
					fTmp3 = m_aafMat[i3][i2+1];
					m_aafMat[i3][i2+1] = fSin*m_aafMat[i3][i2] +
						fCos*fTmp3;
					m_aafMat[i3][i2] = fCos*m_aafMat[i3][i2] -
						fSin*fTmp3;
				}
			}
			m_afDiag[i0] -= fTmp2;
			m_afSubd[i0] = fTmp0;
			m_afSubd[i1] = 0.0f;
		}

		if ( iIter == iMaxIter )
		{
			// should not get here under normal circumstances
			return false;
		}
	}

	return true;
}

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

#endif //_QL_ALGORITHM_H