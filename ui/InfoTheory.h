#ifndef INFO_THEORY_H
#define INFO_THEORY_H

#include "typeOperation.h"
#include "VolumeData.h"
#include <cmath>

inline __host__ __device__ int computeBin(const float& val, const float& minv, const float& intv, const int& numBin){
	int ret;
	if (val<=minv){
		ret = 0;
	} else {
		ret = ((val-minv)/intv);
		if (ret>=numBin){
			ret = numBin-1;
		}
	}
	return ret;
}

//assume all entries in histogram is initialized to zero
inline __host__ __device__ void computeArrayHist(float* vals, const int& numPoint,
												 const float& minv, const float& maxv, const int& numBin,
												 int* hist)
{
	float intv = (maxv-minv)/numBin;
	int bin;
	for (int i=0; i<numPoint; ++i) {
		bin = computeBin(vals[i], minv, intv, numBin);
		++hist[bin];
	}
}

static __host__ __device__ void computeBlockHist(float* volX, const float& volXMin, const float& volXIntv, int numBinX,
								 float* volY, const float& volYMin, const float& volYIntv, int numBinY,
								 int* histYX, int* histY, int* histX,
								 const int& volW, const int& volWH,
								 const int& pos_x, const int& pos_y, const int& pos_z,
								 const int& w, const int& h, const int& d)
{
	int i, j, k, binX, binY;

	int offset;
	for (i=0; i<d; ++i){
		for (j=0; j<h; ++j){
			for (k=0; k<w; ++k){
				offset = (pos_z+i)*volWH+(pos_y+j)*volW+pos_x+k;
				binX = computeBin(volX[offset], volXMin, volXIntv, numBinX);
				binY = computeBin(volY[offset], volYMin, volYIntv, numBinY);
				++histX[binX];
				++histY[binY];
				++histYX[binX*numBinY+binY];
			}
		}
	}
}

static __host__ __device__ void computeBlockHist(float* volX, const float& volXMin, const float& volXIntv, int numBinX,
												 int* histX,
												 const int& volW, const int& volWH,
												 const int& pos_x, const int& pos_y, const int& pos_z,
												 const int& w, const int& h, const int& d)
{
	int i, j, k, binX;
	int offset;
	for (i=0; i<d; ++i){
		for (j=0; j<h; ++j){
			for (k=0; k<w; ++k){
				offset = (pos_z+i)*volWH+(pos_y+j)*volW+pos_x+k;
				binX = computeBin(volX[offset], volXMin, volXIntv, numBinX);
				++histX[binX];
			}
		}
	}
}

//JSD(P|Q) = sum p(x)log(2p(x)/(p(x)+q(x)))+q(x)log(2q(x)/(p(x)+q(x)))
static __host__ __device__ float computeJSD(int* histX, int* histY, const int& numX, const int& numY, const int& numBin)
{
	float px, py;
	float xfac = 1.0f/numX;
	float yfac = 1.0f/numY;
	float d = 0.0f;

	for (int i=0; i<numBin; ++i){
		px = histX[i]*xfac;
		py = histY[i]*yfac;
		if (px>=0.000000001f){
			d += px*logf(2.0f*px/(px+py));
		} 
		if (py>=0.000000001f){
			d += py*logf(2.0f*py/(px+py));
		}
	}

	return (0.7213475204444f*d);//0.5f/ln2
}

//JSD(P|Q) = sum p(x)log(2p(x)/(p(x)+q(x)))+q(x)log(2q(x)/(p(x)+q(x)))
static __host__ __device__ float computeJSD(int* histX, int* histY, const int& numData, const int& numBin)
{
	float px, py, n=numData;
	float d = 0.0f;

	for (int i=0; i<numBin; ++i){
		px = histX[i]/n;
		py = histY[i]/n;
		if (px>=0.000000001f){
			d += px*logf(2.0f*px/(px+py));
		} 
		if (py>=0.000000001f){
			d += py*logf(2.0f*py/(px+py));
		}
	}

	return (0.7213475204444f*d);//0.5f/ln2
}

//JSD(P|Q) = sum p(x)log(2p(x)/(p(x)+q(x)))+q(x)log(2q(x)/(p(x)+q(x)))
static __host__ __device__ float computeJSD(int* histX, int* histY, const int& numBin)
{
	float px, py, Nx=0.0f, Ny=0.0f;
	float d = 0.0f;

	for (int i=0; i<numBin; ++i){
		Nx += histX[i];
		Ny += histY[i];
	}

	for (int i=0; i<numBin; ++i){
		px = histX[i]/Nx;
		py = histY[i]/Ny;
		if (px>=0.000000001f){
			d += px*logf(2.0f*px/(px+py));
		} 
		if (py>=0.000000001f){
			d += py*logf(2.0f*py/(px+py));
		}
	}

	return (0.7213475204444f*d);//0.5f/ln2
}

//I1 = sum_{y} p(y|x)log(p(y|x)/p(y))
//   = log N - log n(x) + (1/n(x))*sum_{y} n(y|x)*(log n(y|x) - log n(y))
//I2 = H(Y)-H(Y|x)
//   = - sum_{y} p(y) log p(y) + sum_{y} p(y|x) log p(y|x)
//   = - 1/N *sum_{y} n(y)*(log n(y) - log n) + (1/n(x))*sum_{y} n(y|x)*(log n(y|x) - log n(x))
static __host__ __device__ void computeBlockI1I2(int* histYX, int* histY, int* histX,
								 const int& numBinX, const int& numBinY, 
								 const int& numData,
								 float* logValue,
								 float* I1, float* I2)
{
	int x, y, yx, nx, ny, nyx;
	float invNx, i1, i2, hY, logN, logNx, logNyx;

	//log(N)
	logN = logValue[numData];

	//compute H(Y)
	hY = 0.0f;
	for (y=0; y<numBinY; ++y){
		ny = histY[y];
		if(ny!=0){
			hY -= ny*(logValue[ny]-logN);
		}
	}
	hY /= numData;

	//compute I1, I2 
	for (x=0, yx=0; x<numBinX; ++x){
		nx = histX[x];

		if(nx==0){
			I1[x] = 0.0f;
			I2[x] = hY;
			yx += numBinY;
		} else {
			invNx = 1.0f/nx;
			logNx = logValue[nx];

			i1 = 0.0f;
			i2 = 0.0f;

			for (y=0; y<numBinY; ++y, ++yx){
				nyx = histYX[yx];
				logNyx = logValue[nyx];
				ny = histY[y];

				if (nyx!=0){
					i1 += nyx*(logNyx-logValue[ny]);
					i2 += nyx*(logNyx-logNx);
				}
			}

			I1[x] = invNx*i1+logN-logNx;
			I2[x] = invNx*i2+hY;
		}
	}
}

void computeLogValue(float *logValue, const int& num);

void computeI1I2(VolumeData<float>* x, VolumeData<float>* y, const int& numBinX, const int& numBinY,
				 const int& blockW, const int& blockH, const int& blockD, float* I1, float* I2);

void cudaComputeI1I2(VolumeData<float>* x, VolumeData<float>* y, const int& numBinX, const int& numBinY,
					 const int& blockW, const int& blockH, const int& blockD, float* I1, float* I2);

void cudaComputeBlockJSDMatrix(VolumeData<float>* x, const int& numBin,
						  const int& blockW, const int& blockH, const int& blockD, float* retMat);

void cudaComputeGroupJSDMatrix(VolumeData<float>* data, VolumeData<int>* group, 
							   int* numEachGroup, const int& numGroup,
							   const int& numBin, float* retMat);

void computeGroupJSDMatrix(VolumeData<float>* data, VolumeData<int>* group, 
						   const int& numGroup, const int& numBin, float* retMat);

#endif //INFO_THEORY_H