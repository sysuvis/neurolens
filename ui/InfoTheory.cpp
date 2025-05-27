#include "InfoTheory.h"
#include <cstring>

void computeLogValue(float *logValue, const int& num){
	for (int i=1; i<=num; ++i){
		logValue[i] = log((float)i);
	}
}

void computeI1I2(VolumeData<float>* x, VolumeData<float>* y, int numBinX, int numBinY,
				 int blockW, int blockH, int blockD, float* I1, float* I2)
{
	float volXMin, volYMin, volXMax, volYMax, volXIntv, volYIntv;
	volXMin = x->getMin();
	volXMax = x->getMax();
	volXIntv = (volXMax-volXMin)/numBinX;
	volYMin = y->getMin();
	volYMax = y->getMax();
	volYIntv = (volYMax-volYMin)/numBinY;

	int volW, volH, volD, volWH;

	if (x->width()!=y->width()||x->height()!=y->height()||x->depth()!=y->depth()){
		printf("Err: cudaComputeI1I2: Volume dimensions do not match.\n\n");
		return;
	} else {
		volW = x->width();
		volH = x->height();
		volD = x->depth();
		volWH = volW*volH;
	}

	int numBlock = x->getNumBlock(blockW, blockH, blockD);

	float *logValue = new float[blockW*blockH*blockD+1];
	int *histX = new int[numBinX];
	int *histY = new int[numBinY];
	int *histYX = new int[numBinX*numBinY];

	computeLogValue(logValue, blockW*blockH*blockD);

	float *volX = x->getData();
	float *volY = y->getData();
	if (!volX || !volY){
		printf("Err: ComputeI1I2: volumes do not exist.\n\n");
	}
	
	int w = iDivUp(volW, blockW);//# block per row
	int h = iDivUp(volH, blockH);//# block per column

	for (int idx=0; idx<numBlock; ++idx){
		int x, y, z;
		x = idx%w;
		y = (idx/w)%h;
		z = idx/(w*h);

		x = ((x+1)*blockW>volW)?(volW-blockW):(x*blockW);
		y = ((y+1)*blockH>volH)?(volH-blockH):(y*blockH);
		z = ((z+1)*blockD>volD)?(volD-blockD):(z*blockD);

		memset(histX, 0, sizeof(int)*numBinX);
		memset(histY, 0, sizeof(int)*numBinY);
		memset(histYX, 0, sizeof(int)*numBinX*numBinY);

		float* i1 = &I1[idx*numBinX];
		float* i2 = &I2[idx*numBinY];

		computeBlockHist(volX, volXMin, volXIntv, numBinX, volY, volYMin, volYIntv, numBinY,
			histYX, histY, histX, volW, volWH,
			x, y, z, blockW, blockH, blockD);

		computeBlockI1I2(histYX, histY, histX, numBinX, numBinY, blockW*blockH*blockD, logValue, i1, i2);
	}

	delete[] logValue;
	delete[] histX;
	delete[] histY;
	delete[] histYX;
}

void computeGroupJSDMatrix(VolumeData<float>* data, VolumeData<int>* group, 
						   int numGroup, int numBin, float* retMat)
{
	float dataMin, dataMax, dataIntv;
	dataMin = data->getMin();
	dataMax = data->getMax();
	dataIntv = (dataMax-dataMin)/numBin;

	int* hist = new int[numGroup*numBin];
	memset(hist, 0, sizeof(int)*numGroup*numBin);
	int bin;
	float* dataf = data->getData();
	int* group_data = group->getData();
	for (int i=0; i<data->volumeSize(); ++i) {
		bin = computeBin(dataf[i], dataMin, dataIntv, numBin);
		++hist[group_data[i]*numBin+bin];
	}

	float** mat = new float*[numGroup];
	for (int i=0; i<numGroup; ++i) mat[i] = &retMat[i*numGroup];
	for (int i=0; i<numGroup; ++i) {
		mat[i][i] = 0.0f;
		for (int j=i+1; j<numGroup; ++j) {
			mat[i][j] = mat[j][i] = computeJSD(&hist[i*numBin], &hist[j*numBin], numBin);
		}
	}

	delete[] hist;
	delete[] mat;
}