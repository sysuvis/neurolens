#ifndef VOLUME_DATA_H
#define VOLUME_DATA_H

#include <cassert>
#include <cmath>
#include <fstream>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "typeOperation.h"

template <typename T>
class VolumeData{
private:
	int w, h, d;
	int wh, whd;
	T minv, maxv;
	T ***slices;
	T **rows;
	T *data;
	T *d_data;

public:
	VolumeData(int vw, int vh, int vd);
	VolumeData(int vw, int vh, int vd, T value);
	VolumeData(int vw, int vh, int vd, T *data);
	VolumeData(int vw, int vh, int vd, std::string file_path);
	~VolumeData();

	//convert volume
	template<typename T2>
	VolumeData<T2>* convert();

	int rowSize(){return w;}
	int sliceSize(){return wh;}
	int volumeSize(){return whd;}
	int width(){return w;}
	int height(){return h;}
	int depth(){return d;}

	static bool writeFile(T* data, int num, std::string file_path);
	static bool readFile(T* data, int num, std::string file_path);
	bool readFile(std::string file_path);
	bool saveToFile(std::string file_path);

	bool inVolume(const int& x, const int& y, const int& z);
	
	void updateMinMax();
	void normalizeData();

	//for cuda
	T* createDeviceMemory(const bool& bKeepHost);
	void freeDeviceMemory();
	void freeHostMemory();

	//getter functions
	T getMin(){return minv;}
	T getMax(){return maxv;}
	int getNumBlock(const int& blockW, const int& blockH, const int& blockD);
	T*** getVolumeData(){return slices;}
	T* getData(){return data;}
	T* getDevicePointer(){return d_data;}
	int posToIdx(const int& x, const int& y, const int& z);
	void idxToPos(const int& idx, int& x, int& y, int& z);

	T** operator [] (const int& z){return slices[z];}
	T getValue(const float& x, const float& y, const float& z, bool &bSuccess);
	T getValue(const int& x, const int& y, const int& z, bool &bSuccess);
	bool setValue(const int& x, const int& y, const int& z, const T &value);
	T getValueQuick(const float& x, const float& y, const float& z);
	T getValueQuick(const int& x, const int& y, const int& z);
	void setValueQuick(const int& x, const int& y, const int& z, const T &value);
};

template <typename T>
VolumeData<T>::VolumeData(int vw, int vh, int vd){
	w = vw;
	h = vh;
	d = vd;
	wh = w*h;
	whd = wh*d;
	data = new T[whd];
	d_data = NULL;

	slices = new T** [d];
	rows = new T* [d*h];
	for (int i=0; i<d; ++i){
		slices[i] = &rows[i*h];
		for (int j=0; j<h; ++j){
			rows[i*h+j] = &data[i*wh+j*w];
		}
	}
}

template <typename T>
VolumeData<T>::VolumeData(int vw, int vh, int vd, T value){
	w = vw;
	h = vh;
	d = vd;
	wh = w*h;
	whd = wh*d;
	data = new T[whd];
	d_data = NULL;

	slices = new T** [d];
	rows = new T* [d*h];
	for (int i=0; i<d; ++i){
		slices[i] = &rows[i*h];
		for (int j=0; j<h; ++j){
			rows[i*h+j] = &data[i*wh+j*w];
		}
	}

	for (int i=0; i<whd; ++i){
		data[i] = value;
	}
	maxv = minv = value;
}

template <typename T>
VolumeData<T>::VolumeData(int vw, int vh, int vd, T *vdata){
	w = vw;
	h = vh;
	d = vd;
	wh = w*h;
	whd = wh*d;
	data = vdata;
	d_data = NULL;

	slices = new T** [d];
	rows = new T* [d*h];
	for (int i=0; i<d; ++i){
		slices[i] = &rows[i*h];
		for (int j=0; j<h; ++j){
			rows[i*h+j] = &data[i*wh+j*w];
		}
	}

	updateMinMax();
}

template <typename T>
VolumeData<T>::VolumeData(int vw, int vh, int vd, std::string file){
	std::ifstream ifile;
	ifile.open(file.c_str(), std::ios_base::in|std::ios::binary);
	if (!ifile.is_open()){
		printf("Err: fail to open volume file: %s\n", file.c_str());
		return;
	}

	w = vw;
	h = vh;
	d = vd;
	wh = w*h;
	whd = wh*d;

	d_data = NULL;
	data = new T[whd];
	ifile.read((char*)data, sizeof(T)*whd);
	ifile.close();

	slices = new T** [d];
	rows = new T* [d*h];
	for (int i=0; i<d; ++i){
		slices[i] = &rows[i*h];
		for (int j=0; j<h; ++j){
			rows[i*h+j] = &data[i*wh+j*w];
		}
	}

	updateMinMax();
}

template <typename T>
bool VolumeData<T>::readFile(std::string file_path){
	std::ifstream ifile;
	ifile.open(file_path.c_str(), std::ios_base::in|std::ios::binary);
	if (!ifile.is_open()){
		printf("Err: fail to open volume file: %s\n", file_path.c_str());
		return false;
	}

	ifile.read((char*)data, sizeof(T)*whd);
	ifile.close();

	return true;
}


template <typename T>
bool VolumeData<T>::saveToFile(std::string file_path){
	std::ofstream ofile;
	ofile.open(file_path.c_str(), std::ios_base::out|std::ios::binary);
	if (!ofile.is_open()){
		printf("Err: fail to open volume file: %s\n", file_path.c_str());
		return false;
	}

	ofile.write((char*)data, sizeof(T)*whd);
	ofile.close();

	return true;
}

template<typename T>
template<typename T2>
VolumeData<T2>* VolumeData<T>::convert(){
	VolumeData<T2>* ret = new VolumeData<T2>(w, h, d);
	for (int i=0; i<whd; ++i){
		ret->getData()[i] = data[i];
	}
	ret->updateMinMax();

	return ret;
}

template <typename T>
bool VolumeData<T>::writeFile(T* data, int num, std::string file_path){
	std::ofstream ofile;
	ofile.open(file_path.c_str(), std::ios_base::out|std::ios::binary);
	if (!ofile.is_open()){
		printf("Err: fail to open volume file: %s\n", file_path.c_str());
		return false;
	}

	ofile.write((char*)data, sizeof(T)*num);
	ofile.close();

	return true;
}

template <typename T>
bool VolumeData<T>::readFile(T* data, int num, std::string file_path){
	std::ifstream ifile;
	ifile.open(file_path.c_str(), std::ios_base::in|std::ios::binary);
	if (!ifile.is_open()){
		printf("Err: fail to open volume file: %s\n", file_path.c_str());
		return false;
	}

	ifile.read((char*)data, sizeof(T)*num);
	ifile.close();

	return true;
}

template <typename T>
VolumeData<T>::~VolumeData(){
	if(slices)
		delete[] slices;
	if(rows)
		delete[] rows;

	if (d_data){
		checkCudaErrors(cudaFree(d_data));
		d_data = NULL;
	}
}

template <typename T>
bool VolumeData<T>::inVolume(const int& x, const int& y, const int& z){
	return (x>=0 && x<w && y>=0 && y<h && z>=0 && z<d);
}

template <typename T>
int VolumeData<T>::getNumBlock(const int& blockW, const int& blockH, const int& blockD){
	return iDivUp(w, blockW)*iDivUp(h, blockH)*iDivUp(d, blockD);
}

template <typename T>
int VolumeData<T>::posToIdx(const int& x, const int& y, const int& z){
	return (z*wh+y*w+x);
}

template <typename T>
void VolumeData<T>::idxToPos(const int& idx, int& x, int& y, int& z){
	x = idx%w;
	y = (idx/w)%h;
	z = idx/wh;
}

template <typename T>
void VolumeData<T>::updateMinMax(){
	computeMinMax(data, whd, minv, maxv);
}

template <typename T>
void VolumeData<T>::normalizeData(){
	for (int i=0; i<whd; ++i){
		data[i] = (data[i]-minv)/(maxv-minv);
	}
	maxv = 1.0f;
	minv = 0.0f;
}

template <typename T>
T* VolumeData<T>::createDeviceMemory(const bool& bKeepHost){
	if (d_data){
		checkCudaErrors(cudaFree(d_data));
		d_data = NULL;
	}

	if (!data){
		printf("Err: host memory does not exist.\n");
		return NULL;
	}

	checkCudaErrors(cudaMalloc((void**)&d_data, sizeof(T)*whd));
	checkCudaErrors(cudaMemcpy(d_data, data, sizeof(T)*whd, cudaMemcpyHostToDevice));

	if (!bKeepHost){
		freeHostMemory();
	}

	return d_data;
}

template <typename T>
void VolumeData<T>::freeHostMemory(){
	if(data)
		delete[] data;
	if(slices)
		delete[] slices;
	if(rows)
		delete[] rows;

	data = NULL;
	slices = NULL;
	rows = NULL;
}

template <typename T>
void VolumeData<T>::freeDeviceMemory(){
	if(d_data)
		checkCudaErrors(cudaFree(d_data));

	d_data = NULL;
}

template <typename T>
T VolumeData<T>::getValue(const float& x, const float& y, const float& z, bool &bSuccess){
	if (x<=0.0000001f || x>=(w-1.0000001f) || y<=0.0000001f || y>=(h-1.0000001f) || z<=0.0000001f || z>=(d-1.0000001f)) {
		bSuccess = false;
		return -1e30;
	}

	int ix = floorf(x);
	int iy = floorf(y);
	int iz = floorf(z);

	float facx=x-ix, facy=y-iy, facz=z-iz;

	T ret = (1.0f-facx)*(1.0f-facy)*(1.0f-facz)*slices[iz][iy][ix]
		+facx*(1.0f-facy)*(1.0f-facz)*slices[iz][iy][ix+1]\
		+(1.0f-facx)*facy*(1.0f-facz)*slices[iz][iy+1][ix]\
		+facx*facy*(1.0f-facz)*slices[iz][iy+1][ix+1]\
		+(1.0f-facx)*(1.0f-facy)*facz*slices[iz+1][iy][ix]\
		+facx*(1.0f-facy)*facz*slices[iz+1][iy][ix+1]\
		+(1.0f-facx)*facy*facz*slices[iz+1][iy+1][ix]\
		+facx*facy*facz*slices[iz+1][iy+1][ix+1];

	bSuccess = true;
	return ret;
}

template <typename T>
T VolumeData<T>::getValue(const int& x, const int& y, const int& z, bool &bSuccess){
	if (!inVolume(x, y, z)) {
		bSuccess = false;
		return -1e30;
	}

	T ret = slices[z][y][x];
	bSuccess = true;

	return ret;
}

template <typename T>
bool VolumeData<T>::setValue(const int& x, const int& y, const int& z, const T &value){
	if (!inVolume(x, y, z)) {
		return false;
	}

	slices[z][y][x] = value;

	return true;
}

template <typename T>
T VolumeData<T>::getValueQuick(const float& x, const float& y, const float& z) {
	int ix = floorf(x);
	int iy = floorf(y);
	int iz = floorf(z);

	float facx = x - ix, facy = y - iy, facz = z - iz;

	T ret = (1.0f - facx)*(1.0f - facy)*(1.0f - facz)*slices[iz][iy][ix]
		+ facx*(1.0f - facy)*(1.0f - facz)*slices[iz][iy][ix + 1]\
		+ (1.0f - facx)*facy*(1.0f - facz)*slices[iz][iy + 1][ix]\
		+ facx*facy*(1.0f - facz)*slices[iz][iy + 1][ix + 1]\
		+ (1.0f - facx)*(1.0f - facy)*facz*slices[iz + 1][iy][ix]\
		+ facx*(1.0f - facy)*facz*slices[iz + 1][iy][ix + 1]\
		+ (1.0f - facx)*facy*facz*slices[iz + 1][iy + 1][ix]\
		+ facx*facy*facz*slices[iz + 1][iy + 1][ix + 1];

	return ret;
}

template <typename T>
T VolumeData<T>::getValueQuick(const int& x, const int& y, const int& z){
	return (slices[z][y][x]);
}

template <typename T>
void VolumeData<T>::setValueQuick(const int& x, const int& y, const int& z, const T &value){
	slices[z][y][x] = value;
}
#endif