#ifndef _MATRIX_DATA_H
#define _MATRIX_DATA_H

#include <cassert>
#include <cmath>
#include <fstream>
#include "typeOperation.h"
#ifdef USE_CUDA 
#include <cuda_runtime.h>
#include "helper_cuda.h"
#endif

template <typename T>
class MatrixData {
private:
	int w, h;
	int wh;
	T minv, maxv;
	T **rows;
	T *data;		// data pointer
	T *d_data;     // device data
	bool b_retain; //if release the original data

public:
	MatrixData();
	MatrixData(int vw, int vh);
	MatrixData(int vw, int vh, T value);
	MatrixData(int vw, int vh, T *data);
	MatrixData(int vw, int vh, const char *file);
	~MatrixData();

	void allocate(int vw, int vh);

	//convert matrix
	template<typename T2>
	MatrixData<T2>* convert();

	void setRetainData(const bool& retain) { b_retain = retain; }

	int rowSize() const { return w; }
	int MatrixSize() const { return wh; }
	int width() const { return w; }
	int height() const { return h; }
	vec2i dim() const { return makeVec2i(w, h); }

	bool writeFile(const char* file);
	bool readFile(const char* file);
	bool readHeader(const char* file);
	static bool writeFile(T* data, int num, const char* file);
	static bool readFile(T* data, int num, const char* file);
	static bool readHeader(int& w, int& h, const char* file);

	bool inMatrix(const int& x, const int& y);

	void updateMinMax();
	void normalizeData();
	void square_transpose();
	void transpose(MatrixData<T>& trans_mat);
	void reverse();

	void printMat();

#ifdef USE_CUDA
	T* createDeviceMemory(const bool& bKeepHost);
	void freeDeviceMemory();
	void freeHostMemory();
#endif

	//getter functions
	T getMin() { return minv; }
	T getMax() { return maxv; }
	int getNumBlock(const int& blockW, const int& blockH);
	T* getData() { return data; }
	int getSize() { return wh; }
	T** getMatrixPointer() { return rows; }
	T* getDevicePointer() { return d_data; }
	int posToIdx(const int& x, const int& y);
	void idxToPos(const int& idx, int& x, int& y);

	T* operator [] (const int& y) { return rows[y]; }
	T getValue(const float& x, const float& y, bool &bSuccess);
	const T& getValue(const int& x, const int& y, bool &bSuccess);
	void appendSubMatrix(std::vector<T>& ret, const IndexRange& x, const IndexRange& y);
	void submatrix(MatrixData<T>& matrix, const vec2f& lb, const vec2f& step);
	void submatrix(MatrixData<T>& matrix, const vec2i& lb, const vec2i& step);
	void submatrix(MatrixData<T>& matrix, const std::vector<float>& x_samples, const std::vector<float>& y_samples);
	void submatrix(MatrixData<T>& matrix, const std::vector<float>& samples);
	bool setValue(const int& x, const int& y, const T &value);
	const T& getValueQuick(const int& x, const int& y);
	void setValueQuick(const int& x, const int& y, const T &value);
	bool setRowValue(const int& y, const T& value);
	bool setColumnValue(const int& x, const T& value);

};

template <typename T>
MatrixData<T>::MatrixData() {
	data = NULL;
	rows = NULL;
	d_data = NULL;
	w = h = wh = 0;
	b_retain = false;
}

template<typename T>
inline void MatrixData<T>::allocate(int vw, int vh) {
	b_retain = false;
	w = vw;
	h = vh;
	wh = w*h;
	data = new T[wh];
	d_data = NULL;

	rows = new T*[h];
	for (int i = 0; i < h; ++i) {
		rows[i] = &data[i*w];
	}
}

template <typename T>
MatrixData<T>::MatrixData(int vw, int vh) {
	allocate(vw, vh);
}

template <typename T>
MatrixData<T>::MatrixData(int vw, int vh, T value) {
	allocate(vw, vh);

	for (int i = 0; i < wh; ++i) {
		data[i] = value;
	}
	maxv = minv = value;
}

template <typename T>
MatrixData<T>::MatrixData(int vw, int vh, T *vdata) {
	b_retain = true;
	w = vw;
	h = vh;
	wh = w*h;
	data = vdata;
	d_data = NULL;
	rows = new T*[h];
	for (int i = 0; i < h; ++i) {
		rows[i] = &data[i*w];
	}

	//updateMinMax();
}

template <typename T>
MatrixData<T>::MatrixData(int vw, int vh, const char *file) {
	std::ifstream ifile;
	ifile.open(file, std::ios_base::in | std::ios::binary);
	if (!ifile.is_open()) {
		printf("Err: fail to open matrix file: %s\n", file);
		return;
	}

	allocate(vw, vh);
	ifile.read((char*)data, sizeof(T)*wh);
	ifile.close();
}

template<typename T>
template<typename T2>
MatrixData<T2>* MatrixData<T>::convert() {
	MatrixData<T2>* ret = new MatrixData<T2>(w, h);
	T2* ret_data = ret->getData();
	for (int i = 0; i < wh; ++i) {
		ret_data[i] = data[i];
	}

	return ret;
}

template <typename T>
bool MatrixData<T>::writeFile(T* data, int num, const char* file) {
	std::ofstream ofile;
	ofile.open(file, std::ios_base::out | std::ios::binary);
	if (!ofile.is_open()) {
		printf("Err: fail to open volume file: %s\n", file);
		return false;
	}

	ofile.write((char*)data, sizeof(T)*num);
	ofile.close();

	return true;
}

template <typename T>
bool MatrixData<T>::writeFile(const char* file) {
	if (!data) {
		return false;
	}
	return writeFile(data, wh, file);
}

template <typename T>
bool MatrixData<T>::readFile(T* data, int num, const char* file) {
	std::ifstream ifile;
	ifile.open(file, std::ios_base::in | std::ios::binary);
	if (!ifile.is_open()) {
		printf("Err: fail to open volume file: %s\n", file);
		return false;
	}

	ifile.read((char*)data, sizeof(T)*num);
	ifile.close();

	return true;
}

template <typename T>
bool MatrixData<T>::readFile(const char* file) {
	if (wh == 0) {
		return false;
	}

	if (!data) {
		data = new T[wh];
	}
	return readFile(data, wh, file);
}

template <typename T>
bool MatrixData<T>::readHeader(int& w, int& h, const char* file) {
	std::ifstream ifile;
	ifile.open(file, std::ios_base::in);
	if (!ifile.is_open()) {
		printf("Err: fail to open volume file: %s\n", file);
		return false;
	}

	ifile >> w >> h;

	ifile.close();

	return true;
}

template <typename T>
bool MatrixData<T>::readHeader(const char* file) {
	return  readHeader(w, h, file);
}

template <typename T>
void MatrixData<T>::printMat() {
	T** td = rows;
	for (int i = 0; i < h; ++i) {
		for (int j = 0; j < w; ++j) {
			printf("%.3f ", td[i][j]);
		}
		printf("\n");
	}
		
}

template <typename T>
MatrixData<T>::~MatrixData() {
	if (!b_retain && data)
		delete[] data;

	if (rows)
		delete[] rows;

#ifdef USE_CUDA
	if (d_data) {
		checkCudaErrors(cudaFree(d_data));
		d_data = NULL;
	}
#endif
}

template <typename T>
bool MatrixData<T>::inMatrix(const int& x, const int& y) {
	return (x >= 0 && x < w && y >= 0 && y < h);
}

template <typename T>
int MatrixData<T>::getNumBlock(const int& blockW, const int& blockH) {
	return iDivUp(w, blockW)*iDivUp(h, blockH);
}

template <typename T>
int MatrixData<T>::posToIdx(const int& x, const int& y) {
	return (y*w + x);
}

template <typename T>
void MatrixData<T>::idxToPos(const int& idx, int& x, int& y) {
	x = idx%w;
	y = idx / w;
}

template <typename T>
void MatrixData<T>::updateMinMax() {
	computeMinMax(data, wh, minv, maxv);
}


template <typename T>
void MatrixData<T>::square_transpose() {
	if (w != h) {
		printf("Not a square matrix. Call transpose instead. \n");
		return;
	}
	for (int i = 0; i < h; ++i) {
		for (int j = i + 1; j < w; ++j) {
			std::swap(rows[i][j], rows[j][i]);
		}
	}
}

template <typename T>
void MatrixData<T>::transpose(MatrixData<T>& trans_mat) {
	T** tmat = trans_mat.getMatrixPointer();
	for (int i = 0; i < h; ++i) {
		for (int j = 0; j < w; ++j) {
			tmat[j][i] = rows[i][j];
		}
	}
}

template <typename T>
void MatrixData<T>::reverse() {
	T** rmat;
	for (int i = 0; i < h; ++i) {
		for (int j = 0; j < w; ++j) {
			rmat[w-j-1][h-i-1] = rows[i][j];
		}
	}
	rows = rmat;
	for (int i = 0; i < wh; ++i) {
		data[i] = rows[i/w][i%w];
	}
}

template <typename T>
void MatrixData<T>::normalizeData() {
	for (int i = 0; i < wh; ++i) {
		data[i] = (data[i] - minv) / (maxv - minv);
	}
	maxv = 1.0f;
	minv = 0.0f;
}

#ifdef USE_CUDA
template <typename T>
T* MatrixData<T>::createDeviceMemory(const bool& bKeepHost) {
	if (d_data) {
		checkCudaErrors(cudaFree(d_data));
		d_data = NULL;
	}

	if (!data) {
		printf("Err: host memory does not exist.\n");
		return NULL;
	}

	checkCudaErrors(cudaMalloc((void**)&d_data, sizeof(T)*wh));
	checkCudaErrors(cudaMemcpy(d_data, data, sizeof(T)*wh, cudaMemcpyHostToDevice));

	if (!bKeepHost) {
		freeHostMemory();
	}

	return d_data;
}

template <typename T>
void MatrixData<T>::freeHostMemory() {
	if (data)
		delete[] data;
	if (rows)
		delete[] rows;

	data = NULL;
	rows = NULL;
}

template <typename T>
void MatrixData<T>::freeDeviceMemory() {
	if (d_data)
		checkCudaErrors(cudaFree(d_data));

	d_data = NULL;
}
#endif 

//include x/y.lower does not include x/y.upper
template <typename T>
void MatrixData<T>::appendSubMatrix(std::vector<T>& ret, const IndexRange& x, const IndexRange& y) {
	int sw = x.upper - x.lower, sh = y.upper - y.lower;
	for (int i = y.lower; i < y.upper; ++i) {
		ret.insert(ret.end(), &rows[i][x.lower], &rows[i][x.upper]);
	}
}

template <typename T>
void MatrixData<T>::submatrix(MatrixData<T>& matrix, const vec2f& lb, const vec2f& step) {
	int sw = matrix.width(), sh = matrix.height();
	for (int i = 0; i < sh; ++i) {
		for (int j = 0; j < sw; ++j) {
			matrix[i][j] = getValueQuick(lb.x + i*step.x, lb.y + j*step.y);
		}
	}
}


template <typename T>
void MatrixData<T>::submatrix(MatrixData<T>& matrix, const vec2i& lb, const vec2i& step) {
	int sw = matrix.width(), sh = matrix.height();
	for (int i = 0; i < sh; ++i) {
		for (int j = 0; j < sw; ++j) {
			matrix[i][j] = getValueQuick(lb.x + i*step.x, lb.y + j*step.y);
		}
	}
}

template <typename T>
void MatrixData<T>::submatrix(MatrixData<T>& matrix,
	const std::vector<float>& x_samples, const std::vector<float>& y_samples)
{
	int sw = matrix.width(), sh = matrix.height();
	if (sw != x_samples.size() || sh != y_samples.size()) {
		printf("Error: the matrix size [%d,% d] and sample size [%d, %d] does not match.\n",
			sw, sh, x_samples.size(), y_samples.size());
	}
	for (int i = 0; i < sh; ++i) {
		for (int j = 0; j < sw; ++j) {
			matrix[i][j] = getValueQuick(x_samples[j], y_samples[i]);
		}
	}
}

template <typename T>
void MatrixData<T>::submatrix(MatrixData<T>& matrix, const std::vector<float>& samples)
{
	int sw = matrix.width(), sh = matrix.height();
	if (sw != samples.size() || sh != samples.size()) {
		printf("Error: the matrix size [%d,% d] and sample size [%d, %d] does not match.\n",
			sw, sh, samples.size(), samples.size());
	}
	for (int i = 0; i < sh; ++i) {
		for (int j = 0; j < sw; ++j) {
			matrix[i][j] = getValueQuick(samples[j], samples[i]);
		}
	}
}

template <typename T>
T MatrixData<T>::getValue(const float& x, const float& y, bool &bSuccess) {
	if (x <= 0.0000001f || x >= (w - 1.0000001f) || y <= 0.0000001f || y >= (h - 1.0000001f)) {
		bSuccess = false;
		vec2i c = clamp(makeVec2i(x, y), makeVec2i(0, 0), makeVec2i(w - 1, h - 1));
		return rows[c.y][c.x];
	}

	int ix = floorf(x);
	int iy = floorf(y);

	float facx = x - ix, facy = y - iy;

	T ret = (1.0f - facx)*(1.0f - facy)*rows[iy][ix]
		+ facx*(1.0f - facy)*rows[iy][ix + 1]\
		+ (1.0f - facx)*facy*rows[iy + 1][ix]\
		+ facx*facy*rows[iy + 1][ix + 1];

	bSuccess = true;
	return ret;
}

template <typename T>
const T& MatrixData<T>::getValue(const int& x, const int& y, bool &bSuccess) {
	if (!inMatrix(x, y)) {
		bSuccess = false;
		return 0;
	}

	T ret = rows[y][x];
	bSuccess = true;

	return ret;
}

template <typename T>
bool MatrixData<T>::setValue(const int& x, const int& y, const T &value) {
	if (!inMatrix(x, y)) {
		return false;
	}

	rows[y][x] = value;

	return true;
}

template <typename T>
const T& MatrixData<T>::getValueQuick(const int& x, const int& y) {
	return (rows[y][x]);
}

template <typename T>
void MatrixData<T>::setValueQuick(const int& x, const int& y, const T &value) {
	rows[y][x] = value;
}

template <typename T>
bool MatrixData<T>::setColumnValue(const int& x, const T& value) {
	if (x<0 || x>w) return false;
	for (int i = 0; i < h; ++i) {
		rows[i][x] = value;
	}
	return true;
}

template <typename T>
bool MatrixData<T>::setRowValue(const int& y, const T& value) {
	if (y<0 || y>h) return false;
	for (int i = 0; i < w; ++i) {
		rows[y][i] = value;
	}
	return true;
}

#endif