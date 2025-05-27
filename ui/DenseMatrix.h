#pragma once

#include <map>

template<typename T>
class DenseMatrix {
public:
	typedef T* Row;
	typedef T* row_iterator;

	DenseMatrix() {}
	DenseMatrix(const int& n) { mMatrix.resize(n); }

	int numRow() { return mMatrix.size(); }
	void resize(const int& n) { mMatrix.resize(n); }
	Row& getRow(const int& r) { return mMatrix[r]; }
	Row& operator [] (const int& r) { return mMatrix[r]; }
	row_iterator getRowBegin(const int& r) { return mMatrix[r]; }
	row_iterator getRowEnd(const int& r) { return &(mMatrix[r][num_column - 1]); }

	void setDefaultValue(const T& default_value) { mDefaultValue = default_value; }
	void setData(const T* data, const int& num_columns);
	void allocateData(const int& num_columns);
	void setValue(const int& i, const int& j, const T& v);
	T getValue(const int& i, const int& j); 
	T getValue(const int& i, const int& j, bool& exist);
	bool isElementExist(const int& i, const int& j);
	void clear();

private:
	T mDefaultValue;
	int mNumColumn;
	bool mSelfAlloc;
	std::vector<T*> mMatrix;
};

template<typename T>
void DenseMatrix<T>::setData(const T* data, const int& num_columns){
	for (int i = 0; i < mMatrix.size(); ++i) {
		mMatrix[i] = &data[i*num_columns];
	}
	mSelfAlloc = false;
}

template<typename T>
void DenseMatrix<T>::allocateData(const int& num_columns){
	mMatrix[0] = new T[mMatrix.size()*num_columns];
	setData(mMatrix[0], num_columns);
	mSelfAlloc = true;
}

template<typename T>
void DenseMatrix<T>::setValue(const int& i, const int& j, const T& v) {
	mMatrix[i][j] = v;
}

template<typename T>
T DenseMatrix<T>::getValue(const int& i, const int& j) {
	return mMatrix[i][j];
}


template<typename T>
T DenseMatrix<T>::getValue(const int& i, const int& j, bool& exist) {
	if (isElementExist(i, j)) {
		exist = true;
		return mMatrix[i][j];
	}
	
	exist = false;
	return mDefaultValue;
}

template<typename T>
bool DenseMatrix<T>::isElementExist(const int& i, const int& j) {
	return (i < mMatrix.size() && i >= 0 && j < mNumColumn && j >= 0);
}

template<typename T>
void DenseMatrix<T>::clear() {
	mMatrix.clear();
}