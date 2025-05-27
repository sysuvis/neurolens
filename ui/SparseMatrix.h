#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <map>

template<typename T>
class SparseMatrix{
public:
	typedef std::map<int,T> Row;
	typedef typename std::map<int, T>::iterator row_iterator;

	SparseMatrix(){}
	SparseMatrix(const int& n){mMatrix.resize(n);}

	int numRow(){return mMatrix.size();}
	void resize(const int& n){mMatrix.resize(n);}
	Row& getRow(const int& r){return mMatrix[r];}
	Row& operator [] (const int& r){return mMatrix[r];}
	row_iterator getRowBegin(const int& r) { return mMatrix[r].begin(); }
	row_iterator getRowEnd(const int& r) { return mMatrix[r].end(); }

	void setDefaultValue(const T& default_value) { mDefaultValue = default_value; }
	void setValue(const int& i, const int& j, const T& v);
	T getValue(const int& i, const int& j);
	T getValue(const int& i, const int& j, bool& exist);
	bool isElementExist(const int& i, const int& j);
	void eraseElement(const int& i, const int& j);
	bool isRowEmpty(const int r);
	void eraseRow(const int& r);
	void clear();

private:
	T mDefaultValue;
	std::vector<std::map<int,T>> mMatrix;
};

template<typename T>
void SparseMatrix<T>::setValue(const int& i, const int& j, const T& v){
	mMatrix[i][j] = v;
}

template<typename T>
T SparseMatrix<T>::getValue( const int& i, const int& j ){
	typename std::map<int,T>& row = mMatrix[i];
	typename std::map<int,T>::iterator it = row.find(j);

	if (it!=row.end()) {
		return it->second;
	}

	return mDefaultValue;
}


template<typename T>
T SparseMatrix<T>::getValue( const int& i, const int& j, bool& exist ){
	typename std::map<int,T>& row = mMatrix[i];
	typename std::map<int,T>::iterator it = row.find(j);

	if (it!=row.end()) {
		exist = true;
		return it->second;
	}

	exist = false;
	return mDefaultValue;
}

template<typename T>
bool SparseMatrix<T>::isElementExist( const int& i, const int& j ){
	typename std::map<int,T>& row = mMatrix[i];
	typename std::map<int,T>::iterator it = row.find(j);
	if (it!=row.end()) {
		return true;
	}
	return false;
}

template<typename T>
void SparseMatrix<T>::eraseElement( const int& i, const int& j ){
	mMatrix[i].erase(j);
}

template<typename T>
bool SparseMatrix<T>::isRowEmpty( const int r ){
	return mMatrix[r].empty();
}

template<typename T>
void SparseMatrix<T>::eraseRow( const int& r ){
	mMatrix.erase(r);
	for (int i=0; i<mMatrix.size(); ++i) {
		typename std::map<int,T>& row = mMatrix[i];
		for (typename std::map<int,T>::iterator it=row.begin(); it!=row.end(); ++it) {
			if (it->first==r) {
				row.erase(it);
			} else if (it->first>r){
				--it->first;
			}
		}
	}
}

template<typename T>
void SparseMatrix<T>::clear(){
	mMatrix.clear();
}

#endif //SPARSE_MATRIX_H