#include "PersistenceReduction.h"

PersistenceReduction::PersistenceReduction(const int& num){
	mBoundaryMatrix.resize(num);
	mVMatrix.resize(num);
	mBuffer.resize(num);
}

PersistenceReduction::~PersistenceReduction(){

}

void PersistenceReduction::clear(){
	mBoundaryMatrix.clear();
	mVMatrix.clear();
	mLowestColLookUp.clear();
}

void PersistenceReduction::resize(const int& num){
	mBoundaryMatrix.resize(num);
	mVMatrix.resize(num);
	mBuffer.resize(num);
}

void PersistenceReduction::setCol(const int& colIdx, std::vector<int>& col){
	mBoundaryMatrix[colIdx] = col;
	mVMatrix[colIdx].push_back(colIdx);
}

void PersistenceReduction::getCol(const int& colIdx, std::vector<int>& col){
	col = mBoundaryMatrix[colIdx];
}

void PersistenceReduction::addCol(const int& i, const int& j){
	std::vector<int>& col_i = mBoundaryMatrix[i];
	std::vector<int>& col_j = mBoundaryMatrix[j];

	int pi=0, pj=0, cnt=0, k;
	while (pi<col_i.size() && pj<col_j.size()){
		if (col_i[pi]<col_j[pj]){
			mBuffer[cnt] = col_i[pi]; 
			++cnt; ++pi;
		} else if (col_i[pi]>col_j[pj]){
			mBuffer[cnt] = col_j[pj];
			++cnt; ++pj;
		} else {
			++pi; ++pj;
		}
	}
	
	col_j.resize(cnt);
	std::copy(mBuffer.begin(), mBuffer.begin()+cnt, col_j.begin());

	//update V matrix and lowest look up
	mVMatrix[j].insert(mVMatrix[j].end(), mVMatrix[i].begin(), mVMatrix[i].end());
}

int PersistenceReduction::getLowest(const int& colIdx){
	if (mBoundaryMatrix[colIdx].empty()){
		return -1;
	}
	return mBoundaryMatrix[colIdx].back();
}

void PersistenceReduction::reduce(const int& start){
	mLowestColLookUp.resize(start);
	mLowestColLookUp.resize(mBoundaryMatrix.size(), -1);
	int lowest;
	for (int i=start; i<mBoundaryMatrix.size(); ++i){
		lowest = getLowest(i);
		while (lowest!=-1 && mLowestColLookUp[lowest]!=-1){
			addCol(mLowestColLookUp[lowest], i);
			lowest = getLowest(i);
		}
		if (lowest!=-1){
			mLowestColLookUp[lowest]=i;
		}
	}
}

void PersistenceReduction::getVCol(const int& colIdx, std::vector<int>& col){
	col = mVMatrix[colIdx];
}