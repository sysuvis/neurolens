#ifndef PERSISTENCE_REDUCTION
#define PERSISTENCE_REDUCTION

#include <vector>

class PersistenceReduction{
public:
	PersistenceReduction(const int& num);
	~PersistenceReduction();

	void clear();
	void resize(const int& num);
	void reduce(const int& start=0);
	int getNumCol(){return mBoundaryMatrix.size();}
	void getVCol(const int& colIdx, std::vector<int>& col);
	void getCol(const int& colIdx, std::vector<int>& col);
	void setCol(const int& colIdx, std::vector<int>& col);
	int getLowest(const int& colIdx);

private:
	void addCol(const int& i, const int& j);

	std::vector<std::vector<int>> mBoundaryMatrix;
	std::vector<std::vector<int>> mVMatrix;
	std::vector<int> mLowestColLookUp;
	std::vector<int> mBuffer;
};

#endif