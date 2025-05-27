#ifndef VR_COMPLEX_H
#define VR_COMPLEX_H

//#include <set>
#include <vector>
#include <map>
#include "typeOperation.h"
#include "PersistenceReduction.h"

#define SIMPLEX_INFINITE	1e30
#define MAX_DIMENSION		3
#define MAX_VERTICES		(MAX_DIMENSION+1)

class GplGraph;
class FRLayout;

typedef struct{
	int index;
	int dim;
	int vertices[MAX_VERTICES];
	float creation;
	int children[MAX_VERTICES];//global index
	std::vector<int> parents;//position in dim+1 vector
} Simplex;

typedef struct{
	int creator;
	int killer;
} PersistencePair;

typedef struct{
	float creation;
	float dead;
	int index;
} FeatureInfo;

typedef struct{
	int index;
	int position;
	int dim;
	float creation;
} SimplexSortElement;

typedef struct {
	Range range;
	int dim;
	int pair_id;
} PersistenceBarcode;

bool operator < (const SimplexSortElement& a, const SimplexSortElement& b);

typedef std::vector<Simplex>				SimplexArray;
typedef std::map<int,std::vector<Simplex>>	SimplexMap;
typedef std::vector<int>					SimplexIndexArray;
typedef std::map<int,std::vector<int>>		SimplexIndexMap;

class VRComplex{
public:
	VRComplex(const int& k, const float& epsilon);
	VRComplex(const int& k, const float& epsilon,
		float* distMat, const int& num);
	~VRComplex();

	void incrementalVR();
	void filterComplex(const float& epsilon, const int& minDim, SimplexArray& ret);
	void initSortedSimplexArray(const int& maxDim = 0xffff);
	void filterSortedSimplexArray(const float& minCreate, const float& maxCreate);
	void computePersistencePairs();//the default value should be big enough
	GplGraph* genPersistenceBarcodeGraph(const float& min_persist=-1.0f);
	void genPersistenceBarcode(std::vector<PersistenceBarcode>& ret, const float& min_persist = -1.0f, const int& min_dim=-1);
	void genPersistenceBarcode(std::vector<Range>& ret, const float& min_persist = -1.0f, const int& min_dim = -1);
	FRLayout* genLayout(SimplexArray& simplices, const float& c, const float& k);
	FRLayout* genLayout(const float& epsilon, const int& minDim, const float& c, const float& k, SimplexArray& ret);
	void updateCreation();

	std::vector<std::vector<int>>* getLoops(){return &mLoops;}
	std::vector<std::vector<int>>* getVoidVertices(){return &mVoidVertices;}
	void genVoidsFromUnpaired();

protected:
	inline virtual float getVertexCreation(const int& u){return 0.0f;}
	inline virtual float getDistance(const int& u, const int& v);
	virtual void lowerNBRs(const int& p, std::vector<int>& ret);
	virtual void intersectWithLowerNBRs(const std::vector<int>& N, const int& p, std::vector<int>& M);
	void addCofaces(Simplex& t, std::vector<int>& neighbors);
	int addSimplex(Simplex& t);
	bool compareSimplex(const Simplex& low, const Simplex& high, const int& skip);
	int findSimplex(const Simplex& high, const int& skip, const int& startIdx);
	int getNumIntersect(const Simplex& s1, const Simplex& s2);
	Simplex& getSimplex(const int& idx);

	bool augmentReductionByLoop(std::vector<int>& loop);

	virtual void updateAdjacentMatrix();
	bool areNeighbors(const int& p1, const int& p2);
	
	int		mK;
	int		mNumVertices;
	float	mEpsilon;

	float**	mDistMat;
	bool**	mAdjMat;
	bool*	mAdjMatData;

	int						mNumSimplex;
	SimplexMap				mSimplexMap;
	std::vector<vec2i>		mSimplexIndexLookUp;
	std::vector<int>		mVertexSimplexPosition;
	std::map<vec2i,int>		mEdgeSimplexPosition;
	std::map<vec3i,int>		mTriangleSimplexPosition;
	//persistence
	std::vector<PersistencePair>	mPersistencePairs;
	PersistenceReduction*			mReduction;
	std::vector<SimplexSortElement>	mSortedSimplex;
	std::vector<int>				mIndexToReductionCol;

	//detected feature
	std::map<int,std::vector<FeatureInfo>> mUnpaired;
	std::vector<std::vector<int>> mLoops;
	std::vector<std::vector<int>> mVoids;
	std::vector<std::vector<int>> mVoidVertices;
};

#endif //VR_COMPLEX_H