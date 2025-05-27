#include "VRComplex.h"
#include "FRLayout.h"
#include <algorithm>
#include "PersistenceReduction.h"

bool operator < (const SimplexSortElement& a, const SimplexSortElement& b){
	if (a.creation==b.creation){
		if (a.dim==b.dim){
			return (a.index<b.index);
		}
		return (a.dim<b.dim);
	}
	return (a.creation<b.creation);
}

VRComplex::VRComplex(const int& k, const float& epsilon)
:mK(k),
mNumVertices(0),
mEpsilon(epsilon),
mDistMat(NULL),
mAdjMat(NULL),
mAdjMatData(NULL),
mReduction(NULL),
mNumSimplex(0)
{
}

VRComplex::VRComplex(const int& k, const float& epsilon, 
					 float* distMat, const int& num)
:mK(k),
mNumVertices(num),
mEpsilon(epsilon),
mAdjMat(NULL),
mAdjMatData(NULL),
mReduction(NULL),
mNumSimplex(0)
{
	mAdjMatData = new bool[mNumVertices*mNumVertices];
	mAdjMat = new bool*[mNumVertices];
	mDistMat = new float*[mNumVertices];
	for (int i=0; i<mNumVertices; ++i){
		mDistMat[i] = &distMat[i*mNumVertices];
		mAdjMat[i] = &mAdjMatData[i*mNumVertices];
	}
}

VRComplex::~VRComplex(){
	for (SimplexMap::iterator it=mSimplexMap.begin(); it!=mSimplexMap.end(); ++it){
		it->second.clear();
	}
	mSimplexMap.clear();
	if(mDistMat) delete[] mDistMat;
	if(mAdjMat) delete[] mAdjMat;
	if(mAdjMatData) delete[] mAdjMatData;
}

void VRComplex::incrementalVR(){
	if(mDistMat) updateAdjacentMatrix();
	mVertexSimplexPosition.resize(mNumVertices, -1);
	mEdgeSimplexPosition.clear();
	mTriangleSimplexPosition.clear();

	//for all 0-simplices (vertices)
	std::vector<int> n;
	float creation;
	for (int i=0; i<mNumVertices; ++i){
		if((creation=getVertexCreation(i))<0.0f)
			continue;

		mVertexSimplexPosition[i]=mSimplexMap[0].size();
		n.clear();
		lowerNBRs(i, n);
		Simplex simplex;
		simplex.dim = 0;
		simplex.vertices[simplex.dim] = i;
		simplex.creation = creation;
		addCofaces(simplex, n);
	}
	//create look-up table
	mSimplexIndexLookUp.resize(mNumSimplex);
	for (SimplexMap::iterator it=mSimplexMap.begin(); it!=mSimplexMap.end(); ++it){
		SimplexArray& simplices = it->second;
		for (int i=0; i<simplices.size(); ++i){
			Simplex& s = simplices[i];
			mSimplexIndexLookUp[s.index].x = s.dim;
			mSimplexIndexLookUp[s.index].y = i;
		}
	}
}

inline float VRComplex::getDistance(const int& u, const int& v){
	return mDistMat[u][v];
}

void VRComplex::lowerNBRs(const int& p, std::vector<int>& ret){
	for (int i=0; i<p; ++i){
		if (mAdjMat[p][i]){
			ret.push_back(i);
		}
	}
}

void VRComplex::intersectWithLowerNBRs(const std::vector<int>& N, const int& p, std::vector<int>& M){
	for (int i=0; i<N.size(); ++i){
		if (N[i]>=p)
			break;
		if (mAdjMat[p][N[i]]){
			M.push_back(N[i]);
		}
	}
}

void VRComplex::addCofaces(Simplex& t, std::vector<int>& N){
	int tid = addSimplex(t);

	if(t.dim>=mK)
		return;

	int v;
	Simplex& prev = mSimplexMap[t.dim][tid];
	for (int i=0; i<N.size(); ++i){
		v = N[i];

		Simplex simplex = prev;
		++simplex.dim;
		simplex.vertices[simplex.dim] = v;
		//find creation time
		simplex.creation = prev.creation;
		float dist;
		for (int j=0; j<simplex.dim; ++j){
			if((dist=getDistance(v,prev.vertices[j]))>simplex.creation){
				simplex.creation = dist;
			}
		}

		simplex.children[simplex.dim] = prev.index;
		int startIdx = tid-1;
		for (int j=simplex.dim-1; j>=0; --j){
			startIdx = findSimplex(simplex, j, startIdx);
			Simplex& sub = mSimplexMap[t.dim][startIdx];
			sub.parents.push_back(mSimplexMap[t.dim+1].size());
			simplex.children[j] = sub.index;
		}

		std::vector<int> M;
		intersectWithLowerNBRs(N, v, M);
		
		addCofaces(simplex, M);
	}
}

int VRComplex::addSimplex(Simplex& t){
	t.index = mNumSimplex;
	if (t.dim==1){
		mEdgeSimplexPosition[makeVec2i(t.vertices[0],t.vertices[1])]=
			mSimplexMap[t.dim].size();
	} else if (t.dim==2){
		mTriangleSimplexPosition[makeVec3i(t.vertices[0],t.vertices[1],t.vertices[2])]=
			mSimplexMap[t.dim].size();
	}
	mSimplexMap[t.dim].push_back(t);
	++mNumSimplex;
	return (mSimplexMap[t.dim].size()-1);
}

bool VRComplex::areNeighbors(const int& p1, const int& p2){
	if(mDistMat[p1][p2]<mEpsilon){
		return true;
	}
	return false;
}

void VRComplex::updateAdjacentMatrix(){
	memset(mAdjMatData, 0, sizeof(bool)*mNumVertices*mNumVertices);

	int i, j;
	for (i=0; i<mNumVertices; ++i){
		for (j=i+1; j<mNumVertices; ++j){
			if (areNeighbors(i, j)){
				mAdjMat[i][j] = mAdjMat[j][i] = true;
			}
		}
	}
}

bool VRComplex::compareSimplex(const Simplex& low, const Simplex& high, const int& skip){
	for (int i=0,j=0; i<=low.dim; ++i, ++j){
		if (j==skip) ++j;
		if (low.vertices[i]!=high.vertices[j])
			return false;
	}
	return true;
}

int VRComplex::findSimplex(const Simplex& high, const int& skip, const int& startIdx){
	if (high.dim==1 && skip==0){
		return mVertexSimplexPosition[high.vertices[1]];
	} else if (high.dim==2){
		vec2i q;
		if (skip==0){
			q.x = high.vertices[1];
			q.y = high.vertices[2];
		} else if (skip==1){
			q.x = high.vertices[0];
			q.y = high.vertices[2];
		}
		return mEdgeSimplexPosition[q];
	} else if (high.dim==3){
		vec3i q;
		if (skip==0){
			q.x = high.vertices[1];
			q.y = high.vertices[2];
			q.z = high.vertices[3];
		} else if (skip==1){
			q.x = high.vertices[0];
			q.y = high.vertices[2];
			q.z = high.vertices[3];
		} else if (skip==2){
			q.x = high.vertices[0];
			q.y = high.vertices[1];
			q.z = high.vertices[3];
		}
		return mTriangleSimplexPosition[q];
	}
	//
	int idx = startIdx;
	int hid=0, lid=0;
	while (1){
		Simplex& low = mSimplexMap[high.dim-1][idx];
		if (hid==skip){
			++hid;
		}
		if (low.vertices[lid]!=high.vertices[hid]){
			--idx;
		} else {
			++lid; ++hid;
			if (lid==high.dim){
				break;
			}
		}
	}
	return idx;
}

void VRComplex::filterComplex(const float& epsilon, const int& minDim, SimplexArray& ret){
	for (SimplexMap::iterator it=mSimplexMap.begin(); it!=mSimplexMap.end(); ++it){
		if (it->first<minDim)
			continue;

		SimplexArray& simplices = it->second;
		for (int i=0; i<simplices.size(); ++i){
			Simplex& s = simplices[i];
			if (s.creation<epsilon){
				ret.push_back(s);
			}
		}
	}
}

int VRComplex::getNumIntersect(const Simplex& s1, const Simplex& s2){
	int i=0, j=0, cnt=0;
	int size1 = s1.dim+1;
	int size2 = s2.dim+1;

	while(i<size1 && j<size2){
		if (s1.vertices[i]<s2.vertices[j]){
			++j;
		} else if (s1.vertices[i]>s2.vertices[j]){
			++i;
		} else {
			++cnt; ++i; ++j;
		}
	}

	return cnt;
}

Simplex& VRComplex::getSimplex(const int& idx){
	return (mSimplexMap[mSimplexIndexLookUp[idx].x][mSimplexIndexLookUp[idx].y]);
}

void VRComplex::updateCreation(){//assume complex is constructed
	int i, j;
	for (i=0; i<mSimplexMap[0].size(); ++i){
		Simplex &s = mSimplexMap[0][i];
		s.creation = getVertexCreation(s.vertices[0]);
	}
	for (i=0; i<mSimplexMap[1].size(); ++i){
		Simplex &s = mSimplexMap[1][i];
		if (getVertexCreation(s.vertices[0])<0.0f||getVertexCreation(s.vertices[1])<0.0f) {
			s.creation = SIMPLEX_INFINITE;
		} else {
			s.creation = getDistance(s.vertices[0], s.vertices[1]);
		}		
	}
	int dim, child;
	float creation;
	for (SimplexMap::iterator it=mSimplexMap.begin(); it!=mSimplexMap.end(); ++it){
		if ((dim=it->first)<2) continue;
		SimplexArray& simplices = it->second;
		for (i=0; i<simplices.size(); ++i){
			Simplex& s = simplices[i];
			child = mSimplexIndexLookUp[s.children[0]].y;
			s.creation = mSimplexMap[dim-1][child].creation;
			for (j=1; j<=dim; ++j){
				child = mSimplexIndexLookUp[s.children[j]].y;
				if ((creation=mSimplexMap[dim-1][child].creation)>s.creation){
					s.creation = creation;
				}
			}
		}
	}
}

FRLayout* VRComplex::genLayout(SimplexArray& simplices, const float& c, const float& k){
	if (simplices.size()==0){
		return NULL;
	}

	FRLayout* layout = new FRLayout(c, k);
	layout->allocateDistMatrix(simplices.size());
	int maxDim = simplices[simplices.size()-1].dim;
	int minDim = simplices[0].dim-1;
	float invDim = 1.0f/(maxDim-minDim);

	int n;
	vec3i initPos;
	for (int i=0; i<simplices.size(); ++i){
		Simplex& s = simplices[i];
		layout->addNode(0, 0, sqrtf((s.dim-minDim)*invDim));
		for (int j=i+1; j<simplices.size(); ++j){
			n = getNumIntersect(simplices[i], simplices[j]);
			if (n!=0){
				layout->addEdge(i, j, 1.0f, (float)n/maxDim);
			}
		}
	}
	layout->updateNumNeighbors(2.0f);
	layout->randomLayout();
	layout->updateLayout();
	return layout;
}

FRLayout* VRComplex::genLayout(const float& epsilon, const int& minDim, const float& c, const float& k, SimplexArray& ret){
	filterComplex(epsilon, minDim, ret);
	FRLayout* layout = genLayout(ret, c, k);
	return layout;
}

void VRComplex::initSortedSimplexArray(const int& maxDim){
	mSortedSimplex.clear();
	//copy to sort array
	for (SimplexMap::iterator it=mSimplexMap.begin(); it!=mSimplexMap.end(); ++it){
		if (it->first>maxDim) continue;
		SimplexArray& simplices = it->second;
		for (int i=0; i<simplices.size(); ++i){
			Simplex& s = simplices[i];
			SimplexSortElement sse = {s.index, i, s.dim, s.creation};
			mSortedSimplex.push_back(sse);
		}
	}
	sort(mSortedSimplex.begin(), mSortedSimplex.end());
}

void VRComplex::filterSortedSimplexArray(const float& minCreate, const float& maxCreate){
	std::vector<SimplexSortElement> tmp;
	mSortedSimplex.swap(tmp);

	for (int i=0; i<tmp.size(); ++i){
		if (tmp[i].creation<=maxCreate && tmp[i].creation>=minCreate){
			mSortedSimplex.push_back(tmp[i]);
		}
	}
}

void VRComplex::computePersistencePairs(){
	if(mSimplexMap.empty() || mSortedSimplex.empty()) return;

	mIndexToReductionCol.clear();
	mIndexToReductionCol.resize(mNumSimplex, -1);
	for (int i=0; i<mSortedSimplex.size(); ++i) {
		mIndexToReductionCol[mSortedSimplex[i].index] = i;
	}

	int num_col = mSortedSimplex.size();
	if (!mReduction){
		mReduction = new PersistenceReduction(num_col);
	} else {
		mReduction->clear();
		mReduction->resize(num_col);
	}

	std::vector<int> temp_col;
	for (int i=0; i<mSortedSimplex.size(); ++i) {
		SimplexSortElement& sse = mSortedSimplex[i];
		Simplex& s = mSimplexMap[sse.dim][sse.position];
		temp_col.clear();
		if (s.dim!=0){
			for (int j=0; j<s.dim+1; ++j) {
				temp_col.push_back(mIndexToReductionCol[s.children[j]]);
			}
			sort(temp_col.begin(), temp_col.end());
		}
		mReduction->setCol(i, temp_col);
	}

	mReduction->reduce();

	//update persistence pairs
	mPersistencePairs.clear();
	int lowest;
	PersistencePair pair;
	for (int i=0; i<mSortedSimplex.size(); ++i){
		if ((lowest=mReduction->getLowest(i))<0)
			continue;

		pair.creator = mSortedSimplex[lowest].index;
		pair.killer = mSortedSimplex[i].index;
		mPersistencePairs.push_back(pair);
	}

	//update unpaired
	mUnpaired.clear();
	std::vector<bool> killed(mNumSimplex, false);
	std::vector<bool> zero_col(mNumSimplex, true);


	float create, dead;
	FeatureInfo f_info;
	for (int i=0; i<mPersistencePairs.size(); ++i){
		killed[mPersistencePairs[i].creator] = true;//creator should be valid
		zero_col[mPersistencePairs[i].killer] = false;
		Simplex& creator = getSimplex(mPersistencePairs[i].creator);
		Simplex& killer = getSimplex(mPersistencePairs[i].killer);
		create = creator.creation;
		dead = killer.creation;
		if (create+0.2000001f<dead) {
			f_info.creation = create;
			f_info.dead = dead;
			f_info.index = creator.index;
			mUnpaired[creator.dim].push_back(f_info);
		}
	}

	for (int i=0; i<mNumSimplex; ++i){
		if (zero_col[i] && !killed[i]){//create something and not killed
			Simplex &s = getSimplex(i);
			f_info.creation = s.creation;
			f_info.dead = SIMPLEX_INFINITE;
			f_info.index = s.index;
			mUnpaired[s.dim].push_back(f_info);
		}
	}
}

GplGraph* VRComplex::genPersistenceBarcodeGraph(const float& minPersist){
	GplGraph* barcode = new GplGraph();
	vec2i u, v;
	float start, end;
	int p1, p2;
	for (int i=0; i<mPersistencePairs.size();++i){
		if (mPersistencePairs[i].killer<0)//ignore those killed by dummy simplex
			continue;

		//find position in map
		u = mSimplexIndexLookUp[mPersistencePairs[i].creator];
		v = mSimplexIndexLookUp[mPersistencePairs[i].killer];
		//find creation time
		start = mSimplexMap[u.x][u.y].creation;
		end = mSimplexMap[v.x][v.y].creation;
		//filter out pairs with small persistence value
		if (end-start<minPersist) continue;

		p1 = barcode->addNode(i, start, 0.0f);
		p2 = barcode->addNode(i, end, 0.0f);
		barcode->addEdge(p1, p2, 1.0f, 1.0f);
	}

	barcode->updateRange();

	return barcode;
}

void VRComplex::genPersistenceBarcode(std::vector<PersistenceBarcode>& ret, const float& min_persist, const int& min_dim){
	ret.clear();

	vec2i u, v;
	PersistenceBarcode p;
	for (int i = 0; i < mPersistencePairs.size(); ++i) {
		if (mPersistencePairs[i].killer < 0)//ignore those killed by dummy simplex
			continue;

		//find position in map
		u = mSimplexIndexLookUp[mPersistencePairs[i].creator];
		v = mSimplexIndexLookUp[mPersistencePairs[i].killer];
		//find creation time
		Simplex& us = mSimplexMap[u.x][u.y];
		Simplex& vs = mSimplexMap[v.x][v.y];
		p.range.lower = us.creation;
		p.range.upper = vs.creation;
		//filter out pairs with small persistence value
		if (p.range.upper - p.range.lower < min_persist || us.dim<min_dim) continue;

		p.dim = us.dim;
		p.pair_id = i;

		ret.push_back(p);
	}
}

void VRComplex::genPersistenceBarcode(std::vector<Range>& ret, const float& minPersist, const int& min_dim){
	ret.clear();

	vec2i u, v;
	Range r;
	for (int i = 0; i < mPersistencePairs.size(); ++i) {
		if (mPersistencePairs[i].killer < 0)//ignore those killed by dummy simplex
			continue;

		//find position in map
		u = mSimplexIndexLookUp[mPersistencePairs[i].creator];
		v = mSimplexIndexLookUp[mPersistencePairs[i].killer];
		//find creation time
		Simplex& us = mSimplexMap[u.x][u.y];
		r.lower = us.creation;
		r.upper = mSimplexMap[v.x][v.y].creation;
		//filter out pairs with small persistence value
		if (r.upper - r.lower < minPersist || us.dim<min_dim) continue;

		ret.push_back(r);
	}
}

bool VRComplex::augmentReductionByLoop(std::vector<int>& loop){
	if (loop.empty() || mReduction->getNumCol()==0) return false;

	std::vector<int> temp_col;
	int k, u, v, uv;
	int hole_size = loop.size();
	int vertex = mReduction->getNumCol();

	mReduction->resize(vertex+2*hole_size+1);

	//add dummy vertex
	temp_col.clear();
	mReduction->setCol(vertex, temp_col);
	//add triangles and edges
	for (int j=0; j<hole_size; ++j) {
		k = (j>0)?(j-1):(hole_size-1);
		u = loop[j];
		v = loop[k];
		if (u>v) {
			uv = mEdgeSimplexPosition[makeVec2i(u,v)];
		} else {
			uv = mEdgeSimplexPosition[makeVec2i(v,u)];
		}
		uv = mIndexToReductionCol[mSimplexMap[1][uv].index];
		u = mIndexToReductionCol[mSimplexMap[0][mVertexSimplexPosition[u]].index];
		v = mIndexToReductionCol[mSimplexMap[0][mVertexSimplexPosition[v]].index];

		//edge (u, vertex),(v, vertex) added before
		temp_col.clear();
		temp_col.push_back(u);
		temp_col.push_back(vertex);
		mReduction->setCol(vertex+j+1, temp_col);

		//triangle 
		temp_col.clear();
		temp_col.push_back(uv);
		if (j>k) {
			temp_col.push_back(vertex+k+1);
			temp_col.push_back(vertex+j+1);
		} else {
			temp_col.push_back(vertex+j+1);
			temp_col.push_back(vertex+k+1);
		}
		mReduction->setCol(vertex+hole_size+j+1, temp_col);
	}

	mReduction->reduce(vertex);
	int lowest;
	for (int i=vertex+hole_size+1; i<mReduction->getNumCol(); ++i) {//for all triangle
		lowest=mReduction->getLowest(i);
		if (lowest>=0 && lowest<vertex) {//kill a previously existing feature
			return true;
		}
	}

	//roll back and return false
	mReduction->resize(vertex);
	return false;
}

void VRComplex::genVoidsFromUnpaired(){
	mVoids.clear();
	std::vector<int> v_col, void_vertices;
	int i, j, k;
	for (i=0; i<mUnpaired[2].size(); ++i){
		v_col.clear();
		void_vertices.clear();
		mReduction->getVCol(mIndexToReductionCol[mUnpaired[2][i].index], v_col);
		for (j=0; j<v_col.size(); ++j){
			k = mSortedSimplex[v_col[j]].index;
			v_col[j] = k;
			Simplex& s = mSimplexMap[mSimplexIndexLookUp[k].x][mSimplexIndexLookUp[k].y];
			void_vertices.push_back(s.vertices[0]);
			void_vertices.push_back(s.vertices[1]);
			void_vertices.push_back(s.vertices[2]);
		}
		mVoids.push_back(v_col);
		mVoidVertices.push_back(void_vertices);
	}
}