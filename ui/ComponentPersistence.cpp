#include "ComponentPersistence.h"
#include "GplGraph.h"
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <queue>

bool operator < (const VertexPersistence& a, const VertexPersistence& b){
	if (a.dead==b.dead) {
		return (a.index<b.index);
	}
	return (a.dead>b.dead);
}

typedef struct{
	int i, j;
	float d;
} CPEdges;

typedef struct{
	int idx;
	int size;
} CPComponent;

bool operator < (const CPComponent& a, const CPComponent& b){
	if (a.size==b.size) {
		return (a.idx<b.idx);
	}
	return (a.size>b.size);
}

bool operator < (const CPEdges& a, const CPEdges& b){
	if (a.d==b.d) {
		if (a.i==b.i) {
			return (a.j<b.j);
		}
		return (a.i<b.i);
	}
	return (a.d<b.d);
}

ComponentPersistence::ComponentPersistence(float* dist_mat, int n){
	mN = n;

	mDistData = dist_mat;
	mDist = new float*[n];
	for (int i=0; i<n; ++i) mDist[i] = &mDistData[i*n];

	mPersist = new VertexPersistence[n];

	computePersistence();
}

ComponentPersistence::~ComponentPersistence(){
	if (mDist!=NULL) delete[] mDist;
	if (mPersist!=NULL) delete[] mPersist;
}

void ComponentPersistence::updateDistanceMatrix(float* dist_mat, const int& n){
	if (dist_mat!=mDistData || n!=mN) {
		delete[] mDist;
		mDist = new float*[n];
		for (int i=0; i<n; ++i) mDist[i] = &mDistData[i*n];
		mDistData = dist_mat;
		mN = n;
	}
}

void ComponentPersistence::computePersistence(){
	float max_of_mins=-1e30, minv;
	for (int i=1; i<mN; ++i) {
		minv = 1e30;
		for (int j=0; j<i; ++j) if(i!=j && mDist[i][j]<minv){
			minv = mDist[i][j];
		}
		if(minv>max_of_mins) max_of_mins = minv;
	}
	max_of_mins += 0.001f;

	std::vector<CPEdges> edges;
	for (int i=0; i<mN; ++i) {
		for (int j=i+1; j<mN; ++j) if (mDist[i][j]<max_of_mins){
			CPEdges e = {i, j, mDist[i][j]};
			edges.push_back(e);
		}
	}

	sort(edges.begin(), edges.end());

	std::vector<int> lowest_look_up(mN, -1);
	for (int u=0,v=-1; u<edges.size(); ++u) {
		while((v=lowest_look_up[edges[u].j])!=-1){
			if (edges[u].i==edges[v].i) {
				edges[u].j = -1;
				break;
			} else if(edges[u].i>edges[v].i){
				edges[u].j = edges[u].i;
				edges[u].i = edges[v].i;
			} else {
				edges[u].j = edges[v].i;
			}
		}
		if (edges[u].j!=-1) {
			lowest_look_up[edges[u].j]=u;
		}
	}

	mPersist[0].index = 0;
	mPersist[0].killer = 0;
	mPersist[0].dead = 1e30;
	for (int i=1; i<mN; ++i) {
		mPersist[i].index = i;
		if (lowest_look_up[i]>=0) {
			mPersist[i].killer = edges[lowest_look_up[i]].i;
			mPersist[i].dead = edges[lowest_look_up[i]].d;
		} else {
			mPersist[i].killer = 0;
			mPersist[i].dead = max_of_mins;
		}
	}

	sort(mPersist, mPersist+mN);
	mPersist[0].dead = mPersist[1].dead*1.1f;
}

GplGraph* ComponentPersistence::genPersistenceBarcodeGraph(const float& minPersist, const bool& bOrder){
	GplGraph* barcode = new GplGraph();
	vec2i u, v;
	int p1, p2;
	if (bOrder) {
		for (int i=0; i<mN; ++i){
			p1 = barcode->addNode(i, 0, 0.0f);
			p2 = barcode->addNode(i, mPersist[i].dead, 0.0f);
			barcode->addEdge(p1, p2, 1.0f, 1.0f);
		}
	} else {
		for (int i=0; i<mN; ++i){
			p1 = barcode->addNode(mPersist[i].index, 0, 0.0f);
			p2 = barcode->addNode(mPersist[i].index, mPersist[i].dead, 0.0f);
			barcode->addEdge(p1, p2, 1.0f, 1.0f);
		}
	}
	

	barcode->updateRange();

	return barcode;
}

int* ComponentPersistence::genComponents(const int& k, float* ret_dist_thresh){
	int* components = new int[mN];
	memset(components, 0xff, sizeof(int)*mN);

	float max_dist = mPersist[k].dead;
	ret_dist_thresh[0] = max_dist;

	std::vector<int> cnt(k, 1);

	std::queue<int> q;
	int u, v;
	for (int i=0; i<k; ++i){
		q.push(mPersist[i].index);
		components[mPersist[i].index] = i;
		while(!q.empty()){
			u = q.front();
			q.pop();
			for (v=0; v<mN; ++v) {
				if (v!=u && components[v]<0 && mDist[u][v]<=max_dist) {
					q.push(v);
					components[v] = i;
					++cnt[i];
				}
			}
		}
	}

	//sort components by size
	std::vector<CPComponent> sort_components(k);
	for (int i=0; i<k; ++i) {
		sort_components[i].idx = i;
		sort_components[i].size = cnt[i];
	}
	sort(sort_components.begin(), sort_components.end());
	
	//size rank of i th-component
	std::vector<int> rank(k);
	for (int i=0; i<k; ++i) {
		rank[sort_components[i].idx] = i;
	}

	//re-assign component indices
	for (int i=0; i<mN; ++i) {
		components[i] = rank[components[i]];
	}

	return components;
}