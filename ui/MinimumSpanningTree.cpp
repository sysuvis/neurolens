#include "MinimumSpanningTree.h"
#include <vector>
#include <queue>

typedef struct{
	int i, j;
	float len;
} MSTEdge;

class CompareMSTEdge{
public:
	bool operator()(MSTEdge& e1, MSTEdge& e2){
		if(e1.len>e2.len) return true;
		return false;
	}
};

inline int find_root_kruskal(const int& v, std::vector<int>& parent){
	if(v==parent[v]){
		return v;
	} else {
		return find_root_kruskal(parent[v], parent);
	}
}

void Kruskal(vec2i* edges, float* lens, const int& V, const int& E, std::vector<vec2i>& ret){
	std::vector<int> parent(V);
	for(int i=0; i<V; ++i) parent[i] = i;

	std::priority_queue<MSTEdge, std::vector<MSTEdge>, CompareMSTEdge> heap;
	for (int i=0; i<E; ++i) {
		MSTEdge tmp_edge = {edges[i].x, edges[i].y, lens[i]};
		heap.push(tmp_edge);
	}

	int ir, jr;
	while (!heap.empty()){
		MSTEdge min_edge = heap.top();
		heap.pop();
		if((ir=find_root_kruskal(min_edge.i,parent))!=(jr=find_root_kruskal(min_edge.j,parent))){
			ret.push_back(makeVec2i(min_edge.i, min_edge.j));
			if(ir>jr) parent[ir]=jr;
			else parent[jr]=ir;
		}
	}
}