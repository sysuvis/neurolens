#include "GplGraph.h"
#include <cstring>
#include <cmath>
#include "Delaunay.h"
#include <ctime>

inline GplNode makeGplNode(const vec2f& pos, const float& size){
	GplNode ret = {pos.x, pos.y, size, 0};
	return ret;
}

inline GplEdge makeGplEdge(const int& i, const int& j, const float& size){
	GplEdge ret = {i, j, size};
	return ret;
}

inline GplTriangles makeGplTriangle(const int& i, const int& j, const int& k){
	GplTriangles ret = {i, j, k};
	return ret;
}

inline GplTriEdges makeGplTriEdge(const int& i, const int& j){
	GplTriEdges ret = {i, j};
	return ret;
}

GplGraph::GplGraph():
dist(NULL),
dist_data(NULL),
reachDist(NULL),
reachDist_data(NULL)
{
}

GplGraph::~GplGraph(){
	clear();	
}

int GplGraph::addNode(const vec2f& pos, const float& size){
	nodes.push_back(makeGplNode(pos, size));
	return (nodes.size()-1);
}

int GplGraph::addNode(const float& x, const float& y, const float& size){
	return addNode(makeVec2f(x, y), size);
}

void GplGraph::addEdge(const int& i, const int& j, const float& len, const float& size){
	edges.push_back(makeGplEdge(i, j, size));
	if(dist) dist[i][j] = dist[j][i] = len;
}

void GplGraph::allocateDistMatrix(const int& n){
	if(dist) delete[] dist;
	if(dist_data) delete[] dist_data;

	dist = new float*[n];
	dist_data = new float[n*n];
	for(int i=0; i<n; ++i) dist[i]=&dist_data[i*n];
	for(int i=0; i<n*n; ++i) dist_data[i] = GPL_INFINITE;
}

void GplGraph::allocateReachDistMatrix(const int& n){
	if(reachDist) delete[] reachDist;
	if(reachDist_data) delete[] reachDist_data;

	reachDist = new float*[n];
	reachDist_data = new float[n*n];
	for(int i=0; i<n; ++i) reachDist[i]=&reachDist_data[i*n];
}

void GplGraph::createFromDistanceMatrix(float* mat, float* nodeSize, const int& num, const float& thresh){
	allocateDistMatrix(num);
	memcpy(dist_data, mat, sizeof(float)*num*num);
	srand(time(NULL));
	for (int i=0; i<num; ++i) {
		addNode((rand()%100000)/1000.0f, (rand()%100000)/1000.0f, nodeSize[i]);
		for (int j=i+1; j<num; ++j) {
			if (dist[i][j]>0 && dist[i][j]<thresh) {
				addEdge(i, j, dist[i][j], thresh-dist[i][j]);
			} else {
				dist[j][i]=dist[i][j]=0;
			}
		}
	}
	updateNumNeighbors(thresh);
}

void GplGraph::updateNumNeighbors(const float& thresh){
	for (int i=0; i<nodes.size(); ++i){
		for (int j=i+1; j<nodes.size(); ++j){
			if (dist[i][j]>0 && dist[i][j]<thresh){
				++nodes[i].numNeighbors;
				++nodes[j].numNeighbors;
			}
		}
	}
}

void GplGraph::randomLayout(){
	for (int i=0; i<nodes.size();++i) {
		nodes[i].pos.x = (rand()%10000)/100.0f;
		nodes[i].pos.y = (rand()%10000)/100.0f;
	}
}

void GplGraph::clear(){
	nodes.clear();
	edges.clear();
	triangles.clear();
	triEdges.clear();
	if(dist) delete[] dist;
	if(dist_data) delete[] dist_data;
	if(reachDist) delete[] reachDist;
	if(reachDist_data) delete[] reachDist_data;
}

void GplGraph::getRange(float& left, float& right, float& bottom, float& top){
	left = _left;
	right = _right;
	bottom = _bottom;
	top = _top;
}

void GplGraph::updateRange(){
	_right = _top = -GPL_INFINITE;
	_left = _bottom = GPL_INFINITE;

	for (int i=0; i<nodes.size(); ++i){
		if (nodes[i].pos.x<_left){
			_left = nodes[i].pos.x;
		}
		if (nodes[i].pos.x>_right){
			_right = nodes[i].pos.x;
		}
		if (nodes[i].pos.y<_bottom){
			_bottom = nodes[i].pos.y;
		}
		if (nodes[i].pos.y>_top){
			_top = nodes[i].pos.y;
		}
	}
}

void GplGraph::updateReachabilityDist(const float& thresh){
	int i, j, k, n=nodes.size();
	float d;

	float **adjDist = new float*[n];
	float *adjDist_data = new float[n*n];

	for (i=0; i<n; ++i) adjDist[i] = &adjDist_data[i*n];
	for(i=0; i<n*n; ++i) adjDist_data[i]=GPL_INFINITE;
	for(k=0; k<edges.size(); ++k){
		i = edges[k].i;
		j= edges[k].j;
		if (dist[edges[k].i][edges[k].j]<=1.01f) {
			adjDist[i][j]=adjDist[j][i]=dist[edges[k].i][edges[k].j];
		}
	}

	for (k=0; k<n; ++k) {
		for (i=0; i<n; ++i) {
			for (j=0; j<n; ++j) {
				if ((d=adjDist[i][k]+adjDist[k][j])<adjDist[i][j]){
					adjDist[i][j]=d;
				}
			}
		}
	}

	float adjDistThresh = thresh*2.0f;
	for(i=0; i<n*n; ++i) reachDist_data[i]=GPL_INFINITE;
	for(k=0; k<edges.size(); ++k){
		i = edges[k].i;
		j= edges[k].j;
		if (dist[edges[k].i][edges[k].j]<=thresh && adjDist[edges[k].i][edges[k].j]<=adjDistThresh) {
			reachDist[i][j]=reachDist[j][i]=dist[edges[k].i][edges[k].j];
		}
	}

	for (k=0; k<n; ++k) {
		for (i=0; i<n; ++i) {
			for (j=0; j<n; ++j) {
				if ((d=reachDist[i][k]+reachDist[k][j])<reachDist[i][j]){
					reachDist[i][j]=d;
				}
			}
		}
	}

	delete[] adjDist_data;
	delete[] adjDist;
}

void GplGraph::triangulate(const float& thresh_graph, const float& thresh_reach){
	triangles.clear();
	triEdges.clear();

	vertexSet vs;
	int i, j, k;
	for (i=0; i<nodes.size(); ++i) {
		vertex v(nodes[i].pos.x, nodes[i].pos.y, i);
		vs.insert(v);
	}

	Delaunay delaunay;
	triangleSet ts;
	edgeSet es;
	delaunay.Triangulate(vs, ts);
	delaunay.TrianglesToEdges(ts, es);


	for (tIterator it=ts.begin(); it!=ts.end(); ++it) {
		i = (*it).GetVertex(0)->GetId();
		j = (*it).GetVertex(1)->GetId();
		k = (*it).GetVertex(2)->GetId();
		if (isTriangleEdgeValid(i,j,thresh_graph,thresh_reach) 
			&& isTriangleEdgeValid(i,k,thresh_graph,thresh_reach) 
			&& isTriangleEdgeValid(j,k,thresh_graph,thresh_reach)) {
			triangles.push_back(makeGplTriangle(i, j, k));
		}
	}
	for (edgeIterator it=es.begin(); it!=es.end(); ++it) {
		i = (*it).m_pV0->GetId();
		j = (*it).m_pV1->GetId();
		if (isTriangleEdgeValid(i,j,thresh_graph,thresh_reach)) {
			triEdges.push_back(makeGplTriEdge(i, j));
		}
	}

	vs.clear();
	ts.clear();
	es.clear();
}

bool GplGraph::isTriangleEdgeValid( int i, int j, const float& thresh, const float& threshReach){
	float dist = vec2dLen(nodes[i].pos-nodes[j].pos);
	if (reachDist){
		return (reachDist[i][j]<threshReach && dist<thresh);
	}
	return (dist<thresh);
}