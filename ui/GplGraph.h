#ifndef GPL_GRAPH_H
#define GPL_GRAPH_H

#include "typeOperation.h"
#include <cstdlib>
#include <vector>

using namespace std;

#define GPL_X			0
#define GPL_Y			1
#define GPL_INFINITE	1e30

typedef struct{
	int i, j, k;
} GplTriangles;

typedef struct{
	int i, j;
} GplTriEdges;

typedef struct{
	union{
		vec2f	pos;
		float	posArr[2];
	};
	float size;
	int numNeighbors;
} GplNode;

typedef struct{
	int i, j;
	float size;
} GplEdge;

inline GplNode makeGplNode(const vec2f& pos, const float& size);
inline GplEdge makeGplEdge(const int& i, const int& j, const float& size);
inline GplTriangles makeGplTriangle(const int& i, const int& j, const int& k);
inline GplTriEdges makeGplTriEdge(const int& i, const int& j);

class GplGraph{
public:
	GplGraph();
	~GplGraph();

	int	addNode(const vec2f& pos, const float& size);
	int addNode(const float& x, const float& y, const float& size);
	void addEdge(const int& i, const int& j, const float& len, const float& size);

	void allocateDistMatrix(const int& n);
	void allocateReachDistMatrix(const int &n);
	void createFromDistanceMatrix(float* mat, float* nodeSize, const int& num, const float& thresh);
	void updateNumNeighbors(const float& thresh);
	
	virtual void updateLayout(){}
	void randomLayout();
	void clear();

	vector<GplNode>* getNodes(){return &nodes;}
	vector<GplEdge>* getEdges(){return &edges;}
	vector<GplTriEdges>* getTriangleEdges(){return &triEdges;}

	void getRange(float& left, float& right, float& bottom, float& top);
	void updateRange();

	float** getDistanceMatrix(){return dist;}
	float** getReachDistanceMatrix(){return reachDist;}
protected:
	void updateReachabilityDist(const float& thresh);
	void triangulate(const float& thresh_graph, const float& thresh_reach);
	bool isTriangleEdgeValid( int i, int j, const float& thresh, const float& threshReach);
	
	vector<GplNode>			nodes;
	vector<GplEdge>			edges;

	float					**dist;
	float					**reachDist;
	float					*dist_data;
	float					*reachDist_data;
	
	//Triangulation
	vector<GplTriangles>	triangles;
	vector<GplTriEdges>		triEdges;

	float _left;
	float _right;
	float _bottom;
	float _top;
};

#endif//GRAPH_H