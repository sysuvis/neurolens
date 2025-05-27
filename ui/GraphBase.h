#pragma once

#include "typeOperation.h"
#include <vector>

template <typename NodeInfo, typename EdgeInfo, typename AdjacencyMatrix>
class GraphBase {
	template<typename EdgeInfo>
	struct EdgeBase {
		EdgeBase(const int& _i, const int& _j, const EdgeInfo& _e) {
			i = _i;
			j = _j;
			e = _e;
		}
		int i, j;//from i to j
		EdgeInfo e;
	};

public:
	GraphBase(){}

	GraphBase(const int& num_nodes){
		nodes.resize(num_nodes);
		graph.resize(num_nodes);
	}

	~GraphBase(){}

	void setNode(const int& nid, const NodeInfo& ninfo) { 
		nodes[nid] = ninfo; 
	}

	void setEdge(const int& i, const int& j, const EdgeInfo& einfo) {
		int eid = graph.getValue(i, j);
		edges[eid] = EdgeBase(i,j,einfo);
	}

	NodeInfo getNode(const int& nid) { return nodes[nid]; }
	EdgeInfo getEdge(const int& i, const int& j) { return edges[graph.getValue(i, j)]; }
	typename AdjacencyMatrix::Row& getNeighbors(const int& nid) { return graph.getRow(nid); }
	typename AdjacencyMatrix::row_iterator getNeighborBegin(const int& nid) { return graph.getRowBegin(nid); }
	typename AdjacencyMatrix::row_iterator getNeighborEnd(const int& nid) { return graph.getRowEnd(nid); }

	std::vector<NodeInfo> nodes;
	std::vector<EdgeBase<EdgeInfo>> edges;
	AdjacencyMatrix graph;
};

template <typename NodeInfo, typename EdgeInfo>
class DenseGraphBase : public GraphBase<NodeInfo, EdgeInfo, DenseMatrix<int>> {
public:
	DenseGraphBase():
	GraphBase()
	{}
	
	DenseGraphBase(const int& num_nodes):
	GraphBase(num_nodes)
	{}
};