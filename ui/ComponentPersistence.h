#ifndef COMPONENT_PERSISTENCE_H
#define COMPONENT_PERSISTENCE_H

typedef struct{
	int index;
	int killer;
	float dead;
} VertexPersistence;

class GplGraph;

class ComponentPersistence{
public:
	ComponentPersistence(float* dist_mat, int n);
	~ComponentPersistence();

	GplGraph* genPersistenceBarcodeGraph(const float& minPersist=-1.0f, const bool& bOrder=false);
	int* genComponents(const int& k, float* ret_dist_thresh);
	
	void updateDistanceMatrix(float* dist_mat, const int& n);
	void computePersistence();

private:

	int		mN;

	VertexPersistence* mPersist;

	float** mDist;
	float*	mDistData;
};

#endif//COMPONENT_PERSISTENCE_H