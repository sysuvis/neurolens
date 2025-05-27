#include "KMeans.h"
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <cstdio>

inline void updateIdx(float** dmat, std::vector<int>& idx, std::vector<int>& centers){
	float minv;
	for (int i=0; i<idx.size(); ++i){
		minv = dmat[i][centers[0]];
		idx[i] = 0;
		for (int j=1; j<centers.size(); ++j){
			if (dmat[i][centers[j]]<minv){
				minv = dmat[i][centers[j]];
				idx[i] = j;
			}
		}
	}
}

inline void updateDistAsCenter(float** dmat, std::vector<int>& idx, std::vector<int>& centers, std::vector<float>& distAsCenter){
	distAsCenter.assign(idx.size(), 0);
	for (int i=0; i<idx.size(); ++i){
		for (int j=0; j<i; ++j){
			if (idx[i]==idx[j]){
				distAsCenter[i] += dmat[i][j];
				distAsCenter[j] += dmat[i][j];
			}
		}
	}
}

inline void updateCenters(float** dmat, std::vector<int>& idx, std::vector<int>& centers, std::vector<float>& distAsCenter){
	for (int i=0; i<idx.size(); ++i){
		if (distAsCenter[i]<distAsCenter[centers[idx[i]]]){
			centers[idx[i]] = i;
		}
	}
}

inline float updateTotalDist(std::vector<int>& centers, std::vector<float>& distAsCenter){
	float ret = 0;
	for (int i=0; i<centers.size(); ++i){
		ret += distAsCenter[centers[i]];
	}
	return ret;
}

float kmeans(std::vector<int>& idx ,float* dist, const int& n, const int& k, const int& maxIter){
	idx.resize(n);

	float** dmat = new float*[n];
	for (int i=0; i<n; ++i){
		dmat[i] = &dist[i*n];
	}

	std::vector<int> centers(n);
	for (int i=0; i<n; ++i) centers[i] = i;

	srand(time(NULL));
	int ran, tmp;
	for (int i=0; i<k; ++i) {
		ran = rand()%(n-i)+i;
		tmp = centers[i];
		centers[i] = centers[ran];
		centers[ran] = tmp;
	}
	centers.resize(k);

	std::vector<float> distAsCenter(n);

	float cur=0.0f, prev;
	int count = 0;
	printf("Kmeans clustering:      0 iter.");
	do{
		prev = cur;
		updateIdx(dmat, idx, centers);
		updateDistAsCenter(dmat, idx, centers, distAsCenter);
		updateCenters(dmat, idx, centers, distAsCenter);
		cur = updateTotalDist(centers, distAsCenter);
		++count;
		if(!(count&0xf)){
			printf("\rKmeans clustering: %5i iters.", count);
		}
	} while (abs(cur-prev)>0.0001f && count<maxIter);
	printf("\n");

	for (int i=0; i<n; ++i){
		idx[i] = centers[idx[i]];
	}

	return cur;
}

void normalizeIndex(std::vector<int>& idx, int num){
	int numCenter = 0;
	int *mk = new int[num];

	memset(mk, 0, sizeof(int)*num);

	for (int i=0; i<num; ++i) {
		if(!mk[idx[i]]){
			mk[idx[i]] = numCenter;
			++numCenter;
		}
	}

	for (int i=0; i<num; ++i) {
		idx[i] = mk[idx[i]];
	}

	delete[] mk;
}