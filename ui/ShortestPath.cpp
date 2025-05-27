#include "ShortestPath.h"
#include <string>
#include <list>

void FloydWarshall(float* dist, int* path_mat, const int& n){
	float** dmat = new float*[n];
	int** pmat = new int*[n];
	int i, j, k;
	for (i=0; i<n; ++i) dmat[i] = &dist[i*n];
	for (i=0; i<n; ++i) pmat[i] = &path_mat[i*n];
	memset(path_mat, 0xff, sizeof(int)*n*n);

	float d;
	for (k=0; k<n; ++k) {
		for (i=0; i<n; ++i) {
			for (j=0; j<n; ++j) {
				if ((d=(dmat[i][k]+dmat[k][j]))<dmat[i][j]) {
					dmat[i][j] = d;
					pmat[i][j] = k;
				}
			}
		}
	}


	delete[] dmat;
	delete[] pmat;
}

void FloydWarshall(float* org, float* ret, int* path_mat, const int& n){
	memcpy(ret, org, sizeof(float)*n*n);
	FloydWarshall(ret, path_mat, n);
}

void recoverPath(const int& i, const int& j, int* path_mat, const int& n, std::vector<int>& ret_path){
	int **pmat = new int*[n];
	for(int i=0; i<n; ++i) pmat[i] = &path_mat[i*n];

	recoverPath(i, j, pmat, ret_path);

	delete[] pmat;
}

void recoverPath(std::list<int>::iterator& b, std::list<int>::iterator& e, std::list<int>& l, int** path_mat){
	if(*b!=*e && path_mat[*b][*e]>=0){
		std::list<int>::iterator i = l.insert(e, path_mat[*b][*e]);
		recoverPath(b, i, l, path_mat);
		recoverPath(i, e, l, path_mat);
	}
}

void recoverPath(const int& i, const int& j, int** path_mat, std::vector<int>& ret_path){
	std::list<int> l;
	l.push_back(i);
	l.push_back(j);
	std::list<int>::iterator b = l.begin();
	std::list<int>::iterator e = l.begin();
	++e;

	recoverPath(b, e, l, path_mat);

	ret_path.resize(l.size());
	int cnt=0;
	for (b=l.begin(); b!=l.end(); ++b) {
		ret_path[cnt++] = *b;
	}
}