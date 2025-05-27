#ifndef KMEANS_H
#define KMEANS_H

#include <vector>

float kmeans(std::vector<int>& idx ,float* dist, const int& n, const int& k=20, const int& maxIter=2000);

void normalizeIndex(std::vector<int>& idx, int num);

#endif//KMEANS_H