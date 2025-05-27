#ifndef SHORTEST_PATH_H
#define SHORTEST_PATH_H

#include <vector>

void FloydWarshall(float* dist, int* path_mat, const int& n);
void FloydWarshall(float* org, float* ret, int* path_mat, const int& n);

void recoverPath(const int& i, const int& j, int* path_mat, const int& n, std::vector<int>& ret_path);
void recoverPath(const int& i, const int& j, int** path_mat, std::vector<int>& ret_path);

#endif//SHORTEST_PATH_H