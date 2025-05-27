#ifndef MINIMUM_SPANNING_TREE_H
#define MINIMUM_SPANNING_TREE_H

#include "typeOperation.h"
#include <vector>

void Kruskal(vec2i* edges, float* lens, const int& V, const int& E, std::vector<vec2i>& ret);

#endif //MINIMUM_SPANNING_TREE_H