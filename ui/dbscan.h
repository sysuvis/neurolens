#ifndef DBSCAN_H
#define DBSCAN_H

#include "math.h"
#include <vector>
#include "TypeOperation.h"
#include "WindowsTimer.h"


static inline bool find(std::vector<int> v, int f) {
	bool result = false;
	for (int i = 0; i < v.size(); i++) {
		if (v[i] == f) {
			result = true;
			break;
		}
	}
	return result;
}

class DBSCAN {
public:
	enum Status {
		undefined = -2,
		noise
	};

	DBSCAN()
	:eps(5.0f),
	min_num_samples(10)
	{}
	~DBSCAN(){}

	void setEps(const float& _eps) { eps = _eps; }
	void setMinNumSamples(const int& _min_num_samples) { min_num_samples = _min_num_samples; }
	float getEps(){ return eps; }
	int getMinNumSamples() { return min_num_samples; }

	void fit() {
		fit(eps, min_num_samples);
	}

	void fit(const float& _eps, const int& _min_num_samples) {
		eps = _eps;
		min_num_samples = _min_num_samples;

		int n = dis.size();
		labels.assign(n, undefined);
		int c = 0;

		for (int i = 0; i < n; i++)
		{
			if (labels[i] == undefined)
			{
				std::vector<int> neighboor;
				getNeighbors(neighboor, i, eps);
				if (neighboor.size() + 1 >= min_num_samples)
				{
					labels[i] = c;
					for (int k = 0; k < neighboor.size(); ++k) {
						labels[neighboor[k]] = c;
					}

					for (int k = 0; k < neighboor.size(); k++)
					{
						//std::cout<<neighboor.size()<<std::endl;
						labels[neighboor[k]] = c;
						std::vector<int> N;
						getNeighbors(N, neighboor[k], eps);
						if (N.size() + 1 >= min_num_samples)
						{
							for (int l = 0; l < N.size(); l++)
							{
								if (labels[N[l]] == undefined) {
									neighboor.push_back(N[l]);
								}
								labels[N[l]] = c;
							}
						}

					}
					c++;

				} else {
					labels[i] = noise;
				}
			}
		}
	}

	std::vector<int> getLabels() { return labels; }

	void computeDistance(const std::vector<vec2f>& data) {
		int n = data.size();
		dis.resize(n);
		for (int i = 0; i < n; ++i) {
			dis[i].resize(n);
		}

		for (int i = 0; i < n; ++i) {
			dis[i][i] = makeSortElemInc(0.0f, i);
			for (int j = i + 1; j < n; ++j) {
				sortElemInc m;
				m.val = length(data[i] - data[j]);
				m.idx = j;
				dis[i][j] = m;
				m.idx = i;
				dis[j][i] = m;
			}
		}

		for (int i = 0; i < data.size(); i++) {
			std::sort(dis[i].begin(), dis[i].end());
		}
	}

	void setDistance(float* dist_mat, const int& n, const float& dist_thresh) {
		WindowsTimer timer;
		printf("dbscan setup:\n");
		dis.resize(n);
		for (int i = 0; i < n; ++i) {
			dis[i].clear();
			dis[i].reserve(n);
		}
		printf("allocate dist matrix: %f\n", timer.end());
		#pragma omp parallel for num_threads(4)
		for (int i = 0; i < n; ++i) {
			const float* di = dist_mat+i*n;
			for (int j = 0; j < n; ++j) {
				if (di[j] < dist_thresh) {
					dis[i].push_back(makeSortElemInc(di[j], j));
				}
			}
			//printf("%d ", dis[i].size());
		}
		printf("assign dist matrix values: %f\n", timer.end());
		#pragma omp parallel for num_threads(4)
		for (int i = 0; i < n; ++i) {
			std::sort(dis[i].begin(), dis[i].end());
		}
		printf("sort neighborhoods: %f\n\n", timer.end());
	}

private:
	float eps;
	int min_num_samples;
	std::vector<int> labels;
	std::vector<std::vector<sortElemInc>> dis;

	void getNeighbors(std::vector<int>& ret, const int& p, const float& eps) {
		for (int i = 0; i < dis[p].size(); i++) {
			if (dis[p][i].val <= eps) {
				ret.push_back(dis[p][i].idx);
			} else {
				break;
			}
		}
	}
};

#endif //DBSCAN_H