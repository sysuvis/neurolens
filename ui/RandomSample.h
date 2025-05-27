#pragma once

#include <vector>
#include <cstdlib>
#include <cmath>
#include <ctime>

class NonUniformSampleTable {
public:
	NonUniformSampleTable(const float& _thresh, const int& _target_sample_num)
	:thresh(_thresh),
	target_num_entries(_target_sample_num)
	{
	}

	void createSampleIndex(const std::vector<float>& distribution) {
		float total=0.0f, maxv=thresh;
		for (int i = 0; i < distribution.size(); ++i) {
			if (distribution[i] > thresh) {
				total += distribution[i];
				if (distribution[i]>maxv) {
					maxv = distribution[i];
				}
			}
		}
		
		//discretize
		float importance_scale = target_num_entries / total;
		total *= target_num_entries;
		int count = 0;
		num_entries.resize(distribution.size());
		for (int i = 0; i < num_entries.size(); ++i) {
			if (distribution[i] > thresh) {
				num_entries[i] = (int)(distribution[i]* importance_scale);
				if (num_entries[i] == 0) {
					num_entries[i] = 1;
				}
				count += num_entries[i];
			}
			else {
				num_entries[i] = 0;
			}
		}

		table_entries.resize(count);
		count = 0;
		for (int i = 0; i < num_entries.size(); ++i) {
			if (num_entries[i] != 0) {
				for (int j = count; j < count + num_entries[i]; ++j) {
					table_entries[j] = i;
				}
				count += num_entries[i];
			}
		}
	}

	float thresh;
	int target_num_entries;

	std::vector<int> table_entries;
	std::vector<int> num_entries;
};

class NonUniformSampleBinary {
public:
	struct uitem {
		int idx;
		float probability;
	};

	struct bitem {
		uitem item1;
		int item2_id;
	};

	NonUniformSampleBinary() {

	}

	void updateDistributions(float* dist, const int& n) {
		createUnaryItems(dist, n);

		int m = uitems.size();
		float pivot = sum_prob / (m - 1);
		int pid = partitionUnaryItems(uitems, pivot);

		createBinaryDistributions(pid, pivot);
	}

	void createUnaryItems(float* dist, const int& n) {
		uitems.reserve(n);

		uitem u;
		sum_prob = max_prob = 0.0f;
		float prob;
		for (int i = 0; i < n; ++i) {
			if ((prob = dist[i]) > 0.0f) {
				u.idx = i;
				u.probability = prob;
				uitems.push_back(u);
				sum_prob += prob;
				if (prob > max_prob) max_prob = prob;
			}
		}
	}

	int partitionUnaryItems(std::vector<uitem>& arr, const float& pivot) {
		if (arr.empty()) return 0;

		int i = 0;
		for (int j = 0; j <= arr.size() - 1; ++j) {
			if (arr[j].probability < pivot) {
				if (i!=j){
					std::swap(arr[i], arr[j]);
				}
				++i;
			}
		}
		return i;
	}

	void createBinaryDistributions(int pid, const float& pval) {
		if (uitems.size() < 2) return;

		bitems.reserve(uitems.size() - 1);

		int i, j;
		for (i = 0; pid<uitems.size(); ++i) {
			uitem& ui = uitems[i];
			uitem& up = uitems[pid];
			addBinaryDistribution(ui, up, pval);
			if (up.probability < pval) {
				++pid;
			}
		}

		for (; i < uitems.size() - 1; ++i) {
			uitem& ui = uitems[i];
			for (j = i + 1; j < uitems.size(); ++j) {
				uitem& uj = uitems[j];
				if (uj.probability + ui.probability >= pval || j==uitems.size()-1) {
					addBinaryDistribution(ui, uj, pval);
					break;
				}
			}
		}
	}

	void addBinaryDistribution(uitem& ui, uitem& uj, const float& pval) {
		bitem b;

		b.item1 = ui;
		b.item2_id = uj.idx;
		uj.probability -= pval - ui.probability;

		b.item1.probability /= pval;

		bitems.push_back(b);
	}

	std::vector<bitem> bitems;
	std::vector<uitem> uitems;
	float sum_prob, max_prob;

};

typedef NonUniformSampleBinary::bitem NUSBItem;

template <typename T>
class RandomElements {
public:
	RandomElements(){}

	RandomElements(const std::vector<T>& _data) {
		data.assign(_data.begin(), _data.end());
	}

	RandomElements(const std::vector<T>& _data, const size_t& num_samples) {
		genRandomSamples(_data, num_samples);
	}

	void genRandomSamples(const std::vector<T>& _data, const size_t& num_samples) {
		data.assign(_data.begin(), _data.end());
		samples.assign(_data.begin(), _data.end());
		if (data.size() < num_samples) return;

		float rmax = RAND_MAX+1e-20;
		srand(time(NULL));
		for (int i = 0; i < num_samples; ++i) {
			int r = (rand() / rmax)*(num_samples-i);
			std::swap(samples[i], samples[i + r]);
		}
		samples.resize(num_samples);
	}

	void genRandomSamples(const size_t& num_samples) {
		genRandomSamples(data, num_samples);
	}

	std::vector<T> data;
	std::vector<T> samples;
};

//class NonUniformSampleBinaryCUDA {
//public:
//	struct item {
//		int idx;
//		float probability;
//	};
//
//	NonUniformSampleBinaryCUDA() {}
//	~NonUniformSampleBinaryCUDA() {}
//
//
//	float distribution_d;
//};