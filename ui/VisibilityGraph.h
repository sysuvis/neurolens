#pragma once

#include "typeOperation.h"
#include "MatrixData.h"
#include "cudaStreamlineRenderer.h"
#include "cudaDeviceMem.h"
#include "cudaStreamlineTracer.h"
#include "definition.h"
#include "VolumeData.h"
#include "RandomSample.h"
#include "StreamlineResample.h"
#include "cnpy.h"
#include "WindowsTimer.h"
#include "dbscan.h"
#include <vector>
#include <rapidjson/document.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include "Registration.h"
#include <numeric>
#include<string>
//#include"FlowEncoder.h"
using namespace std;

#define TIMING_RESULT
#ifdef TIMING_RESULT
#define PRINT_TIME printf
#define START_TIMER mTimer.start()
#endif

#define FRECHET_NOT_INIT	-1.0f
#define FRECHET_INIT_THRESH	-0.9f
#define FRECHET_INFINITY	1e30

typedef struct {
	unsigned int uid, vid;
	unsigned int dist_loc;
} VisGraphCompTask;

typedef StreamlinePool::PointInfo VisGraphPointInfo;

typedef struct {
	std::string directory;
	std::string vec_field_path;
	std::string vec_hdr_path;
	std::string streamline_path;
	std::string vis_graph_path;
	std::string binormal_curve_path;
	std::string binormal_graph_path;
	std::string normal_curve_path;
	std::string normal_graph_path;
	std::string feature_path;
	std::string sample_vg_path;
	std::string all_vg_path;
	std::string latent_feature_path;
	std::string latent_segment_map_path;
	std::string point_latent_map_path;
	std::string tsne_layout_path;
	std::string tsne_latent_path;
	std::string tsne_visgraph_path;
	std::string tsne_id_path;
	std::string curvature_path;
	std::string vis_kunhua_dis_path;
	std::string vis_kunhua_dis_path2;
	std::string vis_kunhua_dis_file;
	std::string vis_kunhua_dis_file_path;
	std::vector<float> sample_steps;
	std::vector<float> filter_scales;
	std::vector<float> resample_scales;
	float segment_length;
	float radius;
	int store_gap_streamline;
	int store_gap_normal_binormal;
	int local_sample_num;
	int dist_thresh_num;
	int max_streamline_length;
	int num_streamlines;
	int num_random_samples;
	float normalize_factor;
	float reverse_normalize_factor;
	int latent_feature_dim;
	int latent_feature_full_dim;
	int sample_steps_num;
	int filter_scales_num;
	int resample_scales_num;
	int latent_feature_num;
	float curvature_sample_rate;
} VisGraphMetaInfo;

extern "C"
void computeVisibiltyGraphs_h(float* ret, VisGraphCompTask * tasks, vec3f * points_d,
	int ret_mat_size, int num_per_thread, int num_tasks);

extern "C"
void computeVisiblityGraphFixedLength_h(float* ret, const vec3f * points,
	int num_point_per_line, int num_per_thread, int num_lines);

extern "C"
void computeDegreeMap_h(float* ret_d, float* dist_mats, const int& dist_mat_size, int* matrix_offsets,
	int* line_num_points, const int& total_num_points, float* dist_threshes, const int& num_dist_thresh,
	const bool& b_acc);

extern "C"
void matchPattern_h(bool* ret, float* match_pattern_d, int pattern_len, int pattern_offset, float match_thresh,
	float* feature_map_d, int feature_dim, int num_per_thread, int num_points);

extern "C"
void findDistToPattern_h(float* ret, float* match_pattern_d, int pattern_len, int pattern_offset, float match_thresh,
	float* feature_map_d, int feature_dim, int num_per_thread, int num_points);

extern "C"
void findClosestTemplate_h(int* ret_d, float* templates_d, float* data_d, int num_templates, int num_data,
	int dim, int num_per_thread);

extern "C"
void computeNarrowMatrixColumnAverageVariance_h(float* ret_avg, float* ret_var, float* matrix_d,
	int num_cols, int num_rows, int num_per_thread);

extern "C" void computePerplexity_h(float* matrix_d, int n);

extern "C"
void filterLargeValues_h(float* data_d, const float& thresh, const float& replace_value, const int& num_per_thread,
	const int& num);

extern "C"
__host__ void filterMapUpperTriangle_h(float* maps_d, int* offsets_d, int* sizes_d, const float& thresh,
	const float& replace_value, const int& num);

extern "C"
void floydWarshall_h(float* maps_d, int* offsets_d, int* sizes_d, const int& num);

extern "C"
__host__ void computeDistanceMatrix_h(float* ret_d, float* vectors_d, int dim, int num);

class VisibilityGraph {
public:
	VisibilityGraph(const int& _n, const int& offset, float* adj_mat_data,
		float* degree_mat_data, const int& num_dist_thresh, float* latent_mat_data, const int& latent_dim) :
		n(_n),
		matrix_offset(offset),
		adj_mat(_n, _n, adj_mat_data),
		degree_mat(num_dist_thresh, _n, degree_mat_data),
		latent_mat(latent_dim, _n, latent_mat_data)
	{
	}

	~VisibilityGraph() {}

	int n;
	int matrix_offset;
	MatrixData<float> adj_mat;//n*n
	MatrixData<float> degree_mat;//n*num_dist_thresh
	MatrixData<float> latent_mat;//n*
};

class VisibilityGraphDataManager {
public:
	enum VisGraphFeatureBit {
		STREAMLINE_BIT = 1,
		BINORMAL_BIT = 2,
		NORMAL_BIT = 4
	};

	enum DBScanDataMode {
		tsne_layout = 0,
		tsne_latent,
		tsne_latent_perplexity
	};

	enum TsneColorMode {
		tsne_dataset_color = 0,
		tsne_dbscan_color
	};

	enum LatentDisplayMode {
		latent_raw_data = 0,
		latent_tsne_color
	};

	VisibilityGraphDataManager(const char* directory_path, const bool& init_dbscan = true) ://dir_path = "D:/data/project_data/VisibilityGraph/"
		mTracer(NULL),
		mDBScanDataMode(tsne_layout),
		mLatentDisplayMode(latent_tsne_color)
	{
		readMetaInfo(directory_path);

		//set default colors
		//设置默认的颜色值映射
		mContextColor = makeVec4f(ColorMap::getColorByName(ColorMap::Lavender_gray).xyz, 0.008f);
		mMatchColor = makeVec4f(ColorMap::getColorByName(ColorMap::Brandeis_blue).xyz, 0.2f);
		mQueryColor = makeVec4f(ColorMap::getColorByName(ColorMap::Harvard_crimson).xyz, 1.0f);

		////read or produce streamline pool
		//if (!readStreamlinePool(mPool, meta.streamline_path.c_str())) {
		//	genAndSaveRandomPool();
		//}
		
		processMATData();
		string streamline_path = "C:/Users/Administrator/Desktop/processed_data/streamline_file/003_S_2374_1_fibers_FA_normed2.stl";

		//获得流线池 mpool
		readBrainStreamlinePool(mPool,streamline_path.c_str());
		
		//allocate data: this need to be performed before computing the distance matrices
		allocateData();

		//for render data
		StreamlinePool renderPool(mPool);

		//create renderer: should be called after reading the streamline pool
		mRenderer = new cudaStreamlineRenderer(mPool.streamlines.data(), mPool.points.data(), mPool.streamlines.size(), 8, meta.radius);
		
		//等距离采样
		resample(mPool, 0.3);
		//均匀采样sample_numpoint个点
		getEqualPointStreamlinePool(mPool,32);
		//标准化
		normalization(mPool);
		
		printf("Dataset is: %s\n", dataset.c_str());
		
		//计算并导出3个representations
		//computeAndSaveVisiblityGraphs();
		//computeAndSaveRotate(mPool);
		//computeAndSaveLocation(mPool);
		

		
		//计算当前数据集距离矩阵，并存储到对应位置
		//computeAndSaveDistanceMatrix(mPool);

		//读取距离矩阵，放到GraphDisplay中，做成MatrixData，然后setData
		readAndSetDisData();
		
		
	


		/*
					// 这里读取了tsne的坐标点数据。
					if (read_array(mTsneLayout, meta.tsne_layout_path.c_str())) {
						int n = mTsneLayout.size();
						mTsneMatchIds.reserve(n);
						std::vector<float> tsne_latent_data;
						if (read_array(tsne_latent_data, meta.tsne_latent_path.c_str())) {
							float s = 0.001f;
							for (auto& f : tsne_latent_data) f *= s;
							mTsneLatentCuMem.allocate(tsne_latent_data.size());
							mTsneLatentCuMem.load(tsne_latent_data.data());
							if (!read_array(mLatentTsneMap, meta.tsne_id_path.c_str())
								&& !mLatentDataCuMem.empty())
							{
								findTsneIdsInLatentSpace();
							}
							if (init_dbscan) initDBScan();
						}
						mTsneLatentMap.resize(n);
						for (int i = 0; i < mLatentTsneMap.size(); ++i) {
						//NOTE: do not handle the special value now
							if (mLatentTsneMap[i]<0) {
								mLatentTsneMap[i] = 0;
								//printf("Abnormal feature value [%d].\n", i);
							}
							mTsneLatentMap[mLatentTsneMap[i]].push_back(i);
						}
						read_array(mTsneVisGraphData, meta.tsne_visgraph_path.c_str());
						for (float& f : mTsneVisGraphData) reverseFeatureTransform(f);
					}
		


					//compute degree map
					updateDistThreshes(makeRange(0.0f, 100.0f));
			*/
		mPool = renderPool;
	}

	~VisibilityGraphDataManager() {
	}

	void processMATData() {
		string MATdata_path = "C:/Users/Administrator/Desktop/processed_data/original_file/";
		string subject = "003_S_2374_1_fibers_FA_normed2";
		string file_path = MATdata_path + subject + ".bin";

		std::ifstream file(file_path, std::ios::binary | std::ios::ate);
		if (!file.is_open()) {
			std::cerr << "Error opening binary file" << std::endl;
		}

		std::streamsize fileSize = file.tellg();
		file.seekg(0, std::ios::beg);

		// 读取数据
		double* data = new double[fileSize / sizeof(double)];
		file.read(reinterpret_cast<char*>(data), fileSize);

		//设置变量
		int stl_num = static_cast<int>(data[0]);
		int sample_num = static_cast<int>(data[1]);
		int points_num = stl_num * sample_num;
		vector<vec3f> Points;
		vector<Streamline> Streamlines;

		for (int i = 2; i < points_num*3; i=i+3) {
			float x = static_cast<float>(data[i]);
			float y = static_cast<float>(data[i+1]);
			float z = static_cast<float>(data[i+2]);
			Points.push_back(makeVec3f(x, y, z));
		}

		for (int i = 0; i < stl_num; i++) {
			Streamlines.push_back(makeStreamline(i, i * sample_num, sample_num));
		}

		// 释放资源
		delete[] data;
		file.close();

		//导出新文件
		ofstream output_file("C:/Users/Administrator/Desktop/processed_data/streamline_file/003_S_2374_1_fibers_FA_normed2.stl", std::ios::binary);

		if (!output_file.is_open()) {
			std::cerr << "Error opening output file" << std::endl;
		}

		output_file.write((char*)(&points_num), sizeof(int));
		output_file.write((char*)(&stl_num), sizeof(int));
		output_file.write((char*)Points.data(), sizeof(vec3f) * points_num);
		output_file.write((char*)Streamlines.data(), sizeof(Streamline) * stl_num);

		output_file.close();



	}

	void computeAndSaveDistanceMatrix(StreamlinePool& mPool) {
		std::vector<float> dE, dG, dM, dH, dEP, dP, dF;
		std::vector<std::string> distype{ "dE", "dG", "dM", "dH", "dEP", "dP", "dF" };
		START_TIMER;
		PRINT_TIME("Timing: Computing all Distance Matrixes.");
		compute_dE(mPool, dE);
		compute_dG(mPool, dG);
		compute_dM(mPool, dM);
		compute_dH(mPool, dH);
		compute_dEP(mPool, dEP);
		compute_dP(mPool, dP);
		
		float* matrix = genDiscreteFrechetDistanceMatrix(mPool.streamlines.data(), mPool.streamlines.size(), mPool.points.data());
		for (int i = 0; i < mPool.streamlines.size() * mPool.streamlines.size(); ++i) {
			dF.push_back(matrix[i]);
		}
		
		float mean_dE = (std::accumulate(begin(dE), end(dE), 0.0) / dE.size());
		float mean_dG = (std::accumulate(begin(dG), end(dG), 0.0) / dG.size());
		float mean_dM = (std::accumulate(begin(dM), end(dM), 0.0) / dM.size());
		float mean_dH = (std::accumulate(begin(dH), end(dH), 0.0) / dH.size());
		float mean_dEP = (std::accumulate(begin(dEP), end(dEP), 0.0) / dEP.size());
		float mean_dP = (std::accumulate(begin(dP), end(dP), 0.0) / dP.size());
		float mean_dF = (std::accumulate(begin(dF), end(dF), 0.0) / dF.size());
		for (int i = 0; i < dE.size(); ++i) {
			dE[i] = dE[i] / mean_dE;
			dG[i] = dG[i] / mean_dG;
			dM[i] = dM[i] / mean_dM;
			dH[i] = dH[i] / mean_dH;
			dEP[i] = dEP[i] / mean_dEP;
			dP[i] = dP[i] / mean_dP;
			dF[i] = dF[i] / mean_dF;
		}

		mKunHuaDisMat.assign(dE.begin(), dE.end());

		mKunhuaDisMats.push_back(dE);
		mKunhuaDisMats.push_back(dG);
		mKunhuaDisMats.push_back(dM);
		mKunhuaDisMats.push_back(dH);
		mKunhuaDisMats.push_back(dEP);
		mKunhuaDisMats.push_back(dP);
		mKunhuaDisMats.push_back(dF);
		/*
		//mKunHuaDisMat_row:将二维的mKunhuaDisMats转成一维
		for (int i = 0; i < mKunhuaDisMats.size(); ++i) {
			for (int j = 0; j < mKunhuaDisMats[i].size(); ++j) {
				mKunHuaDisMat_row.push_back(mKunhuaDisMats[i][j]);
			}
		}*/
		//存储所有的distance matrixes数据
		for (int i = 0; i < mKunhuaDisMats.size(); ++i) {
			
			std::string dis_path = file_path + dataset+"/" + distype[i] + ".dat";
			write_array(mKunhuaDisMats[i].data(), mKunhuaDisMats[i].size(), dis_path.c_str());

		}
		
		PRINT_TIME(" Finish in %5.3f.\n\n", mTimer.end());
		
	}

	void readAndSetDisData() {
		std::vector<std::string> distype{ "dE", "dG", "dM", "dH", "dEP", "dP", "dF" };
		std::string data_path = file_path + dataset + "/";
		std::vector<float> disdata;
		for (int i = 0; i < distype.size(); ++i) {

			read_array(disdata, (data_path+distype[i]+".dat").c_str());
			mKunhuaDisMats.push_back(disdata);
		}



	}

	bool readBrainStreamlinePool(StreamlinePool& ret_pool, const char* file_path) {
		std::ifstream input_file;
		if (!open_file(input_file, file_path, true)) {
			return false;
		}

		int num_points, num_stls;
		input_file.read((char*)&num_points, sizeof(int));
		input_file.read((char*)&num_stls, sizeof(int));

		ret_pool.points.resize(num_points);
		ret_pool.streamlines.resize(num_stls);

		input_file.read((char*)ret_pool.points.data(), sizeof(vec3f) * num_points);
		input_file.read((char*)ret_pool.streamlines.data(), sizeof(Streamline) * num_stls);
		input_file.close();

		ret_pool.fillLineIds();

		return true;
	}
	bool readStreamlinePool(StreamlinePool& Pool, const char* file, bool brain = true) {
		std::ifstream input_file;
		if (!open_file(input_file, file, true)) {
			return false;
		}

		int num_points, num_stls;
		input_file.read((char*)&num_points, sizeof(int));
		input_file.read((char*)&num_stls, sizeof(int));

		float radius;
		if (!brain) {
			input_file.read((char*)&radius, sizeof(float));
		}

		Pool.points.resize(num_points);
		Pool.streamlines.resize(num_stls);

		input_file.read((char*)Pool.points.data(), sizeof(vec3f) * num_points);
		//  false
		std::vector<vec3f> uselessData;
		if (!brain) {
			
			uselessData.resize(num_points);
			input_file.read((char*)uselessData.data(), sizeof(vec3f) * num_points);
		}

		input_file.read((char*)Pool.streamlines.data(), sizeof(Streamline) * num_stls);
		input_file.close();

		Pool.fillLineIds();

		return true;
	}

	void readMetaInfo(const char* directory_path) {
		meta.directory.assign(directory_path);
		if (meta.directory[meta.directory.size() - 1] != '/') {
			meta.directory.push_back('/');
		}

		std::string meta_file_path = meta.directory + "meta.json";
		std::ifstream meta_file;
		open_file(meta_file, meta_file_path.c_str(), false);
		std::string meta_data((std::istreambuf_iterator<char>(meta_file)),
			std::istreambuf_iterator<char>());
		rapidjson::Document meta_info;
		meta_info.Parse(meta_data.c_str());
		//read strings
		//读维度、路径
		meta.vec_field_path = getAbsolutePath(meta.directory, meta_info["vector field path"].GetString());
		meta.vec_hdr_path = getAbsolutePath(meta.directory, meta_info["vector field header path"].GetString());
		meta.streamline_path = getAbsolutePath(meta.directory, meta_info["streamline path"].GetString());
		meta.vis_graph_path = getAbsolutePath(meta.directory, meta_info["visibility graph path"].GetString());
		meta.binormal_curve_path = getAbsolutePath(meta.directory, meta_info["binormal curve path"].GetString());
		meta.binormal_graph_path = getAbsolutePath(meta.directory, meta_info["binormal graph path"].GetString());
		meta.normal_curve_path = getAbsolutePath(meta.directory, meta_info["normal curve path"].GetString());
		meta.normal_graph_path = getAbsolutePath(meta.directory, meta_info["normal graph path"].GetString());
		meta.feature_path = getAbsolutePath(meta.directory, meta_info["feature path"].GetString());
		meta.sample_vg_path = getAbsolutePath(meta.directory, meta_info["sample VisGraph path"].GetString());
		meta.all_vg_path = getAbsolutePath(meta.directory, meta_info["all VisGraph path"].GetString());
		meta.latent_feature_path = getAbsolutePath(meta.directory, meta_info["latent feature path"].GetString());
		meta.latent_segment_map_path = getAbsolutePath(meta.directory, meta_info["latent to segment map"].GetString());
		meta.point_latent_map_path = getAbsolutePath(meta.directory, meta_info["point to latent map"].GetString());
		meta.tsne_layout_path = getAbsolutePath(meta.directory, meta_info["tsne layout"].GetString());
		meta.tsne_latent_path = getAbsolutePath(meta.directory, meta_info["tsne latent"].GetString());
		meta.tsne_visgraph_path = getAbsolutePath(meta.directory, meta_info["tsne visgraph"].GetString());
		meta.tsne_id_path = getAbsolutePath(meta.directory, meta_info["closest tsne id"].GetString());
		meta.curvature_path = getAbsolutePath(meta.directory, meta_info["curvature path"].GetString());
		meta.vis_kunhua_dis_path =  meta_info["kunhua dis path"].GetString();
		meta.vis_kunhua_dis_path2 = meta_info["kunhua dis path2"].GetString();
		meta.vis_kunhua_dis_file_path = meta_info["kunhua dis file path"].GetString();
		meta.vis_kunhua_dis_file = meta_info["kunhua dis file name"].GetString();
		//read parameters
		readArrayInMeta(meta.sample_steps, "sample steps", meta_info);
		readArrayInMeta(meta.filter_scales, "filter scales", meta_info);
		readArrayInMeta(meta.resample_scales, "resample scales", meta_info);

		meta.segment_length = meta_info["segment length"].GetFloat();
		meta.curvature_sample_rate = meta_info["curvature sample rate"].GetFloat();
		meta.radius = meta_info["radius"].GetFloat();
		meta.store_gap_streamline = meta_info["store gap of streamline"].GetInt();
		meta.store_gap_normal_binormal = meta_info["store gap of normal/binormal"].GetInt();
		meta.local_sample_num = meta_info["local sample number"].GetInt();
		meta.dist_thresh_num = meta_info["dist thresh number"].GetInt();
		meta.max_streamline_length = meta_info["max streamline length"].GetInt();
		meta.num_streamlines = meta_info["number of streamlines"].GetInt();
		meta.num_random_samples = meta_info["random sample number"].GetInt();
		meta.normalize_factor = meta_info["distance normalize factor"].GetFloat();
		meta.reverse_normalize_factor = 1.0f / meta.normalize_factor;
		meta.latent_feature_dim = meta_info["latent feature dim"].GetInt();
		meta.sample_steps_num = meta.sample_steps.size();
		meta.filter_scales_num = meta.filter_scales.size();
		meta.resample_scales_num = meta.resample_scales.size();
		meta.latent_feature_num = meta.sample_steps_num + meta.filter_scales_num + meta.resample_scales_num;
		if (meta.curvature_sample_rate > 1e-10) ++meta.latent_feature_num;
		meta.latent_feature_full_dim = meta.latent_feature_dim * meta.latent_feature_num;
	}

	void readArrayInMeta(std::vector<float>& ret, const std::string& array_name,const rapidjson::Document& meta_info)
	{
		const rapidjson::Value& array_value = meta_info[array_name.c_str()];
		for (int i = 0; i < array_value.Size(); ++i) {
			ret.push_back(array_value[i].GetFloat());
		}
	}

	bool isRelativePath(const std::string& file_path) {
		if (file_path.find(":/") != std::string::npos) {
			return false;
		}
		return true;
	}

	std::string getAbsolutePath(const std::string& directory, const std::string& file_path) {
		if (isRelativePath(file_path)) {
			return directory + file_path;
		}
		return file_path;
	}

	bool toSampleCurvature() {
		return (meta.curvature_sample_rate > 1e-10);
	}

	void allocateData() {
		int data_size = 0;
		//data_size为流线池所有流线的distance matrix的大小
		for (int i = 0; i < mPool.streamlines.size(); ++i) {
			int n = mPool.streamlines[i].numPoint;
			data_size += n * n;
		}
		//流线池点的数量
		int num_points = mPool.points.size();
		//流线池流线的数量
		int num_lines = mPool.streamlines.size();
		//数据初始化/分配空间
		mDistData.resize(data_size);//大小是所有visgraphs总和
		mKunHuaDisMat.resize(mPool.streamlines.size() * mPool.streamlines.size());	// 注意啊，这里何颂贤是写死了600*600个点，这个具体要怎么读还不知道。要看读进来有多少个流线，流线的距离吧
		//mKunHuaDisMat_row.resize(mPool.streamlines.size() * mPool.streamlines.size() * 7);
		//mKunHuaDisMat_row2.resize(mPool.streamlines.size() * mPool.streamlines.size());
		mDegreeData.resize(num_points * meta.dist_thresh_num);
		mDegreeDataCuMem.allocate(mDegreeData.size());
		allocateVisGraphs();
	}

	void allocateVisGraphs() {
		int data_offset = 0;
		mGraphs.resize(mPool.streamlines.size());

		for (int i = 0; i < mPool.streamlines.size(); ++i) {
			const Streamline& s = mPool.streamlines[i];
			//n为流线上点的数量
			int n = s.numPoint;

			//create graphs
			int latent_dim = (mLatentDisplayMode == latent_raw_data) ? (meta.latent_feature_full_dim) : meta.latent_feature_num;
			mGraphs[i] = new VisibilityGraph(n, data_offset, &mDistData[data_offset],
				&mDegreeData[s.start * meta.dist_thresh_num], meta.dist_thresh_num,
				mLatentDisplayData.data(), latent_dim);
			//data_offset为流线池所有流线的distance matrix的大小
			data_offset += n * n;
		}
	}

	void initDBScan(const std::vector<vec2f>& display_positions, float mDbscanEps,int mDbscanMinNumNeighbors) {
		int n = mPool.streamlines.size(), d;
		cudaDeviceMem<float> tsne_vector_cumem;
		float* tsne_vector_data_d;
		cudaDeviceMem<float> tsne_dist_mat_cumem(n * n);
		float max_dist;
		//if (mDBScanDataMode == tsne_layout) {
			d = 2;
			tsne_vector_cumem.allocate(d * n);
			tsne_vector_cumem.load((float*)display_positions.data());
			tsne_vector_data_d = tsne_vector_cumem.data_d;
			max_dist = 100.0f;
		//}
		//else {
		//	d = meta.latent_feature_dim;
		//	tsne_vector_data_d = mTsneLatentCuMem.data_d;
		//	max_dist = 10.0f;
		//}
		computeDistanceMatrix_h(tsne_dist_mat_cumem.data_d, tsne_vector_data_d, d, n);
		if (mDBScanDataMode == tsne_latent_perplexity) {
			computePerplexity_h(tsne_dist_mat_cumem.data_d, n);
		}
		std::vector<float> tsne_dist_mat(n * n);
		tsne_dist_mat_cumem.dump(tsne_dist_mat.data());
		mDBScan.setDistance(tsne_dist_mat.data(), n, max_dist);
		mDBScan.fit(mDbscanEps, mDbscanMinNumNeighbors);
	}

	//计算存储可见图
	void computeAndSaveVisiblityGraphs() {
		START_TIMER;
		PRINT_TIME("Timing: Computing all Visibility Graphs.");
	
		std::vector<VisGraphCompTask> tasks;
		tasks.reserve((mDistData.size() - mPool.points.size()) / 2);
		VisGraphCompTask t;
		//create tasks
		//创建任务
		int data_offset, n;
		for (int i = 0; i < mPool.streamlines.size(); ++i) {
			const Streamline& s = mPool.streamlines[i];
			n = s.numPoint;
			data_offset = getVisGraphOfStreamline(i)->matrix_offset;
			for (int j = 0; j < n; ++j) {
				//t.uid为流线上的第j个点
				t.uid = s.start + j;
				for (int k = j + 1; k < n; ++k) {
					//t.vid为流线上从start开始第k个点（第j个点之后）
					t.vid = s.start + k;
					//t.dist_loc为每条流线的distance matrix的第j行第k列，矩阵的位点(j,k)
					t.dist_loc = data_offset + j * n + k;
					tasks.push_back(t);
				}
			}
		}
		//compute the first half of the matrix visibility graph
		//计算上三角distance matrix
		computeVisibiltyGraphs_h(mDistData.data(), tasks.data(), mRenderer->getPoints_d(), mDistData.size(), 16, tasks.size());

		//fill the other half of the matricies
		//下三角通过上三角对称得到
		for (int i = 0; i < mGraphs.size(); ++i) {
			float** dist_mat = mGraphs[i]->adj_mat.getMatrixPointer();
			int n = mGraphs[i]->n;
			for (int j = 0; j < n; ++j) {
				//对角线
				dist_mat[j][j] = 0.0f;
				//下三角
				for (int k = 0; k < j; ++k) {
					dist_mat[j][k] = dist_mat[k][j];
				}
			}
			//在每条流线的distance matrix加上起始点和结束点的id
			const Streamline& s = mPool.streamlines[i];
			dist_mat[0][0] = s.start;
			//dist_mat[mGraphs.size() - 1][mGraphs.size() - 1] = s.start + s.numPoint - 1;
		}

		std::vector<float> vg;
		std::vector<vector<float>> vgs;
		for (int i = 0; i < mPool.streamlines.size(); ++i) {
			Streamline currentline = mPool.streamlines[i];
			for (int m = currentline.start; m < currentline.numPoint; ++m) {
				for (int n = currentline.start; n < currentline.numPoint; ++n) {
					float dist = dist3d(mPool.points[m], mPool.points[n]);
					vg.push_back(dist);
					
				}
			}
			vgs.push_back(vg);
		}

		std::vector<float> vgs_data;
		for (int i = 0; i < vgs.size(); ++i) {
			for (int j = 0; j < vgs[i].size(); ++j) {
				vgs_data.push_back(vgs[i][j]);
			}
		}

		PRINT_TIME(" Finish in %5.3f.\n\n", mTimer.end());

		std::string vis_graph_path = file_path+dataset+"/visibilitygraph.dat";
		write_array(vgs_data.data(), vgs_data.size(),vis_graph_path.c_str());
	}

	// 标准化
	void normalization(StreamlinePool& Pool) {
		for (int i = 0; i < Pool.streamlines.size(); ++i) {
			const Streamline& s = Pool.streamlines[i];
			// xyz的均值
			float x_mean = 0.0f, y_mean = 0.0f, z_mean = 0.0f;
			for (int j = s.start; j < s.start + s.numPoint; ++j) {
				x_mean += Pool.points[j].x;
				y_mean += Pool.points[j].y;
				z_mean += Pool.points[j].z;
			}
			// 平均点
			x_mean /= s.numPoint;
			y_mean /= s.numPoint;
			z_mean /= s.numPoint;

			float scale = 0.0f;
			for (int j = s.start; j < s.start + s.numPoint; ++j) {
				scale += sqrt((Pool.points[j].x - x_mean) * (Pool.points[j].x - x_mean) + (Pool.points[j].y - y_mean) * (Pool.points[j].y - y_mean) + (Pool.points[j].z - z_mean) * (Pool.points[j].z - z_mean));
			}
			for (int j = s.start; j < s.start + s.numPoint; ++j) {
				Pool.points[j].x = (Pool.points[j].x - x_mean) / scale;
				Pool.points[j].y = (Pool.points[j].y - y_mean) / scale;
				Pool.points[j].z = (Pool.points[j].z - z_mean) / scale;
			}
			TandS.push_back(makeVec4f(x_mean, y_mean, z_mean, scale));
		}
	}

	//计算并存储旋转数据
	void computeAndSaveRotate(StreamlinePool Pool, int train_num=1) {
		// const float PI = 3.141592;
		START_TIMER;
		PRINT_TIME("Timing: Computing all rotations.");
		// 归一化
		for (int i = 0; i < Pool.streamlines.size(); ++i) {
			const Streamline& s = Pool.streamlines[i];
			// xyz的均值
			float x_mean = 0.0, y_mean = 0.0, z_mean = 0.0;
			for (int j = s.start; j < s.start + s.numPoint; ++j) {
				x_mean += Pool.points[j].x;
				y_mean += Pool.points[j].y;
				z_mean += Pool.points[j].z;
			}
			x_mean /= s.numPoint;
			y_mean /= s.numPoint;
			z_mean /= s.numPoint;

			float dx_sum = 0.0, dy_sum = 0.0, dz_sum = 0.0;
			for (int j = s.start; j < s.start + s.numPoint; ++j) {
				dx_sum += (Pool.points[j].x - x_mean) * (Pool.points[j].x - x_mean);
				dy_sum += (Pool.points[j].y - y_mean) * (Pool.points[j].y - y_mean);
				dz_sum += (Pool.points[j].z - z_mean) * (Pool.points[j].z - z_mean);
			}
			dx_sum = sqrt(dx_sum);
			dy_sum = sqrt(dy_sum);
			dz_sum = sqrt(dz_sum);
			for (int j = s.start; j < s.start + s.numPoint; ++j) {
				Pool.points[j].x = (Pool.points[j].x - x_mean) / dx_sum;
				Pool.points[j].y = (Pool.points[j].y - y_mean) / dy_sum;
				Pool.points[j].z = (Pool.points[j].z - z_mean) / dz_sum;
			}
		}

        vector<vector<float>> rotatetheta;
		vector<vector<vector<vec3f>>> After_Rotate;

		// 旋转
		for (int i = 0; i < Pool.streamlines.size(); ++i) {
			const Streamline& s = Pool.streamlines[i];
			vec3f start = Pool.points[s.start];
			vec3f end = Pool.points[s.start + s.numPoint - 1];
			//一条流线旋转得到的流线们
			vector<vector<vec3f>> v;
			vector<float> theta_of_streamline;
			for (int j = 0; j < train_num; ++j) {
				//一条流线上的点
				vector<vec3f> points;
				float theta = (rand() % 360) * PI / 180;
				theta_of_streamline.push_back(theta);
				float** matrix = new float* [4];
				for (int i = 0; i < 4; i++) {
					matrix[i] = new float[4];
				}
				Rotate(matrix, start, end, theta);
				for (int k = s.start; k < s.start + s.numPoint; ++k) {
					vec3f currentPoint = Pool.points[k];
					currentPoint.x = matrix[0][0] * Pool.points[k].x + matrix[1][0] * Pool.points[k].y + matrix[2][0] * Pool.points[k].z + matrix[3][0] * 1;
					currentPoint.y = matrix[0][1] * Pool.points[k].x + matrix[1][1] * Pool.points[k].y + matrix[2][1] * Pool.points[k].z + matrix[3][1] * 1;
					currentPoint.z = matrix[0][2] * Pool.points[k].x + matrix[1][2] * Pool.points[k].y + matrix[2][2] * Pool.points[k].z + matrix[3][2] * 1;
					points.push_back(currentPoint);
				}
				v.push_back(points);
				delete matrix;
			}

			rotatetheta.push_back(theta_of_streamline);
			After_Rotate.push_back(v);
		}

		//先把上面两个存储数据refine了 rotatetheta->thetas   after_rotate->rotated_streamlines
		vector<float> thetas;
		vector<vector<vec3f>> rotated_streamlines;
		for (int i = 0; i < rotatetheta.size(); ++i) {
			thetas.push_back(rotatetheta[i][0]);
			rotated_streamlines.push_back(After_Rotate[i][0]);
		}

		vector<float> test_data1;//存放原始流线池数据
		vector<float> test_data2;//存放旋转后流线池数据
		

		
		for (int i = 0; i < Pool.streamlines.size(); ++i) {
			const Streamline& s = Pool.streamlines[i];
			for (int j = s.start; j < s.numPoint+s.start;++j) {
				test_data1.push_back(Pool.points[j].x);
				test_data1.push_back(Pool.points[j].y);
				test_data1.push_back(Pool.points[j].z);
			}

		}

		for (int i = 0; i < rotated_streamlines.size(); ++i) {
			for (int j = 0; j < rotated_streamlines[0].size(); ++j) {
				test_data2.push_back(rotated_streamlines[i][j].x);
				test_data2.push_back(rotated_streamlines[i][j].y);
				test_data2.push_back(rotated_streamlines[i][j].z);
			}
		}
		

		PRINT_TIME(" Finish in %5.3f.\n\n", mTimer.end());

		std::string rotate_dataset_path1 = file_path +dataset+ "/rotate_dataset1.dat";
		std::string rotate_dataset_path2 = file_path +dataset+ "/rotate_dataset2.dat";
		std::string rotate_label_path = file_path +dataset+ "/rotate_label.dat";

		write_array(test_data1.data(), test_data1.size(), rotate_dataset_path1.c_str());
		write_array(test_data2.data(), test_data2.size(), rotate_dataset_path2.c_str());
		write_array(thetas.data(), thetas.size(), rotate_label_path.c_str());
	}

	// 计算旋转变换矩阵
	void Rotate(float** matrix, vec3f p1, vec3f p2, float theta) {
		float a = p1.x;
		float b = p1.y;
		float c = p1.z;
		float normalization = sqrt((p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y) + (p2.z - p1.z) * (p2.z - p1.z));
		float u = (p2.x - p1.x) / normalization;
		float v = (p2.y - p1.y) / normalization;
		float w = (p2.z - p1.z) / normalization;
		float costheta = cosf(theta);
		float sintheta = sinf(theta);
		matrix[0][0] = u * u + (v * v + w * w) * costheta;
		matrix[0][1] = u * v * (1 - costheta) + w * sintheta;
		matrix[0][2] = u * w * (1 - costheta) - v * sintheta;
		matrix[0][3] = 0;

		matrix[1][0] = u * v * (1 - costheta) - w * sintheta;
		matrix[1][1] = v * v + (u * u + w * w) * costheta;
		matrix[1][2] = v * w * (1 - costheta) + u * sintheta;
		matrix[1][3] = 0;

		matrix[2][0] = u * w * (1 - costheta) + v * sintheta;
		matrix[2][1] = v * w * (1 - costheta) - u * sintheta;
		matrix[2][2] = w * w + (u * u + v * v) * costheta;
		matrix[2][3] = 0;

		matrix[3][0] = (a * (v * v + w * w) - u * (b * v + c * w)) * (1 - costheta) + (b * w - c * v) * sintheta;
		matrix[3][1] = (b * (u * u + w * w) - v * (a * u + c * w)) * (1 - costheta) + (c * u - a * w) * sintheta;
		matrix[3][2] = (c * (u * u + v * v) - w * (a * u + b * v)) * (1 - costheta) + (a * v - b * u) * sintheta;
		matrix[3][3] = 1;
	}

	// 计算并存储位置信息
	void computeAndSaveLocation(StreamlinePool Pool) {
		vector<vec3f> StartAndEnd;
		vector<float> StartAndEnd_coordinate;
		START_TIMER;
		PRINT_TIME("Timing: Computing all locations.");
		for (int i = 0; i < Pool.streamlines.size(); ++i) {
			const Streamline& s = Pool.streamlines[i];
			vec3f start = Pool.points[s.start];
			vec3f end = Pool.points[s.start + s.numPoint - 1];
			// 两个端点
			StartAndEnd.push_back(start);
			StartAndEnd.push_back(end);
			// 端点的空间坐标
			StartAndEnd_coordinate.push_back(start.x);
			StartAndEnd_coordinate.push_back(start.y);
			StartAndEnd_coordinate.push_back(start.z);
			StartAndEnd_coordinate.push_back(end.x);
			StartAndEnd_coordinate.push_back(end.y);
			StartAndEnd_coordinate.push_back(end.z);
		}
		PRINT_TIME(" Finish in %5.3f.\n\n", mTimer.end());
		std::string location_path = file_path +dataset+ "/location.dat";
		write_array(StartAndEnd_coordinate.data(), StartAndEnd_coordinate.size(), location_path.c_str());
	}




	void filterVisibilityGraph(std::vector<float>& ret, const float& thresh) {
		cudaDeviceMem<float> ret_d(mDistData.size());
		ret_d.load(mDistData.data());

		std::vector<int> offsets(mGraphs.size());
		std::vector<int> sizes(mGraphs.size());
		for (int i = 0; i < mGraphs.size(); ++i) {
			offsets[i] = mGraphs[i]->matrix_offset;
			sizes[i] = mGraphs[i]->n;
		}
		cudaDeviceMem<int> offsets_d(offsets);
		cudaDeviceMem<int> size_d(sizes);

		filterMapUpperTriangle_h(ret_d.data_d, offsets_d.data_d, size_d.data_d, thresh, 1e30, mGraphs.size());
		floydWarshall_h(ret_d.data_d, offsets_d.data_d, size_d.data_d, mGraphs.size());

		ret.resize(ret_d.size);
		ret_d.dump(ret.data());
	}

	void updateDistThreshes(const Range& dist_thresh_range) {
		mDistThreshes.resize(meta.dist_thresh_num);
		for (int i = 0; i < meta.dist_thresh_num; ++i) {
			mDistThreshes[i] = interpolate(0.0f, meta.dist_thresh_num - 1, i, dist_thresh_range.lower, dist_thresh_range.upper);
		}
		computeDegreeMap();
	}

	void computeDegreeMap() {
		std::vector<int> matrix_offsets;
		std::vector<int> line_num_points;
		for (int i = 0; i < mPool.streamlines.size(); ++i) {
			VisibilityGraph* g = getVisGraphOfStreamline(i);
			int n = g->n;
			for (int j = 0; j < n; ++j) {
				matrix_offsets.push_back(g->matrix_offset + j * n);
			}
			line_num_points.insert(line_num_points.end(), n, n);
		}

		computeDegreeMap_h(mDegreeDataCuMem.data_d, mDistData.data(), mDistData.size(), matrix_offsets.data(),
			line_num_points.data(), mPool.points.size(), mDistThreshes.data(), mDistThreshes.size(), true);
		mDegreeDataCuMem.dump(mDegreeData.data());

		mDegreeAvg.resize(mDistThreshes.size());
		mDegreeVar.resize(mDistThreshes.size());
		computeNarrowMatrixColumnAverageVariance_h(mDegreeAvg.data(), mDegreeVar.data(), mDegreeDataCuMem.data_d,
			mDegreeAvg.size(), mPool.points.size(), 32);
	}

	void findStartsBinormalNormalPool(std::vector<int>& ret, const StreamlinePool& pool, const int& num_samples) {
		int first_half = num_samples / 2;
		ret.assign(pool.streamlines.size(), -1);
		for (int i = 0; i < pool.streamlines.size(); ++i) {
			const Streamline& s = pool.streamlines[i];
			if (s.numPoint >= num_samples) {
				ret[i] = s.sid - first_half;
				if (ret[i] < 0) {
					ret[i] = 0;
				}
				else if (ret[i] + num_samples > s.numPoint) {
					ret[i] = s.numPoint - num_samples;
				}
				ret[i] += s.start;
			}
		}
	}

	void propagateValidStart(std::vector<int>& ret_starts, std::vector<int>& ret_ids,
		const std::vector<int>& starts, const bool& b_forward)
	{
		ret_starts.assign(starts.size(), -1);
		ret_ids.assign(starts.size(), -1);
		int prev_valid_start = -1, prev_valid_id = -1;
		int last_id = starts.size() - 1;
		for (int i = 0; i <= last_id; ++i) {
			int j = (b_forward > 0) ? i : (last_id - i);
			if (starts[j] >= 0) {
				prev_valid_start = starts[j];
				prev_valid_id = j;
			}
			else {
				ret_starts[j] = prev_valid_start;
				ret_ids[j] = prev_valid_id;
			}
		}
	}

	inline int findValidStart(const int& start, const int& i, const int& fw_start, const int& bw_start,
		const int& fw_id, const int& bw_id)
	{
		if (start >= 0) {
			return start;
		}
		int fw_diff = (fw_id >= 0 && mPool.line_ids[i] == mPool.line_ids[fw_id]) ? (i - fw_id) : 0xffffff;
		int bw_diff = (bw_id >= 0 && mPool.line_ids[i] == mPool.line_ids[bw_id]) ? (i - bw_id) : 0xffffff;
		if (fw_diff < bw_diff) {
			return fw_start;
		}
		else if (bw_diff < fw_diff) {
			return bw_start;
		}
		else if (fw_diff != 0xffffff) {
			return fw_start;
		}
		return -1;
	}

	void fixBinormalNormalPool(std::vector<vec3f>& ret, const StreamlinePool& pool, const int& num_samples) {
		int first_half = num_samples / 2;
		int second_half = num_samples - first_half;
		std::vector<int> starts;
		findStartsBinormalNormalPool(starts, pool, num_samples);

		std::vector<int> forward_starts, forward_ids, backward_starts, backward_ids;
		propagateValidStart(forward_starts, forward_ids, starts, true);
		propagateValidStart(backward_starts, backward_ids, starts, false);
		for (int i = 0; i < starts.size(); ++i) {
			starts[i] = findValidStart(starts[i], i, forward_starts[i],
				backward_starts[i], forward_ids[i], backward_ids[i]);
		}

		ret.clear();
		ret.reserve(pool.points.size() * num_samples);
		for (int i = 0; i < pool.streamlines.size(); ++i) {
			if (starts[i] >= 0) {
				const vec3f* p = &(pool.points[starts[i]]);
				ret.insert(ret.end(), p, p + num_samples);
			}
			else {
				printf("line not valid %d", i);
			}
		}
	}

	void genAndSaveRandomPool() {
		cudaStreamlineTracer* tracer = getTracer();
		StreamlineTraceParameter pars = getTracingParameters();
		tracer->genAndSaveRandomPool(pars, meta.num_streamlines, meta.streamline_path.c_str());
		readStreamlinePool(mPool, meta.streamline_path.c_str());
	}

	void computeAndSaveBinormalNormalCurves(std::vector<vec3f>& ret_curves, const std::string& file_path,
		const int& trace_type)
	{
		cudaStreamlineTracer* tracer = getTracer();
		StreamlineTraceParameter pars;
		pars.max_point = meta.local_sample_num;
		pars.segment_length = meta.segment_length;
		pars.store_gap = meta.store_gap_normal_binormal;
		pars.trace_type = trace_type;
		StreamlinePool pool;
		tracer->trace(pool, mPool.points.data(), mPool.points.size(), pars);
		fixBinormalNormalPool(ret_curves, pool, meta.local_sample_num);
		write_array(ret_curves.data(), ret_curves.size(), file_path.c_str());

	}

	void computeAndSaveBinormalNormalDistMat(std::vector<float>& ret_mats, const std::vector<vec3f>& curves,
		const std::string& file_path)
	{
		computeVisiblityGraphFixedLength_h(ret_mats.data(), curves.data(), meta.local_sample_num, 32, mPool.points.size());
		write_array(ret_mats.data(), ret_mats.size(), file_path.c_str());
	}

	IndexRange getDegreeIndex(const Range& dist_range) {
		IndexRange ret = makeIndexRange(-1, -1);
		for (int i = 0; i < mDistThreshes.size(); ++i) {
			if (inRange(dist_range, mDistThreshes[i])) {
				if (ret.lower < 0) {
					ret.lower = i;
				}
				ret.upper = i;
			}
			else if (ret.lower >= 0) {
				break;
			}
		}
		return ret;
	}

	void matchDegreePattern(std::vector<StreamlineSegment>& ret_segs, const int& point_id, Range dist_range,
		const float& thresh)
	{
		//index range
		if (dist_range.lower > dist_range.upper) std::swap(dist_range.lower, dist_range.upper);
		IndexRange ir = getDegreeIndex(dist_range);
		int pattern_offset = ir.lower, pattern_len = ir.upper - ir.lower + 1;
		std::vector<int> match_point_ids;
		matchPattern(match_point_ids, point_id, pattern_offset, pattern_len,
			mDegreeDataCuMem.data_d, mDistThreshes.size(), mPool.points.size(), thresh);
		extendPointsToSegments(ret_segs, match_point_ids, dist_range.upper);
	}

	void matchLatentPattern(std::vector<StreamlineSegment>& ret_segs, const int& point_id, const int& feature_level,
		const float& thresh)
	{
		START_TIMER;
		PRINT_TIME("Timing: Start matching latent features: \n");

		int num_points = mPool.points.size();
		int pattern_offset = 0, pattern_len = meta.latent_feature_dim;
		int query_id = getFeatureId(point_id, feature_level);
		std::vector<int> match_feature_ids;
		matchPattern(match_feature_ids, query_id, pattern_offset, pattern_len,
			mLatentDataCuMem.data_d, meta.latent_feature_dim, mLatentSegmentMap.size(), thresh);
		PRINT_TIME("Timing: Match latent features: %5.3f\n", mTimer.end());
		extendPointsToSegmentsForLatentFeatures(ret_segs, match_feature_ids);
		PRINT_TIME("Timing: Compute matched segments: %5.3f\n", mTimer.end());
		updateRendererWithMatchResults(ret_segs, mLatentSegmentMap[query_id]);
		PRINT_TIME("Timing: Update streamline renderer: %5.3f\n", mTimer.end());
		updateTsneGraphWithMatchResults(match_feature_ids, query_id);
		PRINT_TIME("Timing: Update t-SNE display: %5.3f\n\n", mTimer.end());
	}

	void matchPattern(std::vector<int>& ret_ids, const int& match_id,
		const int& pattern_offset, const int& pattern_len,
		float* feature_map_d, const int& feature_dim, const int& num_points, const float& thresh)
	{
		ret_ids.clear();
		float* pattern_d = feature_map_d + match_id * feature_dim + pattern_offset;
		bool* masks = new bool[num_points];

		matchPattern_h(masks, pattern_d, pattern_len, pattern_offset, thresh,
			feature_map_d, feature_dim, 16, num_points);

		//compact
		for (int i = 0; i < num_points; ++i) {
			if (masks[i]) ret_ids.push_back(i);
		}

		delete[] masks;
	}

	void findTsneIdsInDisplay(std::vector<int>& ret, const vec2f& p, const float& dist_thresh) {
		START_TIMER;
		PRINT_TIME("Timing: Start finding t-SNE points: \n");
		static cudaDeviceMem<float> p_cumem(2);
		p_cumem.load((float*)&p);
		int n = mPool.streamlines.size();
		
		std::vector<float> dist(n);
		findDistToPattern_h(dist.data(), p_cumem.data_d, 2, 0, dist_thresh, mTsneDisplayLayoutCuMem.data_d, 2, 4, n);
		PRINT_TIME("Timing: Match t-SNE points: %5.3f\n", mTimer.end());

		std::vector<int> all_ids(n);
		std::iota(all_ids.begin(), all_ids.end(), 0);
		thrust::sort_by_key(thrust::host, dist.data(), dist.data() + n, all_ids.data());

		float thresh_square = dist_thresh * dist_thresh;
		for (int i = 0; i < n; ++i) {
			if (dist[i] < thresh_square) {
				//if (tsneHaveLatent(all_ids[i])) {
					ret.push_back(all_ids[i]);
				//}
			}
			else {
				break;
			}
		}

 		PRINT_TIME("Timing: Sort t-SNE points: %5.3f\n\n", mTimer.end());
	}

	void updateDisplayWithTsneIds(const std::vector<int>& tsne_ids) {
		START_TIMER;
		PRINT_TIME("Timing: Start updating rendering with t-SNE points: \n");

		// std::vector<int> feature_ids;
		// feature_ids.reserve(mLatentTsneMap.size());
		// 根据tsne_ids，找到对应的latenMap，从而得到feature_ids
		/*for (const int& tid : tsne_ids) {
			feature_ids.insert(feature_ids.end(), mTsneLatentMap[tid].begin(), mTsneLatentMap[tid].end());
		}
		std::sort(feature_ids.begin(), feature_ids.end());
		PRINT_TIME("Timing: Finding feature ids: %5.3f\n", mTimer.end());*/

		// 直接根据ID号码找到对应的streamline的id号码，也就是找到对应的segs
		//std::vector<StreamlineSegment> segs;
		//extendPointsToSegmentsForLatentFeatures(segs, tsne_ids);
		//PRINT_TIME("Timing: Find matched segments: %5.3f\n", mTimer.end());
		// 这个好像就是点击了左边的图之后，根据feature_ids,找到对应的流线，并且改颜色。
		//updateRendererWithMatchResults(segs, makeStreamlineSegment(-1, -1, -1));
		//PRINT_TIME("Timing: Update streamline renderer: %5.3f\n", mTimer.end());
		// 这个是修改tsne点的颜色。
		updateTsneGraphWithMatchResults(tsne_ids, -1);
		PRINT_TIME("Timing: Update t-SNE display: %5.3f\n\n", mTimer.end());
	}

	void updateRendererWithMatchResults(const std::vector<StreamlineSegment>& match_segs,
		const StreamlineSegment& query_seg)
	{
		std::vector<unsigned char> color_ids(getNumPoints(), 0);
		for (int i = 0; i < match_segs.size(); ++i) {
			const StreamlineSegment& s = match_segs[i];
			int start = mPool.streamlines[s.streamline_id].start + s.segment.lower;
			int end = mPool.streamlines[s.streamline_id].start + s.segment.upper;
			std::fill(color_ids.begin() + start, color_ids.begin() + end, 1);
		}
		if (query_seg.streamline_id >= 0) {
			int start = mPool.streamlines[query_seg.streamline_id].start + query_seg.segment.lower;
			int end = mPool.streamlines[query_seg.streamline_id].start + query_seg.segment.upper;
			std::fill(color_ids.begin() + start, color_ids.begin() + end, 2);
		}

		std::vector<vec4f> color_map = { mContextColor, mMatchColor, mQueryColor };
		mRenderer->updateColor(color_ids, color_map);
	}

	void updateTsneGraphWithMatchResults(const std::vector<int>& match_ids, const int& query_id) {
		mTsneMatchIds.clear();
		std::vector<bool> tsne_masks(mTsneLayout.size(), false);
		for (const int& lid : match_ids) {
			mTsneMatchIds.push_back(lid);
			//tsne_masks[mLatentTsneMap[lid]] = true;
		}
		//for (int i = 0; i < tsne_masks.size(); ++i) {
			//if (tsne_masks[i]) mTsneMatchIds.push_back(i);
		//}
		mTsneQueryIds.clear();
		if (query_id >= 0) mTsneQueryIds.push_back(mLatentTsneMap[query_id]);
	}

	StreamlineSegment neighborhood(const VisGraphPointInfo& pinfo, const float& dist_thresh) {
		const Streamline& s = mPool.streamlines[pinfo.streamline_id];
		float* dist_vec = getDistanceVectorOfPoint(pinfo);

		StreamlineSegment seg = makeStreamlineSegment(pinfo.streamline_id, s.numPoint, -1);
		for (int j = 0; j < s.numPoint; ++j) {
			if (dist_vec[j] < dist_thresh) {
				if (j < seg.segment.lower) seg.segment.lower = j;
				if (j > seg.segment.upper) seg.segment.upper = j;
			}
		}

		return seg;
	}

	StreamlineSegment neighborhood(const VisGraphPointInfo& pinfo, const int& fixed_range) {
		const Streamline& s = mPool.streamlines[pinfo.streamline_id];
		StreamlineSegment seg = makeStreamlineSegment(pinfo.streamline_id,
			pinfo.point_local_id - fixed_range, pinfo.point_local_id + fixed_range);
		if (seg.segment.lower < 0) seg.segment.lower = 0;
		if (seg.segment.upper >= s.numPoint) seg.segment.upper = s.numPoint - 1;
		return seg;
	}

	StreamlineSegment neighborhoodLatent(const VisGraphPointInfo& pinfo, const int& feature_level) {
		int feature_id = getFeatureId(getGlobalPointId(pinfo.streamline_id, pinfo.point_local_id), feature_level);
		return neighborhoodLatent(feature_id);
	}

	StreamlineSegment neighborhoodLatent(const int& global_feature_id) {
		return mLatentSegmentMap[global_feature_id];
	}

	void extendPointsToSegments(std::vector<StreamlineSegment>& ret, const std::vector<int>& points, const float& dist_thresh) {
		ret.clear();
		for (int i = 0; i < points.size(); ++i) {
			VisGraphPointInfo pinfo = getPointInfo(points[i]);
			ret.push_back(neighborhood(pinfo, dist_thresh));
		}
		mergeSegments(ret);
	}

	void extendPointsToSegments(std::vector<StreamlineSegment>& ret, const std::vector<int>& points, const int& fixed_range) {
		ret.clear();
		for (int i = 0; i < points.size(); ++i) {
			VisGraphPointInfo pinfo = getPointInfo(points[i]);
			const Streamline& s = mPool.streamlines[pinfo.streamline_id];
			StreamlineSegment seg = neighborhood(pinfo, fixed_range);
			if (seg.segment.lower < seg.segment.upper) {
				ret.push_back(seg);
			}
		}
		mergeSegments(ret);
	}

	void extendPointsToSegmentsForLatentFeatures(std::vector<StreamlineSegment>& ret, const std::vector<int>& tsne_ids) {
		ret.clear();
		int num_feature_levels = meta.sample_steps.size();
		// 遍历tsne的ID，
		for (const int& fid : tsne_ids) {
			StreamlineSegment seg = mLatentSegmentMap[fid];
			if (seg.segment.lower < seg.segment.upper) {
				ret.push_back(seg);
			}
		}
		mergeSegments(ret);
	}

	void updateLatentDisplayData(const int& streamline_id) {
		const Streamline& s = mPool.streamlines[streamline_id];
		mLatentDisplayData.clear();
		if (mLatentDisplayMode == latent_raw_data) {
			int d = meta.latent_feature_dim;
			for (int i = s.start; i < s.start + s.numPoint; ++i) {
				for (int j = 0; j < meta.latent_feature_num; ++j) {
					float* data = &mLatentData[getFeatureId(i, j) * d];
					mLatentDisplayData.insert(mLatentDisplayData.end(), data, data + d);
				}
			}
		}
		else {
			for (int i = s.start; i < s.start + s.numPoint; ++i) {
				for (int j = 0; j < meta.latent_feature_num; ++j) {
					mLatentDisplayData.push_back(mTsneColorIds[mLatentTsneMap[getFeatureId(i, j)]]);
				}
			}
		}
	}

	void findTsneIdsInLatentSpace() {
		START_TIMER;
		PRINT_TIME("Timing: Find closest tSNE sample.");
		int num_latents = mLatentSegmentMap.size();
		int num_tsne_points = mTsneLayout.size();
		cudaDeviceMem<int> tsne_ids_cumem(num_latents);
		findClosestTemplate_h(tsne_ids_cumem.data_d, mTsneLatentCuMem.data_d, mLatentDataCuMem.data_d,
			num_tsne_points, num_latents, meta.latent_feature_dim, 1);
		mLatentTsneMap.resize(num_latents);
		tsne_ids_cumem.dump(mLatentTsneMap.data());
		PRINT_TIME(" Finish in %5.3f.\n\n", mTimer.end());
		write_array(mLatentTsneMap.data(), mLatentTsneMap.size(), meta.tsne_id_path.c_str());
	}

	bool tsneHaveLatent(const int& tsne_id) {
		return !mTsneLatentMap[tsne_id].empty();
	}

	void updateTsneColorId(const TsneColorMode& color_mode) {
		std::vector<float> candidate_ids = { 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };
		//int n = mTsneLayout.size();
		int n = mPool.streamlines.size();
		int num_colors = candidate_ids.size();
		if (color_mode == tsne_dataset_color) {
			mTsneColorIds.reserve(n);
			mTsneColorIds.clear();
			for (int i = 0; i < 7; ++i) {
				mTsneColorIds.insert(mTsneColorIds.end(), n / 7, candidate_ids[i]);
			}
			for (int i = 0; i < n; ++i) {
				if (!tsneHaveLatent(i)) {
					mTsneColorIds[i] = 7;//gray
				}
			}
		}
		else {
			mTsneColorIds.resize(n);
			const std::vector<int>& dbscan_label = mDBScan.getLabels();
			for (int i = 0; i < n; ++i) {
				//if (dbscan_label[i] == -1 || !tsneHaveLatent(i)) {
				if (dbscan_label[i] == -1 ) {
					mTsneColorIds[i] = 7;//gray
				}
				else if (dbscan_label[i] < num_colors) {
					mTsneColorIds[i] = candidate_ids[dbscan_label[i]];
				}
				else {
					mTsneColorIds[i] = candidate_ids[dbscan_label[i] % num_colors];
				}
			}
		}
	}

	void updateTsneDisplayLayout(const std::vector<vec2f>& display_positions) {
		mTsneDisplayLayoutCuMem.allocate(display_positions.size() * 2);
		mTsneDisplayLayoutCuMem.load((float*)display_positions.data());
	}

	inline int getFeatureId(const int& point_global_id, const int& feature_level) {
		return mPointLatentLookup[feature_level * getNumPoints() + point_global_id];
	}

	inline int getGlobalPointId(const int& streamline_id, const int& point_id) {
		return (mPool.streamlines[streamline_id].start + point_id);
	}

	MatrixData<float>* getDegreeMapOfLine(const int& streamline_id) {
		return &(getVisGraphOfStreamline(streamline_id)->degree_mat);
	}

	float* getDegreeVectorOfPoint(const int& point_id) {
		VisGraphPointInfo pinfo = getPointInfo(point_id);
		float** degree_map = getDegreeMapOfLine(pinfo.streamline_id)->getMatrixPointer();
		return degree_map[pinfo.point_local_id];
	}

	MatrixData<float>* getDistanceMatrixOfLine(const int& streamline_id) {
		return &(getVisGraphOfStreamline(streamline_id)->adj_mat);
	}

	float* getDistanceVectorOfPoint(const VisGraphPointInfo& pinfo) {
		float** dist_mat = getDistanceMatrixOfLine(pinfo.streamline_id)->getMatrixPointer();
		return dist_mat[pinfo.point_local_id];
	}

	float* getDistanceVectorOfPoint(const int& point_id) {
		VisGraphPointInfo pinfo = getPointInfo(point_id);
		return getDistanceVectorOfPoint(pinfo);
	}

	Streamline getStreamlineOfPoint(const int& point_id) {
		return mPool.streamlines[getStreamlineIDOfPoint(point_id)];
	}

	int getStreamlineIDOfPoint(const int& point_id) {
		return mPool.line_ids[point_id];
	}

	VisGraphPointInfo getPointInfo(const int& pid) {
		VisGraphPointInfo ret;
		ret.streamline_id = mPool.line_ids[pid];
		ret.point_local_id = pid - mPool.streamlines[ret.streamline_id].start;
		return ret;
	}

	VisibilityGraph* getVisGraphOfStreamline(const int& streamline_id) {
		return mGraphs[streamline_id];
	}

	vec3f* getDevicePoints() {
		return mRenderer->getPoints_d();
	}

	Streamline* getDeviceStreamlines() {
		return mRenderer->getStreamlines_d();
	}

	vec3f getPoint(const int& point_id) {
		return mPool.points[point_id];
	}

	int getNumPoints() {
		return mPool.points.size();
	}

	int getNumStreamlines() {
		return mPool.streamlines.size();
	}

	void getNormalCurveAtPoint(std::vector<vec3f>& ret_curve, const int& point_id) {
		int n = meta.local_sample_num;
		ret_curve.assign(&mNormalCurves[point_id * n], &mNormalCurves[point_id * n + n]);
	}

	void getBinormalCurveAtPoint(std::vector<vec3f>& ret_curve, const int& point_id) {
		int n = meta.local_sample_num;
		ret_curve.assign(&mBinormalCurves[point_id * n], &mBinormalCurves[point_id * n + n]);
	}

	cudaStreamlineTracer* getTracer() {
		if (mTracer == NULL) {
			int size = mDim.x * mDim.y * mDim.z;
			std::vector<vec3f> vec_field(size);
			read_array<vec3f>(vec_field.data(), size, meta.vec_field_path.c_str());
			mTracer = new cudaStreamlineTracer(vec_field.data(), mDim.x, mDim.y, mDim.z);
		}
		return mTracer;
	}

	inline StreamlineTraceParameter getTracingParameters() {
		StreamlineTraceParameter pars;
		pars.max_point = meta.max_streamline_length;
		pars.min_point = meta.local_sample_num * meta.sample_steps.back();
		pars.segment_length = meta.segment_length;
		pars.max_streamline = meta.num_streamlines * 10;
		pars.store_gap = meta.store_gap_streamline;
		return pars;
	}

	Range getLatentFeatureRange() { return mLatentRange; }

	int getNumFeatures() { return meta.latent_feature_num; }
	int getLatentFeatureDim() { return meta.latent_feature_dim; }
	int getDisplayLatentFeatureDim() {
		return (mLatentDisplayMode == latent_raw_data) ? meta.latent_feature_dim : 1;
	}

	int getSampleSize() {
		return meta.local_sample_num;
	}

	void filterMatrix(MatrixData<float>& submat, const float& filter_thresh) {
		int n = submat.rowSize();
		for (int i = 0; i < n; ++i) {
			for (int j = i + 1; j < n; ++j) {
				if (submat[i][j] > filter_thresh) {
					submat[j][i] = submat[i][j] = 1e30;
				}
			}
		}
		for (int k = 0; k < n; ++k) {
			for (int i = 0; i < n; ++i) {
				float ik = submat[k][i];
				for (int j = i + 1; j < n; ++j) {
					float ikj = ik + submat[k][j];
					if (ikj < submat[i][j]) {
						submat[i][j] = submat[j][i] = ikj;
					}
				}
			}
		}
	}

	StreamlineSegment getSampleMatrix(MatrixData<float>& submat,
		const int& streamline_id, const int& point_id, int feature_id)
	{
		int global_point_id = getGlobalPointId(streamline_id, point_id);
		StreamlineSegment ret;
		if (feature_id < meta.sample_steps_num) {//sample steps
			return getSampleMatrix(submat, global_point_id, meta.sample_steps[feature_id]);
		}
		else if ((feature_id -= meta.sample_steps_num) < meta.filter_scales_num) {//filters
			ret = getSampleMatrix(submat, global_point_id, 1.0f);
			filterMatrix(submat, meta.filter_scales[feature_id] * meta.segment_length);
		}
		else if ((feature_id -= meta.filter_scales_num) < meta.resample_scales_num) {//resample arc-length
			auto& pool = *mResamplePools.getResamplePool(meta.resample_scales[feature_id] * meta.segment_length, StreamlineResample::arc_length);
			int resample_point_id = pool.getPointGlobalId(pool.mapToResample(streamline_id, point_id), streamline_id);
			return getSampleMatrix(submat, pool, resample_point_id);
		}
		else if (toSampleCurvature()) {//resample curvature
			auto& pool = *mResamplePools.getResamplePool(meta.curvature_sample_rate, StreamlineResample::acc_attrib);
			int resample_point_id = pool.getPointGlobalId(pool.mapToResample(streamline_id, point_id), streamline_id);
			return getSampleMatrix(submat, pool, resample_point_id);
		}
		return ret;
	}

	StreamlineSegment getSampleMatrix(MatrixData<float>& submat, const int& global_point_id, float step_size) {
		static float hs = 0.5f * (meta.local_sample_num - 1);

		VisGraphPointInfo pinfo = getPointInfo(global_point_id);
		MatrixData<float>* mat = getDistanceMatrixOfLine(pinfo.streamline_id);
		int n = mat->width();
		Range bound = makeRange(0.0f, n - 1.0f);
		float half_size = hs * step_size;
		Range r;
		if (2.0f * half_size > bound.upper) {
			r = bound;
			step_size = bound.upper / (meta.local_sample_num - 1);
		}
		else {
			r = makeRange(pinfo.point_local_id - half_size, pinfo.point_local_id + half_size);
			r = moveRangeInBound(r, bound);
		}
		mat->submatrix(submat, makeVec2f(r.lower), makeVec2f(step_size));
		float* submat_data = submat.getData();
		float scale_fac = 1.0f / (meta.segment_length * (r.upper - r.lower) / meta.local_sample_num);
		for (int i = 0; i < submat.MatrixSize(); ++i) {
			submat_data[i] *= scale_fac;
		}
		return makeStreamlineSegment(pinfo.streamline_id, r.lower, r.upper);
	}

	StreamlineSegment getSampleMatrix(MatrixData<float>& submat, StreamlineResample::Pool<float>& resample_pool,
		const int& resample_id)
	{
		static int hs = meta.local_sample_num / 2;

		VisGraphPointInfo pinfo = resample_pool.getPointInfo(resample_id);
		MatrixData<float>* mat = getDistanceMatrixOfLine(pinfo.streamline_id);
		const Streamline& rs = resample_pool.streamlines[pinfo.streamline_id];

		StreamlineSegment ret_seg;
		if (rs.numPoint < meta.local_sample_num) {
			const Streamline& s = mPool.streamlines[pinfo.streamline_id];
			float step_size = (s.numPoint - 1) / (float)(meta.local_sample_num - 1);
			mat->submatrix(submat, makeVec2f(0.0f), makeVec2f(step_size));
			ret_seg = makeStreamlineSegment(pinfo.streamline_id, 0, s.numPoint - 1);
		}
		else {
			IndexRange bound = makeIndexRange(0, rs.numPoint);
			IndexRange r = makeIndexRange(pinfo.point_local_id - hs, pinfo.point_local_id + hs);
			r = moveRangeInBound(r, bound);
			std::vector<float> samples;
			samples.assign(resample_pool.points.begin() + rs.start + r.lower, resample_pool.points.begin() + rs.start + r.upper);
			mat->submatrix(submat, samples);
			ret_seg = resample_pool.mapToOriginal(makeStreamlineSegment(pinfo.streamline_id, r.lower, r.upper - 1));
		}
		float* submat_data = submat.getData();
		float scale_fac = 1.0f / (meta.segment_length * (ret_seg.segment.upper - ret_seg.segment.lower) / meta.local_sample_num);
		for (int i = 0; i < submat.MatrixSize(); ++i) {
			submat_data[i] *= scale_fac;
		}
		return ret_seg;
	}

	void getTsneSampleMatrix(MatrixData<float>& mat, const int& tsne_id) {
		int n = meta.local_sample_num;
		memcpy(mat.getData(), mTsneVisGraphData.data() + tsne_id * n * n, n * n * sizeof(float));
	}

	void featureTransform(float& feature) {
		feature = sqrtf(feature) * meta.normalize_factor;
	}

	void reverseFeatureTransform(float& feature) {
		feature *= meta.reverse_normalize_factor;
		feature *= feature;
	}

	void writeAllVisGraphFile() {
		START_TIMER;
		PRINT_TIME("Timing: generating all vis graph for streamline segments.");
		size_t mat_size = meta.local_sample_num; //32
		size_t mat_size_sq = mat_size * mat_size;  //32*32
		std::vector<float> all_vg;
		//7*点的个数*32*32
		all_vg.reserve(meta.latent_feature_num * getNumPoints() * mat_size_sq);
		MatrixData<float> submat(mat_size, mat_size);
		mLatentSegmentMap.reserve(meta.latent_feature_num * getNumPoints());
		mLatentSegmentMap.clear();
		mPointLatentLookup.reserve(meta.latent_feature_num * getNumPoints());
		mPointLatentLookup.clear();
		float half_size_num = 0.5f * (meta.local_sample_num - 1);

		//sample steps
		//采样
		for (float& sample_step : meta.sample_steps) {
			int half_size = half_size_num * sample_step;
			std::vector<int> samples;
			std::vector<int> id_lookup;
			getSamples(samples, id_lookup, mLatentSegmentMap.size(), half_size, mPool);
			for (int sample : samples) {
				mLatentSegmentMap.push_back(getSampleMatrix(submat, sample, sample_step));
				all_vg.insert(all_vg.end(), submat.getData(), submat.getData() + mat_size_sq);
			}
			mPointLatentLookup.insert(mPointLatentLookup.end(), id_lookup.begin(), id_lookup.end());
		}

		int half_size = half_size_num;
		//filtered distance matrix
		std::vector<float> dist_data(mDistData);
		for (float& filter_scale : meta.filter_scales) {
			//filter graph
			float filter_dist = filter_scale * meta.segment_length;
			filterVisibilityGraph(mDistData, filter_dist);
			//sample
			std::vector<int> samples;
			std::vector<int> id_lookup;
			getSamples(samples, id_lookup, mLatentSegmentMap.size(), half_size, mPool);
			for (int sample : samples) {
				mLatentSegmentMap.push_back(getSampleMatrix(submat, sample, 1.0f));
				all_vg.insert(all_vg.end(), submat.getData(), submat.getData() + mat_size_sq);
			}
			mPointLatentLookup.insert(mPointLatentLookup.end(), id_lookup.begin(), id_lookup.end());
			//recover visgraphs
			mDistData.assign(dist_data.begin(), dist_data.end());
		}
		dist_data.clear();

		//arc-length resample pools
		//弧长重采样
		for (float resample_scale : meta.resample_scales) {
			float resample_len = resample_scale * meta.segment_length;
			auto& resample_pool = *(mResamplePools.getResamplePool(resample_len, StreamlineResample::arc_length));
			std::vector<int> samples;
			std::vector<int> id_lookup;
			getSamples(samples, id_lookup, mLatentSegmentMap.size(), half_size, resample_pool);
			for (int sample : samples) {
				mLatentSegmentMap.push_back(getSampleMatrix(submat, resample_pool, sample));
				all_vg.insert(all_vg.end(), submat.getData(), submat.getData() + mat_size_sq);
			}
			mPointLatentLookup.insert(mPointLatentLookup.end(), id_lookup.begin(), id_lookup.end());
		}

		//curvature resample
		//曲率重采样
		if (toSampleCurvature()) {
			auto& resample_pool = *(mResamplePools.getResamplePool(meta.curvature_sample_rate, StreamlineResample::acc_attrib));
			std::vector<int> samples;
			std::vector<int> id_lookup;
			getSamples(samples, id_lookup, mLatentSegmentMap.size(), half_size, resample_pool);
			for (int sample : samples) {
				mLatentSegmentMap.push_back(getSampleMatrix(submat, resample_pool, sample));
				all_vg.insert(all_vg.end(), submat.getData(), submat.getData() + mat_size_sq);
			}
			mPointLatentLookup.insert(mPointLatentLookup.end(), id_lookup.begin(), id_lookup.end());
		}

		for (float& s : all_vg) featureTransform(s);
		//将所有的数据写到visgragh-all.dat
		write_array(all_vg.data(), all_vg.size(), meta.all_vg_path.c_str());
		//将曲率采样数据写到latent-segment.dat
		write_array(mLatentSegmentMap.data(), mLatentSegmentMap.size(), meta.latent_segment_map_path.c_str());
		//将弧长采样数据写到point-latent.dat
		write_array(mPointLatentLookup.data(), mPointLatentLookup.size(), meta.point_latent_map_path.c_str());

		std::vector<int> all_ids(mLatentSegmentMap.size());
		std::iota(all_ids.begin(), all_ids.end(), 0);
		RandomElements<int> sample_ids(all_ids, meta.num_random_samples);
		std::vector<float> sample_vg;
		sample_vg.reserve(sample_ids.samples.size() * mat_size_sq);
		for (int& s : sample_ids.samples) {
			float* data = &all_vg[s * mat_size_sq];
			sample_vg.insert(sample_vg.end(), data, data + mat_size_sq);
		}
		PRINT_TIME(" Finish in %5.3f.\n\n", mTimer.end());
		write_array(sample_vg.data(), sample_vg.size(), meta.sample_vg_path.c_str());
	}

	void writeSampleVisGraphFile() {
		std::vector<float> all_vg;
		//读取visgraph-all.dat的数据
		read_array(all_vg, meta.all_vg_path.c_str());
		//每个采样visgraph的大小为mat_size_sq=32*32
		size_t mat_size_sq = meta.local_sample_num * meta.local_sample_num;
		//all_ids的个数为all_vg.size()/mat_size_sq，所有图的大小之和除以每个图的大小
		std::vector<int> all_ids(all_vg.size() / mat_size_sq);
		std::iota(all_ids.begin(), all_ids.end(), 0);
		//采样数量num_random_samples=10000个
		RandomElements<int> sample_ids(all_ids, meta.num_random_samples);
		std::vector<float> sample_vg;
		//采样visgraph数量的大小，每个32*32
		sample_vg.reserve(sample_ids.samples.size() * mat_size_sq);
		for (int& s : sample_ids.samples) {
			float* data = &all_vg[s * mat_size_sq];
			sample_vg.insert(sample_vg.end(), data, data + mat_size_sq);
		}
		//将数据写到plume-visgraph-sample.dat
		write_array(sample_vg.data(), sample_vg.size(), meta.sample_vg_path.c_str());
	}

	template <typename PoolType>
	void getSamples(std::vector<int>& samples, std::vector<int>& id_lookup, const int& id_offset,
		const int& half_size, PoolType& pool)
	{
		samples.reserve(getNumPoints());
		id_lookup.reserve(getNumPoints());
		for (Streamline& s : pool.streamlines) {
			if (s.numPoint <= 2 * half_size) {
				id_lookup.insert(id_lookup.end(), s.numPoint, samples.size() + id_offset);
				samples.push_back(s.start);
			}
			else {
				id_lookup.insert(id_lookup.end(), half_size, samples.size() + id_offset);
				for (int j = half_size; j < s.numPoint - half_size; ++j) {
					id_lookup.push_back(samples.size() + id_offset);
					samples.push_back(j + s.start);
				}
				id_lookup.insert(id_lookup.end(), half_size, samples.size() - 1 + id_offset);
			}
		}
	}

	void getSamples(std::vector<int>& samples, std::vector<int>& id_lookup, const int& id_offset,
		const int& half_size, StreamlineResample::Pool<float>& pool)
	{
		samples.reserve(getNumPoints());
		id_lookup.reserve(getNumPoints());
		for (int i = 0; i < pool.streamlines.size(); ++i) {
			const Streamline& s = pool.streamlines[i];

			if (s.numPoint <= 2 * half_size) {
				id_lookup.insert(id_lookup.end(), mPool.streamlines[i].numPoint, samples.size() + id_offset);
				samples.push_back(s.start);
			}
			else {
				std::vector<int> resample_id;
				pool.mapToResample(resample_id, i);
				for (int j = 0; j < resample_id.size(); ++j) {
					if (resample_id[j] >= s.numPoint - half_size && !samples.empty()) {
						id_lookup.push_back(samples.size() - 1 + id_offset);
					}
					else {
						id_lookup.push_back(samples.size() + id_offset);
						if (resample_id[j] >= half_size && resample_id[j] != resample_id[j - 1])
						{
							samples.push_back(resample_id[j] + s.start);
						}
					}
				}
			}
		}
	}

	// 何颂贤把KunhuaDistance复制过来用了，
	//每条流线采样相同的点数sample_numpoint
	void getEqualPointStreamlinePool(StreamlinePool& Pool, const int& sample_numpoint) {//均匀采样
		StreamlinePool tempPool = Pool;
		Pool.points.clear();
		Pool.streamlines.clear();
		Pool.line_ids.clear();
		Pool.streamlines.reserve(tempPool.streamlines.size());
		for (int i = 0; i < tempPool.streamlines.size(); ++i) {
			const Streamline& s = tempPool.streamlines[i];
			Streamline new_line = makeStreamline(0, Pool.points.size(), 0);
			//采样相同的点数
			float interval = s.numPoint * 1.0 / sample_numpoint;
			//vec3f first_point = tempPool.points[s.start];
			//Pool.points.push_back(first_point);
			for (int j = 0; j < sample_numpoint; ++j) {
				if (j != sample_numpoint - 1) {
					int position = floor(s.start + j * interval);
					Pool.points.push_back(tempPool.points[position]);
				}
				else {
					Pool.points.push_back(tempPool.points[s.start + s.numPoint - 1]);
				}
			}
			//流线上点的数量
			new_line.numPoint = Pool.points.size() - new_line.start;
			//每个点所属的流线id
			Pool.line_ids.insert(Pool.line_ids.end(), new_line.numPoint, i);
			//将新构造的流线加入流线池
			Pool.streamlines.push_back(new_line);
		}
		Pool.fillLineIds();
	}

	void resample(StreamlinePool& Pool, const float& thresh) {//等距离采样
		StreamlinePool tempPool = Pool;
		Pool.points.clear();
		Pool.streamlines.clear();
		Pool.line_ids.clear();
		Pool.streamlines.reserve(tempPool.streamlines.size());
		for (int i = 0; i < tempPool.streamlines.size(); ++i) {
			const Streamline& s = tempPool.streamlines[i];
			Streamline resample_line = makeStreamline(0, Pool.points.size(), 0);
			vec3f* line_points = tempPool.points.data() + s.start; //line_points指向tempPool的流线s的起始位置
			arc_resample(line_points, s.numPoint, thresh, Pool.points);
			//计算当前流线上的点的个数
			resample_line.numPoint = Pool.points.size() - resample_line.start;
			//保存流线上的点的line_ids
			Pool.line_ids.insert(Pool.line_ids.end(), resample_line.numPoint, i);
			//将流线加入采样流线池
			Pool.streamlines.push_back(resample_line);
		}
		//Pool.fillLineIds();
	}
	void arc_resample(vec3f* points, const int& num, const float& thresh, std::vector<vec3f>& sample_points) {
		std::vector<float> line_resample_indices;
		std::vector<vec3f>& line_resample_points = sample_points;
		line_resample_indices.reserve(num);
		line_resample_indices.push_back(0.0f);
		vec3f p = points[0];

		//line_resample_points.push_back(points[0]);

		float di1 = 0.0f, di, fac;
		for (int i = 1; i < num; ++i) {
			di = length(p - points[i]);
			if (di > thresh) {
				fac = interpolate(di1, di, thresh, 0.0f, 1.0f);
				p = interpolate(points[i - 1], points[i], fac);
				//line_resample_points.push_back(p);
				line_resample_indices.push_back(i - 1 + fac);
				di1 = 0.0f;
			}
			else {
				di1 = di;
			}
		}
		for (int i = 0; i < line_resample_indices.size(); ++i) {
			float resample_index = line_resample_indices[i];
			int int_index = (int)resample_index;
			if (int_index == num - 1) {
				line_resample_points.push_back(points[num - 1]);
			}
			else {
				float fac = resample_index - int_index;
				line_resample_points.push_back(interpolate(points[int_index], points[int_index + 1], fac));
			}
		}
	}

	//欧氏距离
	void compute_dE(StreamlinePool Pool, std::vector<float>& distance_of_dE) {
		for (int i = 0; i < Pool.streamlines.size(); ++i) {
			Streamline currentLine = Pool.streamlines[i];
			for (int j = 0; j < Pool.streamlines.size(); ++j) {
				Streamline computeLine = Pool.streamlines[j];
				float distance = 0.0f;
				for (int r = currentLine.start, k = computeLine.start; r < currentLine.start + currentLine.numPoint, k < computeLine.start + computeLine.numPoint; ++r, ++k) {
					//distance += sqrt((Pool.points[r].x - Pool.points[k].x) * (Pool.points[r].x - Pool.points[k].x) + (Pool.points[r].y - Pool.points[k].y) * (Pool.points[r].y - Pool.points[k].y) + (Pool.points[r].z - Pool.points[k].z) * (Pool.points[r].z - Pool.points[k].z));
					distance += dist3d(Pool.points[r], Pool.points[k]);
				}
				printf("dE: %d/%d complete\n", i * Pool.streamlines.size() + j + 1, Pool.streamlines.size() * Pool.streamlines.size());
				distance_of_dE.push_back(distance / currentLine.numPoint);
			}
		}
	}
	//几何相似性度量
	void compute_dG(StreamlinePool Pool, std::vector<float>& distance_of_dG) {
		for (int i = 0; i < Pool.streamlines.size(); ++i) {
			Streamline currentLine = Pool.streamlines[i];
			for (int j = 0; j < Pool.streamlines.size(); ++j) {
				Streamline computeLine = Pool.streamlines[j];
				float geometric_distance = 0.0f;
				float parallelism = 0.0f;
				for (int r = currentLine.start + 1, k = computeLine.start + 1; r < currentLine.start + currentLine.numPoint, k < computeLine.start + computeLine.numPoint; ++r, ++k) {
					float segmentX_i = Pool.points[r].x - Pool.points[r - 1].x;
					float segmentY_i = Pool.points[r].y - Pool.points[r - 1].y;
					float segmentZ_i = Pool.points[r].z - Pool.points[r - 1].z;
					float segmentX_j = Pool.points[k].x - Pool.points[k - 1].x;
					float segmentY_j = Pool.points[k].y - Pool.points[k - 1].y;
					float segmentZ_j = Pool.points[k].z - Pool.points[k - 1].z;
					float theta = (segmentX_i * segmentX_j + segmentY_i * segmentY_j + segmentZ_i * segmentZ_j) / (sqrt(segmentX_i * segmentX_i + segmentY_i * segmentY_i + segmentZ_i * segmentZ_i) * sqrt(segmentX_j * segmentX_j + segmentY_j * segmentY_j + segmentZ_j * segmentZ_j));
					parallelism += acos(theta);
				}
				if (isnan(parallelism)) {
					parallelism = 0.0f;
				}
				geometric_distance = parallelism / (currentLine.numPoint - 1);
				printf("dG: %d/%d complete\n", i * Pool.streamlines.size() + j + 1, Pool.streamlines.size() * Pool.streamlines.size());
				distance_of_dG.push_back(geometric_distance);
			}
		}
	}
	//累积旋转差异
	void computer_dR(StreamlinePool Pool, std::vector<float>& distance_of_dR) {

	}
	//Mean-of-closest-point(MCP)
	void compute_dM(StreamlinePool Pool, std::vector<float>& distance_of_dM) {
		for (int i = 0; i < Pool.streamlines.size(); ++i) {
			Streamline currentLine = Pool.streamlines[i];
			for (int j = 0; j < Pool.streamlines.size(); ++j) {
				Streamline computeLine = Pool.streamlines[j];
				float distance1 = mean_closest_point_distance(Pool, i, j);
				float distance2 = mean_closest_point_distance(Pool, j, i);
				//两个平均值之和的平均值
				distance_of_dM.push_back((distance1 + distance2) / 2.0f);
				//printf("dM: %d/%d complete\n", i * Pool.streamlines.size() + j + 1, Pool.streamlines.size() * Pool.streamlines.size());
			}
		}
	}
	float mean_closest_point_distance(StreamlinePool Pool, int i, int j) {
		//所有最近点距离的平均值
		float mean = 0.0f;
		Streamline s1 = Pool.streamlines[i];
		Streamline s2 = Pool.streamlines[j];
		for (int i = s1.start; i < s1.start + s1.numPoint; ++i) {
			//当前点到另外一条流线的最近点距离
			float min_distance = FLT_MAX;
			for (int j = s2.start; j < s2.start + s2.numPoint; ++j) {
				float distance = sqrt((Pool.points[i].x - Pool.points[j].x) * (Pool.points[i].x - Pool.points[j].x) + (Pool.points[i].y - Pool.points[j].y) * (Pool.points[i].y - Pool.points[j].y) + (Pool.points[i].z - Pool.points[j].z) * (Pool.points[i].z - Pool.points[j].z));
				if (distance < min_distance) {
					min_distance = distance;
				}
			}
			mean += min_distance;
		}
		return mean / s1.numPoint;
	}
	//Hausdorff distance
	void compute_dH(StreamlinePool Pool, std::vector<float>& distance_of_dH) {
		for (int i = 0; i < Pool.streamlines.size(); ++i) {
			Streamline currentLine = Pool.streamlines[i];
			for (int j = 0; j < Pool.streamlines.size(); ++j) {
				Streamline computeLine = Pool.streamlines[j];
				float distance1 = h_distance(Pool, i, j);
				float distance2 = h_distance(Pool, j, i);
				//取两者之间的最大值
				if (distance1 > distance2) {
					distance_of_dH.push_back(distance1);
				}
				else {
					distance_of_dH.push_back(distance2);
				}
				printf("dH: %d/%d complete\n", i * Pool.streamlines.size() + j + 1, Pool.streamlines.size() * Pool.streamlines.size());
			}
		}
	}
	float h_distance(StreamlinePool Pool, int i, int j) {
		Streamline s1 = Pool.streamlines[i];
		Streamline s2 = Pool.streamlines[j];
		//所有最近点距离中的最大值
		float max_distance = 0.0f;
		for (int i = s1.start; i < s1.start + s1.numPoint; ++i) {
			//当前点到另外一条流线的最近点距离
			float min_distance = FLT_MAX;
			for (int j = s2.start; j < s2.start + s2.numPoint; ++j) {
				float distance = sqrt((Pool.points[i].x - Pool.points[j].x) * (Pool.points[i].x - Pool.points[j].x) + (Pool.points[i].y - Pool.points[j].y) * (Pool.points[i].y - Pool.points[j].y) + (Pool.points[i].z - Pool.points[j].z) * (Pool.points[i].z - Pool.points[j].z));
				if (distance < min_distance) {
					min_distance = distance;
				}
			}
			if (min_distance > max_distance) {
				max_distance = min_distance;
			}
		}
		return max_distance;
	}

	// Endpoints distance
	void compute_dEP(StreamlinePool Pool, std::vector<float>& distance_of_dEP) {
		for (int i = 0; i < Pool.streamlines.size(); ++i) {
			Streamline currentLine = Pool.streamlines[i];
			for (int j = 0; j < Pool.streamlines.size(); ++j) {
				Streamline computeLine = Pool.streamlines[j];
				float distance = 0.0f;
				float distance1 = dist3d(Pool.points[currentLine.start], Pool.points[computeLine.start]);
				float distance2 = dist3d(Pool.points[currentLine.start + currentLine.numPoint - 1], Pool.points[computeLine.start + computeLine.numPoint - 1]);
				distance = (distance1 + distance2) / 2.0f;
				distance_of_dEP.push_back(distance);
			}
		}
	}

	//Adapted Procrustes distance
	void compute_dP(StreamlinePool Pool, std::vector<float>& distance_of_dP) {
		for (int i = 0; i < Pool.streamlines.size(); ++i) {
			Streamline currentLine = Pool.streamlines[i];
			for (int j = 0; j < Pool.streamlines.size(); ++j) {
				Streamline computeLine = Pool.streamlines[j];
				if (i == 109 && j == 480) {
					printf("Reach!\n");
				}
				float Procrustes_distance = computeProcrustesDistanceWithoutOrder(Pool.points.data() + currentLine.start, Pool.points.data() + computeLine.start, currentLine.numPoint, true);
				printf("dP: %d/%d complete\n", i * Pool.streamlines.size() + j + 1, Pool.streamlines.size() * Pool.streamlines.size());
				distance_of_dP.push_back(Procrustes_distance / currentLine.numPoint);
			}
		}
	}

	//Frechet distance
	float discreteFrechetGetCa(vec3f* p1, vec3f* p2, const int& i, const int& j, float** ca) {
		if (ca[i][j] > FRECHET_INIT_THRESH) {
			return ca[i][j];
		}
		else if (i == 0 && j == 0) {
			ca[i][j] = dist3d(p1[i], p2[j]);
		}
		else if (i > 0 && j == 0) {
			float c = discreteFrechetGetCa(p1, p2, i - 1, 0, ca);
			float d = dist3d(p1[i], p2[j]);
			ca[i][j] = (c > d) ? c : d;
		}
		else if (i == 0 && j > 0) {
			float c = discreteFrechetGetCa(p1, p2, 0, j - 1, ca);
			float d = dist3d(p1[i], p2[j]);
			ca[i][j] = (c > d) ? c : d;
		}
		else if (i > 0 && j > 0) {
			float c1 = discreteFrechetGetCa(p1, p2, i - 1, j, ca);
			float c2 = discreteFrechetGetCa(p1, p2, i, j - 1, ca);
			float c3 = discreteFrechetGetCa(p1, p2, i - 1, j - 1, ca);
			float d = dist3d(p1[i], p2[j]);
			if (c1 > d) d = c1;
			if (c2 > d) d = c2;
			if (c3 > d) d = c3;
			ca[i][j] = d;
		}
		else {
			ca[i][j] = FRECHET_INFINITY;
		}
		return ca[i][j];
	}

	float discreteFrechetDistance(vec3f* p1, vec3f* p2, const int& n1, const int& n2, float** ca) {
		for (int i = 0; i < n1; ++i) {
			for (int j = 0; j < n2; ++j) {
				ca[i][j] = FRECHET_NOT_INIT;
			}
		}

		return discreteFrechetGetCa(p1, p2, n1 - 1, n2 - 1, ca);
	}

	float* genDiscreteFrechetDistanceMatrix(Streamline* stls, const int& num_stls, vec3f* points) {
		int num_point = stls[num_stls - 1].start + stls[num_stls - 1].numPoint;
		int max_point_stl = stls[0].numPoint;
		for (int i = 1; i < num_stls; ++i) {
			if (stls[i].numPoint > max_point_stl) {
				max_point_stl = stls[i].numPoint;
			}
		}

		float* ret_mat_data, ** ret_mat;
		float* ca_data, ** ca;
		allocateMatrix(ret_mat_data, ret_mat, num_stls, num_stls);
		allocateMatrix(ca_data, ca, max_point_stl, max_point_stl);

		int num_pair = (num_stls) * (num_stls - 1) >> 1, count = 0;

		vec3f* p1, * p2;
		int n1, n2;

		for (int i = 0; i < num_stls; ++i) {
			p1 = &points[stls[i].start];
			n1 = stls[i].numPoint;
			ret_mat[i][i] = 0.0f;

			for (int j = i + 1; j < num_stls; ++j) {
				p2 = &points[stls[j].start];
				n2 = stls[j].numPoint;

				ret_mat[i][j] = ret_mat[j][i] = discreteFrechetDistance(p1, p2, n1, n2, ca);

				++count;
				printf("\rDistance Matrix Computation: %d/%d", count, num_pair);
			}
		}
		printf("\n");

		delete[] ret_mat;
		delete[] ca;
		delete[] ca_data;

		return ret_mat_data;
	}


#ifdef TIMING_RESULT
	WindowsTimer mTimer;
	WindowsTimer nTimer;
#endif

	VisGraphMetaInfo meta;

	std::string file_path = "D:/data/project_data/VisibilityGraph/result/";

	std::vector<std::string> datasets = { "abc", "bernard", "brain", "computer_room","crayfish", "cylinder","electro3D","vessel",
			"plume","random-5cp","two_swirl","tornado" };

	
	std::string dataset = "brain";
	vec3i mDim;
	StreamlinePool mPool;
	
	std::vector<std::vector<float>> mKunhuaDisMats;
	std::vector<float> mDistData;
	std::vector<float> mKunHuaDisMat;
	std::vector<float> mKunHuaDisMat_row;

	std::vector<float> mCurvatures;
	cudaStreamlineRenderer* mRenderer;
	cudaStreamlineTracer* mTracer;
	StreamlineResample::MultiPool<float> mResamplePools;

	std::vector<float> mDistThreshes;
	std::vector<float> mDegreeData;
	cudaDeviceMem<float> mDegreeDataCuMem;
	std::vector<float> mDegreeAvg;
	std::vector<float> mDegreeVar;
	std::vector<float> mLatentData;
	cudaDeviceMem<float> mLatentDataCuMem;
	Range mLatentRange;
	std::vector<StreamlineSegment> mLatentSegmentMap;
	std::vector<int> mLatentFeatureLevels;
	std::vector<int> mPointLatentLookup;
	std::vector<float> mLatentDisplayData;
	LatentDisplayMode mLatentDisplayMode;
	std::vector<vec2f> mTsneLayout;
	cudaDeviceMem<float> mTsneDisplayLayoutCuMem;
	cudaDeviceMem<float> mTsneLatentCuMem;
	std::vector<float> mTsneVisGraphData;
	std::vector<int> mLatentTsneMap;
	std::vector<std::vector<int>> mTsneLatentMap;
	std::vector<int> mTsneColorIds;
	std::vector<int> mTsneMatchIds;
	std::vector<int> mTsneQueryIds;

	DBSCAN mDBScan;
	DBScanDataMode mDBScanDataMode;

	
	std::vector<float> mKunHuaDisMat_row2;
	

	std::vector<vec3f> mNormalCurves;
	std::vector<float> mNormalDistData;
	std::vector<vec3f> mBinormalCurves;
	std::vector<float> mBinormalDistData;
	std::vector<VisibilityGraph*> mGraphs;
	
	std::vector<vec4f> TandS;

	vec4f mQueryColor;
	vec4f mMatchColor;
	vec4f mContextColor;
};

