#pragma once
#include "cudaDeviceMem.h"
#include "cudaStreamlineRenderer.h"
#include "cudaStreamlineTracer.h"
#include "DataUser.h"
#include "DisplayWidget.h"
#include "StreamlineResample.h"
#include "StreamlinePool3d.h"
#include "typeOperation.h"
#include "MatrixData.h"
#include "VisibilityGraph.h"
#include "Registration.h"
#include "VolumeData.h"
#include "WindowsTimer.h"
#include <string>
#include <vector>
#include <rapidjson/document.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <fstream>
#include <algorithm>

#define FRECHET_NOT_INIT	-1.0f
#define FRECHET_INIT_THRESH	-0.9f
#define FRECHET_INFINITY	1e30
#define MIN(A,B)  ((A) <= (B) ?  (A) : (B))

using namespace StreamlineResample;
using std::vector;
using std::string;

class FlowEncoderDataManager;
//1
//VisibilityGraphDisplayWidget从DisplayWidget和DataUser继承
class FlowEncoder : public DisplayWidget, public DataUser {
public:
	FlowEncoder(int x, int y, int w, int h, std::string name,bool Reflag);

protected:
	void init();
	void display();
	void menuCallback(const std::string& message) {}
	void onDataItemChanged(const std::string& name) {}
	void mouseReleaseEvent(const MouseEvent& e);

private:
	void drawStreamlineSegments() {}
	//void drawStreamlines();
	void drawStreamline(const int& sid) {}
	void singleStepJitterDisplay();
	int mStreamlineId;
	std::vector<StreamlineSegment> mStreamlineSegments;

	//数据维度
	vec3i dim;
	vector<vector<vec3f>> BiSeed;
	vector<vec3f>points3;
	vector<Streamline> stl3;
	vector<float> sacc_curvs3;

	StreamlinePool VelocityFromStreamlinePool;

	//string streamline_path = "E:/data/project_data/VisibilityGraph/429/5cp/streamline.stl";
	string name = "tornado";
	//string streamline_path = "E:/VS2019WorkSpace/visibilitygraph/data/streamline/" + name + ".stl";
	string streamline_path = "E:/data/flow2/"+name+".stl";
	std::string vec_field_path= "E:/data/flow/" + name + ".vec";
	std::string vec_hdr_path= "E:/data/flow/" + name + ".hdr";
	StreamlinePool RenderPool;
	FlowEncoderDataManager* mData;
	cudaStreamlineRenderer* mRender;

	bool test;
	         
};

class FlowEncoderDataManager {
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
	FlowEncoderDataManager(StreamlinePool& mPool) {
		VelocityFromBinormalPool = mPool;

	}
	~FlowEncoderDataManager() {

	}

	// 对pool内的流线做times次的smooth
	void smooth(StreamlinePool& pool, int times) {
		int num = pool.streamlines.size();
		StreamlinePool retPool;
		for (int i = 0; i < num; ++i) {
			Streamline stl = pool.streamlines[i];
			auto begin = pool.points.begin() + stl.start;
			auto end = begin + stl.numPoint;
			vector<vec3f>tmp(begin, end);
			int time = times;
			while (time--) {
				for (int j = 1; j < stl.numPoint - 1; ++j) {
					tmp[j] = 0.5 * (tmp[j - 1] + tmp[j + 1]);
				}
			}
			retPool.points.insert(retPool.points.end(), tmp.begin(), tmp.end());
		}
		retPool.streamlines = pool.streamlines;
		retPool.fillLineIds();
		pool.points.clear();
		pool.streamlines.clear();
		pool.points = retPool.points;
		pool.streamlines = retPool.streamlines;
		pool.fillLineIds();
	}

	// 从脑网络流线筛选流线生成数据
	void genStreamlineDataFromBrain() {
		string stlPath = "E:/data/brain/127_S_5056_1_fibers_FA_normed2.stl";
		StreamlinePool Pool;
		int minNum = 160;
		genBrainStreamlinePool(Pool, stlPath.c_str(), minNum);
	}

	void genBrainStreamlinePool(StreamlinePool& Pool, const char* file, int minNum) {
		StreamlinePool oriPool, resPool;
		readBrainStreamlinePool(oriPool, file);
		for (int i = 0; i < oriPool.streamlines.size(); ++i) {
			const Streamline& s = oriPool.streamlines[i];
			if (s.numPoint >= minNum) {
				Streamline sampleStl = makeStreamline(0, resPool.points.size(), 0);
				for (int j = s.start; j < s.start + s.numPoint; ++j) {
					resPool.points.push_back(oriPool.points[j]);
				}
				sampleStl.numPoint = s.numPoint;
				resPool.streamlines.push_back(sampleStl);
			}
		}
		resPool.fillLineIds();
		Pool = resPool;
		string stlpath = "E:/data/flow/brain.stl";
		saveStreamlinePool(Pool.points.data(), Pool.streamlines.data(), Pool.points.size(), Pool.streamlines.size(), stlpath.c_str());
		StreamlinePool testPool;
		readStreamlinePoolForBrain(testPool, stlpath.c_str());
		resample(Pool, segment_length);
		getEqualPointStreamlinePool(Pool, sample_numpoint);

		// 流线数据：shape,endpoints,orientation
		genStreamlineDataFromBrain(Pool);

		//// distance type
		//genDistanceTypeFromBrain(Pool);

		//// distance value
		//genDistanceFromBrain(Pool);
	}

	bool saveStreamlinePool(vec3f* points, Streamline* stls, int num_point, int num_stl, const char* file) {
		std::ofstream outfile;
		outfile.open(file, std::ios::binary);
		if (!outfile.is_open()) {
			printf("Unable to write file: %s.", file);
			return false;
		}

		outfile.write((char*)&num_point, sizeof(int));
		outfile.write((char*)&num_stl, sizeof(int));
		outfile.write((char*)points, sizeof(vec3f) * num_point);
		outfile.write((char*)stls, sizeof(Streamline) * num_stl);
		outfile.close();

		return true;
	}

	bool readStreamlinePoolForBrain(StreamlinePool& Pool, const char* file) {
		std::ifstream input_file;
		if (!open_file(input_file, file, true)) {
			return false;
		}

		int num_points, num_stls;
		input_file.read((char*)&num_points, sizeof(int));
		input_file.read((char*)&num_stls, sizeof(int));

		Pool.points.resize(num_points);
		Pool.streamlines.resize(num_stls);

		input_file.read((char*)Pool.points.data(), sizeof(vec3f) * num_points);
		input_file.read((char*)Pool.streamlines.data(), sizeof(Streamline) * num_stls);
		input_file.close();

		Pool.fillLineIds();

		return true;
	}

	void genStreamlineDataFromBrain(StreamlinePool Pool) {
		string filepath = "E:/VS2019WorkSpace/visibilitygraph/data/brain/";

		StreamlinePool norPool = Pool;
		vector<vec4f>TandS;
		normalization(norPool, TandS);

		// if normalization
		//Pool = norPool;

		// distance matrix
		mDistData.clear();
		allocateData(norPool);
		cudaStreamlineRenderer* mmRender;
		mmRender = new cudaStreamlineRenderer(norPool.streamlines.data(), norPool.points.data(), norPool.streamlines.size(), 8, radius);
		computeAndSaveDistanceMatrix(norPool, mmRender);

		vector<float> shape, endpoints, orientation;
		for (int i = 0; i < Pool.streamlines.size(); ++i) {
			Streamline s1 = Pool.streamlines[i];
			for (int j = 0; j < Pool.streamlines.size(); ++j) {
				Streamline s2 = Pool.streamlines[j];
				// shape: s1
				int shapeSize = s1.numPoint * s1.numPoint;
				for (int r = i * shapeSize; r < (i + 1) * shapeSize; ++r) {
					shape.push_back(mDistData[r]);
				}
				// shape: s1
				for (int r = j * shapeSize; r < (j + 1) * shapeSize; ++r) {
					shape.push_back(mDistData[r]);
				}
				// endpoints: s1
				endpoints.push_back(Pool.points[s1.start].x);
				endpoints.push_back(Pool.points[s1.start].y);
				endpoints.push_back(Pool.points[s1.start].z);
				endpoints.push_back(Pool.points[s1.start + s1.numPoint - 1].x);
				endpoints.push_back(Pool.points[s1.start + s1.numPoint - 1].y);
				endpoints.push_back(Pool.points[s1.start + s1.numPoint - 1].z);
				// endpoints: s2
				endpoints.push_back(Pool.points[s2.start].x);
				endpoints.push_back(Pool.points[s2.start].y);
				endpoints.push_back(Pool.points[s2.start].z);
				endpoints.push_back(Pool.points[s2.start + s2.numPoint - 1].x);
				endpoints.push_back(Pool.points[s2.start + s2.numPoint - 1].y);
				endpoints.push_back(Pool.points[s2.start + s2.numPoint - 1].z);
				// orientation: s1
				for (int r = s1.start; r < s1.start + s1.numPoint; ++r) {
					orientation.push_back(Pool.points[r].x);
					orientation.push_back(Pool.points[r].y);
					orientation.push_back(Pool.points[r].z);
				}
				// orientation: s2
				for (int r = s2.start; r < s2.start + s2.numPoint; ++r) {
					orientation.push_back(Pool.points[r].x);
					orientation.push_back(Pool.points[r].y);
					orientation.push_back(Pool.points[r].z);
				}
				
			}
		}
		string shapepath = filepath + "shape/brain.dat";
		string endpointpath = filepath + "endpoints/brain.dat";
		string orientationpath = filepath + "orientation/brain.dat";
		write_array(shape.data(), shape.size(), shapepath.c_str());
		write_array(endpoints.data(), endpoints.size(), endpointpath.c_str());
		write_array(orientation.data(), orientation.size(), orientationpath.c_str());
	}

	void genDistanceTypeFromBrain(StreamlinePool Pool) {
		string filepath = "E:/VS2019WorkSpace/visibilitygraph/data/brain/";
		string disType[] = { "dE", "dG", "dM", "dH", "dEP", "dP", "dF" };

		int distanceNum = 7;
		for (int count = 0; count < distanceNum; ++count) {
			vector<float> tag;
			for (int i = 0; i < Pool.streamlines.size(); ++i) {
				Streamline s1 = Pool.streamlines[i];
				for (int j = 0; j < Pool.streamlines.size(); ++j) {
					Streamline s2 = Pool.streamlines[j];

					// distance type
					float flag[7] = { 0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f };
					flag[count] = 1.0f;
					for (int r = 0; r < 7; ++r) {
						tag.push_back(flag[r]);
					}
				}
			}
			string tagpath = filepath + "tag/" + disType[count] + "_tag.dat";
			write_array(tag.data(), tag.size(), tagpath.c_str());
		}
	}

	void genDistanceFromBrain(StreamlinePool Pool) {
		vector<float> dE, dG, dM, dH, dEP, dP, dF;
		compute_dE(Pool, dE);
		compute_dG(Pool, dG);
		compute_dM(Pool, dM);
		compute_dH(Pool, dH);
		compute_dEP(Pool, dEP);
		compute_dP(Pool, dP);
		float* matrix = genDiscreteFrechetDistanceMatrix(Pool.streamlines.data(), Pool.streamlines.size(), Pool.points.data());
		for (int i = 0; i < Pool.streamlines.size() * Pool.streamlines.size(); ++i) {
			dF.push_back(matrix[i]);
		}
		vector<vector<float>>allDistance;
		allDistance.push_back(dE);
		allDistance.push_back(dG);
		allDistance.push_back(dM);
		allDistance.push_back(dH);
		allDistance.push_back(dEP);
		allDistance.push_back(dP);
		allDistance.push_back(dF);
		string filepath = "E:/VS2019WorkSpace/visibilitygraph/data/brain/";
		string disType[] = { "dE", "dG", "dM", "dH", "dEP", "dP", "dF" };
		for (int count = 0; count < allDistance.size(); ++count) {
			vector<float>curDistance = allDistance[count];
			string curPath = filepath + "label/" + disType[count] + ".dat";
			write_array(curDistance.data(), curDistance.size(), curPath.c_str());
		}
	}

	// 从不同数据集生成streamline数据
	/*void readPoolAndGenDistance() {
		srand((int)time(0));
		string dataset[] = { "abc", "bernard", "computer_room", "cylinder", "electro3D", "plume", "random-5cp", "tornado", "two_swirl", "vessel" };
		for (auto i : dataset) {
			string filename = "E:/VS2019WorkSpace/visibilitygraph/data/transform/streamline/";
			for (int j = 11; j < 15; ++j) {
				string str;
				std::stringstream ss;
				ss << j;
				ss >> str;
				string stlPath = filename + i + str + ".stl";
				StreamlinePool Pool;
				readStreamlinePool(Pool, stlPath.c_str());
				distanceGeneration(Pool, i, str);
				printf("%s: version %d finished!\n", i.c_str(), j);
			}
			printf("%s finished!\n", i.c_str());
		}
	}*/

	// 将流线通过平移和缩变换到指定的大小维度，生成数据
	void transform(vec3i dim) {
		//vec3f center = makeVec3f(dim.x * 1.0f / 2, dim.y * 1.0f / 2, dim.z * 1.0f / 2);
		srand((int)time(0));
		string dataset[] = { "abc", "bernard", "computer_room", "cylinder", "electro3D", "plume", "random-5cp", "tornado", "two_swirl", "vessel"};
		for (auto i : dataset) {
			string filename = "E:/data/flow/" + i + ".stl";
			StreamlinePool Pool;
			readStreamlinePool(Pool, filename.c_str());
			resample(Pool, segment_length);
			getEqualPointStreamlinePool(Pool, sample_numpoint);

			vector<vec4f> TandS;
			normalization(Pool, TandS);

			float maxS = 0.0f;
			checkBoundary(Pool, dim, maxS);
			int addTimes = 15;
			StreamlinePool aPool;
			for (int k = 10; k < addTimes; ++k) {
				aPool = Pool;

				// 每条流线加上随机的尺度和移动
				addTandS(aPool, dim, maxS / 2, maxS);

				string str;
				std::stringstream ss;
				ss << k;
				ss >> str;
				string stlPath = "E:/VS2019WorkSpace/visibilitygraph/data/transform/streamline/" + i + str + ".stl";
				writeStreamlinePool(aPool, stlPath.c_str());

				// 生成reconstruction数据
				dataGeneration(aPool, i, str);

				// 生成distance数据(train/test)
				distanceGeneration(aPool, i, str);
				printf("%s: version %d finished!\n", i.c_str(), k);
			}
			printf("%s finished!\n", i.c_str());
		}
	}

	// 每条流线加上随机的尺度和移动
	// 先随机加尺度，计算box的边界，将box移动到target space的中心，确定可随机移动的范围， 再随机加移动
	void addTandS(StreamlinePool& Pool, vec3i dim, float min, float max) {
		// target space的中心
		vec3f targetCenter = makeVec3f(dim.x * 1.0f / 2, dim.y * 1.0f / 2, dim.z * 1.0f / 2);
		// 随机加尺度
		for (int i = 0; i < Pool.streamlines.size(); ++i) {
			float randomScale = min + static_cast<float>(rand()) / static_cast<float>(RAND_MAX / (max - min));
			Streamline stl = Pool.streamlines[i];
			for (int j = stl.start; j < stl.start + stl.numPoint; ++j) {
				Pool.points[j] = Pool.points[j] * randomScale;
			}
		}
		//变换后的space移到target space的中间
		vec3f center;
		float minx = 0.0f, maxx = 0.0f;
		float miny = 0.0f, maxy = 0.0f;
		float minz = 0.0f, maxz = 0.0f;
		vector<float>x, y, z;
		for (int i = 0; i < Pool.points.size(); ++i) {
			x.push_back(Pool.points[i].x);
			y.push_back(Pool.points[i].y);
			z.push_back(Pool.points[i].z);
		}
		computeMinMax(x.data(), Pool.points.size(), minx, maxx);
		computeMinMax(y.data(), Pool.points.size(), miny, maxy);
		computeMinMax(y.data(), Pool.points.size(), minz, maxz);
		/*printf("x: min = %f, max = %f\n", minx, maxx);
		printf("y: min = %f, max = %f\n", miny, maxy);
		printf("z: min = %f, max = %f\n", minz, maxz);*/
		center = makeVec3f((maxx + minx) / 2, (maxy + miny) / 2, (maxz + minz) / 2);
		for (int i = 0; i < Pool.points.size(); ++i) {
			Pool.points[i] = Pool.points[i] + targetCenter - center;
		}
		/*float tmp = 0.0f;
		checkBoundary(Pool, dim, tmp);*/

		// 对每条线进行随机的平移: 计算可移动的范围，再取随机的平移向量
		for (int i = 0; i < Pool.streamlines.size(); ++i) {
			Streamline s = Pool.streamlines[i];
			coordinateRange xBoundary, yBoundary, zBoundary;
			computeTranslateBoundaryForStreamline(Pool, i, xBoundary, yBoundary, zBoundary, dim);
			// 根据可移动的范围取随机的平移向量
			float x = xBoundary.lower + static_cast<float>(rand()) / static_cast<float>(RAND_MAX / (xBoundary.upper - xBoundary.lower));
			float y = yBoundary.lower + static_cast<float>(rand()) / static_cast<float>(RAND_MAX / (yBoundary.upper - yBoundary.lower));
			float z = zBoundary.lower + static_cast<float>(rand()) / static_cast<float>(RAND_MAX / (zBoundary.upper - zBoundary.lower));
			//流线整体平移(x,y,z)
			for (int j = s.start; j < s.start + s.numPoint; ++j) {
				Pool.points[j] = Pool.points[j] + makeVec3f(x, y, z);
			}
		}
	}

	// 生成数据: shape, endpoints, orientation
	void dataGeneration(StreamlinePool Pool, string dataset, string version) {
		string filename = "E:/VS2019WorkSpace/visibilitygraph/data/transform/restruction/train/";

		// before normalization
		// orientation and endpoints
		vector<float> orientationOri, endpointsOri;
		string orientationOriPath = filename + "orientationOri/" + dataset + version + ".dat";
		string endpointsOriPath = filename + "endpointsOri/" + dataset + version + ".dat";
		for (int i = 0; i < Pool.streamlines.size(); ++i) {
			Streamline s = Pool.streamlines[i];
			vec3f start = Pool.points[s.start];
			vec3f end = Pool.points[s.start + s.numPoint - 1];
			endpointsOri.push_back(start.x);
			endpointsOri.push_back(start.y);
			endpointsOri.push_back(start.z);
			endpointsOri.push_back(end.x);
			endpointsOri.push_back(end.y);
			endpointsOri.push_back(end.z);
			for (int j = s.start; j < s.start + s.numPoint; ++j) {
				orientationOri.push_back(Pool.points[j].x);
				orientationOri.push_back(Pool.points[j].y);
				orientationOri.push_back(Pool.points[j].z);
			}
		}
		write_array(orientationOri.data(), orientationOri.size(), orientationOriPath.c_str());
		write_array(endpointsOri.data(), endpointsOri.size(), endpointsOriPath.c_str());

		StreamlinePool norPool = Pool;
		vector<vec4f>TandS;
		normalization(norPool, TandS);

		// after normalization
		// shape
		allocateData(norPool);
		cudaStreamlineRenderer* mmRender;
		mmRender = new cudaStreamlineRenderer(norPool.streamlines.data(), norPool.points.data(), norPool.streamlines.size(), 8, radius);
		computeAndSaveDistanceMatrix(norPool, mmRender);
		string shapePath = filename + "shape/" + dataset + version + ".dat";
		write_array(mDistData.data(), mDistData.size(), shapePath.c_str());

		// orientation and endpoints
		vector<float> orientationNor, endpointsNor;
		string orientationNorPath = filename + "orientationNor/" + dataset + version + ".dat";
		string endpointsNorPath = filename + "endpointsNor/" + dataset + version + ".dat";
		for (int i = 0; i < norPool.streamlines.size(); ++i) {
			Streamline s = norPool.streamlines[i];
			vec3f start = norPool.points[s.start];
			vec3f end = norPool.points[s.start + s.numPoint - 1];
			endpointsNor.push_back(start.x);
			endpointsNor.push_back(start.y);
			endpointsNor.push_back(start.z);
			endpointsNor.push_back(end.x);
			endpointsNor.push_back(end.y);
			endpointsNor.push_back(end.z);
			for (int j = s.start; j < s.start + s.numPoint; ++j) {
				orientationNor.push_back(norPool.points[j].x);
				orientationNor.push_back(norPool.points[j].y);
				orientationNor.push_back(norPool.points[j].z);
			}
		}
		write_array(orientationNor.data(), orientationNor.size(), orientationNorPath.c_str());
		write_array(endpointsNor.data(), endpointsNor.size(), endpointsNorPath.c_str());
	}


	// 生成distance数据
	void distanceGeneration(StreamlinePool Pool, string dataset, string version) {
		string filename = "E:/VS2019WorkSpace/visibilitygraph/data/transform/distance/test1/";
		StreamlinePool norPool = Pool;
		vector<vec4f>TandS;
		normalization(norPool, TandS);

		// if normalization
	    //Pool = norPool;

		// distance matrix
		mDistData.clear();
		allocateData(norPool);
		cudaStreamlineRenderer* mmRender;
		mmRender = new cudaStreamlineRenderer(norPool.streamlines.data(), norPool.points.data(), norPool.streamlines.size(), 8, radius);
		computeAndSaveDistanceMatrix(norPool, mmRender);

		// compute distance
		vector<float> dP, dF;
		vector<vector<float>>distance;
		compute_dP(Pool, dP);
		float* distanceMatrix = genDiscreteFrechetDistanceMatrix(Pool.streamlines.data(), Pool.streamlines.size(), Pool.points.data());
		for (int i = 0; i < Pool.streamlines.size() * Pool.streamlines.size(); ++i) {
			dF.push_back(distanceMatrix[i]);
		}

		// random stl pairs and generation distance data
		vector<float> shape, endpoints, orientation, tag, pairDistance;
		int sampleNum = 150;
		int distanceNum = 7;
		for (int count = 0; count < distanceNum; ++count) {
			for (int m = 0; m < sampleNum; ++m) {
				int i = rand() % Pool.streamlines.size();
				Streamline s1 = Pool.streamlines[i];
				for (int n = 0; n < sampleNum; ++n) {
					int j = rand() % Pool.streamlines.size();
					Streamline s2 = Pool.streamlines[j];
					// streamline data
					// shape: s1
					int shapeSize = s1.numPoint * s1.numPoint;
					for (int r = i * shapeSize; r < (i + 1) * shapeSize; ++r) {
						shape.push_back(mDistData[r]);
					}
					// shape: s2
					for (int r = j * shapeSize; r < (j + 1) * shapeSize; ++r) {
						shape.push_back(mDistData[r]);
					}
					// endpoints: s1
					endpoints.push_back(Pool.points[s1.start].x);
					endpoints.push_back(Pool.points[s1.start].y);
					endpoints.push_back(Pool.points[s1.start].z);
					endpoints.push_back(Pool.points[s1.start + s1.numPoint - 1].x);
					endpoints.push_back(Pool.points[s1.start + s1.numPoint - 1].y);
					endpoints.push_back(Pool.points[s1.start + s1.numPoint - 1].z);
					// endpoints: s2
					endpoints.push_back(Pool.points[s2.start].x);
					endpoints.push_back(Pool.points[s2.start].y);
					endpoints.push_back(Pool.points[s2.start].z);
					endpoints.push_back(Pool.points[s2.start + s2.numPoint - 1].x);
					endpoints.push_back(Pool.points[s2.start + s2.numPoint - 1].y);
					endpoints.push_back(Pool.points[s2.start + s2.numPoint - 1].z);
					// orientation: s1
					for (int r = s1.start; r < s1.start + s1.numPoint; ++r) {
						orientation.push_back(Pool.points[r].x);
						orientation.push_back(Pool.points[r].y);
						orientation.push_back(Pool.points[r].z);
					}
					// orientation: s2
					for (int r = s2.start; r < s2.start + s2.numPoint; ++r) {
						orientation.push_back(Pool.points[r].x);
						orientation.push_back(Pool.points[r].y);
						orientation.push_back(Pool.points[r].z);
					}
					// distance type
					float flag[7] = { 0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f };
					flag[count] = 1.0f;
					for (int r = 0; r < 7; ++r) {
						tag.push_back(flag[r]);
					}
					// distance between s1 and s2
					if (count < 5) {
						float d = computeDistance(Pool, i, j, count);
						pairDistance.push_back(d);
					}
					else if(count == 5){
						pairDistance.push_back(dP[i * Pool.streamlines.size() + j]);
					}
					else {
						pairDistance.push_back(dF[i * Pool.streamlines.size() + j]);
					}
				}
			}
		}
		string shapePath = filename + "shape/" + dataset + version + ".dat";
		string endpointsPath = filename + "endpoints/" + dataset + version + ".dat";
		string orientationPath = filename + "orientation/" + dataset + version + ".dat";
		string tagPath = filename + "tag/" + dataset + version + ".dat";
		string oriLabelPath = filename + "oriLabel/" + dataset + version + ".dat";
		write_array(shape.data(), shape.size(), shapePath.c_str());
		write_array(endpoints.data(), endpoints.size(), endpointsPath.c_str());
		write_array(orientation.data(), orientation.size(), orientationPath.c_str());
		write_array(tag.data(), tag.size(), tagPath.c_str());
		write_array(pairDistance.data(), pairDistance.size(), oriLabelPath.c_str());
	}

	float computeDistance(StreamlinePool Pool, int sid1, int sid2, int distanceType) {
		float distance = 0.0f;
		if (distanceType == 0) {
			distance = computeDEforStl(Pool, sid1, sid2);
		}
		else if (distanceType == 1) {
			distance = computeDGforStl(Pool, sid1, sid2);
		}
		else if (distanceType == 2) {
			distance = computeDMforStl(Pool, sid1, sid2);
		}
		else if (distanceType == 3) {
			distance = computeDHforStl(Pool, sid1, sid2);
		}
		else if (distanceType == 4) {
			distance = computeDEPforStl(Pool, sid1, sid2);
		}
		return distance;
	}

	// 欧氏距离
	float computeDEforStl(StreamlinePool Pool, int sid1, int sid2) {
		Streamline s1 = Pool.streamlines[sid1];
		Streamline s2 = Pool.streamlines[sid2];
		float distance = 0.0f;
		for (int r = s1.start, k = s2.start; r < s1.start + s1.numPoint, k < s2.start + s2.numPoint; ++r, ++k) {
			distance += dist3d(Pool.points[r], Pool.points[k]);
		}
		distance = distance / s1.numPoint;
		return distance;
	}

	// 几何相似性度量
	float computeDGforStl(StreamlinePool Pool, int sid1, int sid2) {
		Streamline s1 = Pool.streamlines[sid1];
		Streamline s2 = Pool.streamlines[sid2];
		float distance = 0.0f;
		float parallelism = 0.0f;
		for (int r = s1.start + 1, k = s2.start + 1; r < s1.start + s1.numPoint, k < s2.start + s2.numPoint; ++r, ++k) {
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
		distance = parallelism / (s1.numPoint - 1);
		return distance;
	}

	// MCP distance
	float computeDMforStl(StreamlinePool Pool, int sid1, int sid2) {
		float distance = 0.0f;
		float distance1 = mean_closest_point_distance(Pool, sid1, sid2);
		float distance2 = mean_closest_point_distance(Pool, sid2, sid1);
		distance = (distance1 + distance2) / 2.0f;
		return distance;
	}

	// Hausdorff distance
	float computeDHforStl(StreamlinePool Pool, int sid1, int sid2) {
		float distance = 0.0f;
		float distance1 = h_distance(Pool, sid1, sid2);
		float distance2 = h_distance(Pool, sid2, sid1);
		if (distance1 > distance2) {
			distance = distance1;
		}
		else {
			distance = distance2;
		}
		return distance;
	}

	// Endpoints distance 
	float computeDEPforStl(StreamlinePool Pool, int sid1, int sid2) {
		Streamline s1 = Pool.streamlines[sid1];
		Streamline s2 = Pool.streamlines[sid2];
		float distance = 0.0f;
		float distance1 = dist3d(Pool.points[s1.start], Pool.points[s2.start]);
		float distance2 = dist3d(Pool.points[s1.start + s1.numPoint - 1], Pool.points[s2.start + s2.numPoint - 1]);
		distance = (distance1 + distance2) / 2.0f;
		return distance;
	}

	void checkBoundary(StreamlinePool Pool, vec3i dim, float& maxS) {
		vec3f point = Pool.points[0];
		float minx = point.x, maxx = point.x;
		float miny = point.y, maxy = point.y;
		float minz = point.z, maxz = point.z;
		vector<float>x, y, z;
		for (int i = 0; i < Pool.points.size(); ++i) {
			x.push_back(Pool.points[i].x);
			y.push_back(Pool.points[i].y);
			z.push_back(Pool.points[i].z);
		}
		computeMinMax(x.data(), Pool.points.size(), minx, maxx);
		computeMinMax(y.data(), Pool.points.size(), miny, maxy);
		computeMinMax(y.data(), Pool.points.size(), minz, maxz);
		
		maxS = MIN(MIN(floor(dim.x * 1.0f / (maxx - minx)), floor(dim.y * 1.0f / (maxy - miny))), floor(dim.z * 1.0f / (maxz - minz))) * 1.0f;
		
	}

	void computeTranslateBoundaryForStreamline(StreamlinePool Pool, int sid, coordinateRange& x, coordinateRange& y, coordinateRange& z, vec3i dim) {
		Streamline s = Pool.streamlines[sid];
		vector<float> x_axis, y_axis, z_axis;
		for (int j = s.start; j < s.start + s.numPoint; ++j) {
			x_axis.push_back(Pool.points[j].x);
			y_axis.push_back(Pool.points[j].y);
			z_axis.push_back(Pool.points[j].z);
		}
		float x_lower = 0.0f, x_upper = 0.0f;
		float y_lower = 0.0f, y_upper = 0.0f;
		float z_lower = 0.0f, z_upper = 0.0f;
		computeMinMax(x_axis.data(), x_axis.size(), x_lower, x_upper);
		computeMinMax(y_axis.data(), y_axis.size(), y_lower, y_upper);
		computeMinMax(z_axis.data(), z_axis.size(), z_lower, z_upper);
		x.lower = -x_lower;
		x.upper = dim.x - x_upper;
		y.lower = -y_lower;
		y.upper = dim.y - y_upper;
		z.lower = -z_lower;
		z.upper = dim.z - z_upper;
	}

	void resample(StreamlinePool& tmpPool) {
		StreamlinePool mPool = VelocityFromBinormalPool;
		StreamlinePool targetPool = mPool;

		// 等距采样sample_numpoint个点
		resample(targetPool, resample_scales[0] * segment_length);
		getEqualPointStreamlinePool(targetPool, sample_numpoint);

		// 归一化流线池
		StreamlinePool samplePool = targetPool;
		vector<vec4f> TandS;
		normalization(samplePool, TandS);
		tmpPool = samplePool;
		tmpPool = targetPool;
		allocateData(samplePool);

	}

	// original theta
	void gentheta() {
		string dataset[] = { "abc", "bernard", "computer_room","cylinder","electro3D","plume","random-5cp","tornado","two_swirl","vessel" };
		for (auto i : dataset) {
			string filename = "E:/VS2019WorkSpace/visibilitygraph/data/streamline/" + i + ".stl";
			StreamlinePool Pool;
			readStreamlinePool(Pool, filename.c_str());
			//sampling
			resample(Pool, segment_length);
			getEqualPointStreamlinePool(Pool, sample_numpoint);
			//normailzation
			StreamlinePool norPool = Pool;
			vector<vec4f> TandS;
			normalization(norPool, TandS);
			//normalize or not
			//Pool = norPool;
			//流线旋转
			vector<vector<vector<vec3f>>> After_Rotate_test;
			string datasetpath = "E:/VS2019WorkSpace/visibilitygraph/data/originaltheta/test/dataset/" + i + ".dat";
			string labelpath = "E:/VS2019WorkSpace/visibilitygraph/data/originaltheta/test/label/" + i + ".dat";
			gettestData(After_Rotate_test, Pool, 10, datasetpath, labelpath);
			printf("%s finished!\n", i.c_str());
		}
	}

	void computeDistanceMean(vector<vector<float>>distancetype, vector<float>& distanceMean) {
		for (int i = 0; i < distancetype.size(); ++i) {
			vector<float>current_distance = distancetype[i];
			float sum = 0.0f;
			for (int j = 0; j < current_distance.size(); ++j) {
				sum += current_distance[j];
			}
			distanceMean.push_back(sum / current_distance.size());
		}
	}

	void enlarge(StreamlinePool& Pool, int n) {
		for (int i = 0; i < Pool.streamlines.size(); ++i) {
			Streamline s = Pool.streamlines[i];
			for (int j = s.start; j < s.start + s.numPoint; ++j) {
				Pool.points[j].x *= n;
				Pool.points[j].y *= n;
				Pool.points[j].z *= n;
			}
		}
	}

	//void shapeAndlocation() {
	//	string dataset[] = { "abc", "bernard", "computer_room", "cylinder", "electro3D", "plume", "random-5cp", "tornado", "two_swirl", "vessel", "crayfish" };
	//	//string dataset[] = { "bernard", "computer_room", "plume" };
	//	for (auto i : dataset) {
	//		//string filename = "E:/VS2019WorkSpace/visibilitygraph/data/streamline/" + i + ".stl";
	//		string filename = "E:/data/flow/" + i + ".stl";
	//		//printf("%s\n", filename.c_str());
	//		StreamlinePool Pool;
	//		readStreamlinePool(Pool, filename.c_str());
	//		resample(Pool, segment_length);
	//		getEqualPointStreamlinePool(Pool, sample_numpoint);

	//		//original points(before normalization)
	//		vector<float>originalpointdata;
	//		string originalpointpath = "E:/VS2019WorkSpace/visibilitygraph/data/T-Net/originalpoint/" + i + "1.dat";
	//		for (int i = 0; i < Pool.streamlines.size(); ++i) {
	//			Streamline s = Pool.streamlines[i];
	//			for (int j = s.start; j < s.start + s.numPoint; ++j) {
	//				originalpointdata.push_back(Pool.points[j].x);
	//				originalpointdata.push_back(Pool.points[j].y);
	//				originalpointdata.push_back(Pool.points[j].z);
	//			}
	//		}
	//		write_array(originalpointdata.data(), originalpointdata.size(), originalpointpath.c_str());

	//		//endpoints (before normalization)
	//		string endpointpath = "E:/VS2019WorkSpace/visibilitygraph/data/T-Net/originalendpoint/" + i + "1.dat";
	//		vector<float>endpointdata;
	//		for (int i = 0; i < Pool.streamlines.size(); ++i) {
	//			Streamline s = Pool.streamlines[i];
	//			vec3f start = Pool.points[s.start];
	//			vec3f end = Pool.points[s.start + s.numPoint - 1];
	//			endpointdata.push_back(start.x);
	//			endpointdata.push_back(start.y);
	//			endpointdata.push_back(start.z);
	//			endpointdata.push_back(end.x);
	//			endpointdata.push_back(end.y);
	//			endpointdata.push_back(end.z);
	//		}
	//		write_array(endpointdata.data(), endpointdata.size(), endpointpath.c_str());

	//		// normalization
	//		StreamlinePool tempPool = Pool;
	//		vector<vector<float>>matrixdata;
 //			shape_normalization(tempPool, matrixdata);

	//		//shape (after normalization)
	//		allocateData(tempPool);
	//		cudaStreamlineRenderer* mmRender;
	//		mmRender = new cudaStreamlineRenderer(tempPool.streamlines.data(), tempPool.points.data(), tempPool.streamlines.size(), 8, radius);
	//		computeAndSaveDistanceMatrix(tempPool, mmRender);
	//		string shapepath = "E:/VS2019WorkSpace/visibilitygraph/data/T-Net/shape/" + i + "1.dat";
	//		write_array(mDistData.data(), mDistData.size(), shapepath.c_str());

	//		//endpoint(after normalization)
	//		string endpointpath2 = "E:/VS2019WorkSpace/visibilitygraph/data/T-Net/norendpoint/" + i + "1.dat";
	//		vector<float>endpointdata2;
	//		for (int i = 0; i < tempPool.streamlines.size(); ++i) {
	//			Streamline s = tempPool.streamlines[i];
	//			vec3f start = tempPool.points[s.start];
	//			vec3f end = tempPool.points[s.start + s.numPoint - 1];
	//			endpointdata2.push_back(start.x);
	//			endpointdata2.push_back(start.y);
	//			endpointdata2.push_back(start.z);
	//			endpointdata2.push_back(end.x);
	//			endpointdata2.push_back(end.y);
	//			endpointdata2.push_back(end.z);
	//		}
	//		write_array(endpointdata2.data(), endpointdata2.size(), endpointpath2.c_str());

	//		//points(after normalization)
	//		string pointpath = "E:/VS2019WorkSpace/visibilitygraph/data/T-Net/norpoint/" + i + "1.dat";
	//		vector<float>pointdata;
	//		for (int i = 0; i < tempPool.streamlines.size(); ++i) {
	//			Streamline s = tempPool.streamlines[i];
	//			for (int j = s.start; j < s.start + s.numPoint; ++j) {
	//				pointdata.push_back(tempPool.points[j].x);
	//				pointdata.push_back(tempPool.points[j].y);
	//				pointdata.push_back(tempPool.points[j].z);
	//			}
	//		}
	//		write_array(pointdata.data(), pointdata.size(), pointpath.c_str());

	//		//translation and scale matrix
	//		string matrixpath = "E:/VS2019WorkSpace/visibilitygraph/data/T-Net/matrix/" + i + "1.dat";
	//		vector<float> TandS;
	//		for (int i = 0; i < tempPool.streamlines.size(); ++i) {
	//			vector<float> matrixOfline = matrixdata[i];
	//			vector<float> matrix = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	//			for (int j = 0; j < 16; ++j) {
	//				if (j % 5 == 0) {
	//					TandS.push_back(matrixOfline[3]);
	//				}
	//				else if (j >= 12) {
	//					TandS.push_back(matrixOfline[j % 4]);
	//				}
	//				else {
	//					TandS.push_back(0.0);
	//				}
	//			}
	//		}
	//		write_array(TandS.data(), TandS.size(), matrixpath.c_str());
	//	}
	//}

	// AddTandS:平移和尺度
	void AddTandS(StreamlinePool Pool, string path, bool tag) {
		vector<float>pointdata;
		for (int i = 0; i < Pool.streamlines.size(); ++i) {
			Streamline s = Pool.streamlines[i];
			//xyz的均值
			float x_mean = 0.0f, y_mean = 0.0f, z_mean = 0.0f;
			for (int j = s.start; j < s.start + s.numPoint; ++j) {
				x_mean += Pool.points[j].x;
				y_mean += Pool.points[j].y;
				z_mean += Pool.points[j].z;
			}
			//平均点
			x_mean /= s.numPoint;
			y_mean /= s.numPoint;
			z_mean /= s.numPoint;
			float scale = 0.0f;
			for (int j = s.start; j < s.start + s.numPoint; ++j) {
				scale += sqrt((Pool.points[j].x - x_mean) * (Pool.points[j].x - x_mean) + (Pool.points[j].y - y_mean) * (Pool.points[j].y - y_mean) + (Pool.points[j].z - z_mean) * (Pool.points[j].z - z_mean));
			}
			// AddT
			if (tag) {
				for (int j = s.start; j < s.start + s.numPoint; ++j) {
					pointdata.push_back(Pool.points[j].x - x_mean);
					pointdata.push_back(Pool.points[j].y - y_mean);
					pointdata.push_back(Pool.points[j].z - z_mean);
				}
			}
			// AddS
			else {
				for (int j = s.start; j < s.start + s.numPoint; ++j) {
					pointdata.push_back(Pool.points[j].x / scale);
					pointdata.push_back(Pool.points[j].y / scale);
					pointdata.push_back(Pool.points[j].z / scale);
				}
			}
		}
		write_array(pointdata.data(), pointdata.size(), path.c_str());
	}

	// without translate and scale
	void getReconstructPool(StreamlinePool& tmpPool, StreamlinePool samplePool) {
		string reconstruct_point_path = "D:/Pycharm/pointProject6/result2/tornado2.dat";
	
		vector<float>coordinate;
		read_array(coordinate, reconstruct_point_path.c_str());
		vector<vec3f>points;
		tmpPool = samplePool;
		for (int i = 0; i < tmpPool.streamlines.size(); ++i) {
			Streamline s = tmpPool.streamlines[i];
			for (int j = 0; j < s.numPoint; ++j) {
				float x = coordinate[i * 96 + j];
				float y = coordinate[i * 96 + s.numPoint + j];
				float z = coordinate[i * 96 + s.numPoint * 2 + j];
				vec3f newPoint = makeVec3f(x, y, z);
				points.push_back(newPoint);
			}
		}
		for (int i = 0; i < tmpPool.streamlines.size(); ++i) {
			Streamline s = tmpPool.streamlines[i];
			for (int j = s.start; j < s.start + s.numPoint; ++j) {
				tmpPool.points[j] = points[j];
			}
		}
	}

	//欧氏距离
	void compute_dE(StreamlinePool Pool, vector<float>&distance_of_dE) {
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
	void compute_dG(StreamlinePool Pool, vector<float>&distance_of_dG) {
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
	void computer_dR(StreamlinePool Pool, vector<float>&distance_of_dR) {

	}

	//MCP distance
	void compute_dM(StreamlinePool Pool, vector<float>&distance_of_dM) {
		for (int i = 0; i < Pool.streamlines.size(); ++i) {
			Streamline currentLine = Pool.streamlines[i];
			for (int j = 0; j < Pool.streamlines.size(); ++j) {
				Streamline computeLine = Pool.streamlines[j];
				float distance1 = mean_closest_point_distance(Pool, i, j);
				float distance2 = mean_closest_point_distance(Pool, j, i);
				//两个平均值之和的平均值
				distance_of_dM.push_back((distance1 + distance2) / 2.0f);
				printf("dM: %d/%d complete\n", i * Pool.streamlines.size() + j + 1, Pool.streamlines.size() * Pool.streamlines.size());
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
				//float distance = sqrt((Pool.points[i].x - Pool.points[j].x) * (Pool.points[i].x - Pool.points[j].x) + (Pool.points[i].y - Pool.points[j].y) * (Pool.points[i].y - Pool.points[j].y) + (Pool.points[i].z - Pool.points[j].z) * (Pool.points[i].z - Pool.points[j].z));
				float distance =  dist3d(Pool.points[i], Pool.points[j]);
				if (distance < min_distance) {
					min_distance = distance;
				}
			}
			mean += min_distance;
		}
		return mean / s1.numPoint;
	}

	//Hausdorff distance
	void compute_dH(StreamlinePool Pool, vector<float>&distance_of_dH) {
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
				//float distance = sqrt((Pool.points[i].x - Pool.points[j].x) * (Pool.points[i].x - Pool.points[j].x) + (Pool.points[i].y - Pool.points[j].y) * (Pool.points[i].y - Pool.points[j].y) + (Pool.points[i].z - Pool.points[j].z) * (Pool.points[i].z - Pool.points[j].z));
				float distance = dist3d(Pool.points[i], Pool.points[j]);
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

	// Procrustes distance
	void compute_dP(StreamlinePool Pool, vector<float>&distance_of_dP) {
		for (int i = 0; i < Pool.streamlines.size(); ++i) {
			Streamline currentLine = Pool.streamlines[i];
			for (int j = 0; j < Pool.streamlines.size(); ++j) {
				Streamline computeLine = Pool.streamlines[j];
				float Procrustes_distance = computeProcrustesDistanceWithoutOrder(Pool.points.data() + currentLine.start, Pool.points.data() + computeLine.start, currentLine.numPoint, false);
				printf("dP: %d/%d complete\n", i * Pool.streamlines.size() + j + 1, Pool.streamlines.size() * Pool.streamlines.size());
				distance_of_dP.push_back(Procrustes_distance / currentLine.numPoint);
			}
		}
	}

	// Frechet distance
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

	cudaStreamlineTracer* getTracer() {
		if (mTracer == NULL) {
			int size = mDim.x * mDim.y * mDim.z;
			std::vector<vec3f> vec_field(size);
			read_array<vec3f>(vec_field.data(), size, vec_field_path.c_str());
			mTracer = new cudaStreamlineTracer(vec_field.data(), mDim.x, mDim.y, mDim.z);
		}
		return mTracer;
	}
	inline StreamlineTraceParameter getTracingParameters() {
		StreamlineTraceParameter pars;
		pars.max_point = 500;
		pars.min_point = 32 * 2.0;
		pars.segment_length = 2.0;
		pars.max_streamline = 300 * 10;
		pars.store_gap = 1;
		return pars;
	}
	void shape_normalization(StreamlinePool& Pool, vector<vector<float>>& matrixdata) {
		for (int i = 0; i < Pool.streamlines.size(); ++i) {
			vector<float> matrixOfline;
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
			matrixOfline.push_back(x_mean);
			matrixOfline.push_back(y_mean);
			matrixOfline.push_back(z_mean);
			float scale = 0.0f;
			for (int j = s.start; j < s.start + s.numPoint; ++j) {
				scale += sqrt((Pool.points[j].x - x_mean) * (Pool.points[j].x - x_mean) + (Pool.points[j].y - y_mean) * (Pool.points[j].y - y_mean) + (Pool.points[j].z - z_mean) * (Pool.points[j].z - z_mean));
			}
			matrixOfline.push_back(scale);
			matrixdata.push_back(matrixOfline);
			for (int j = s.start; j < s.start + s.numPoint; ++j) {
				Pool.points[j].x = (Pool.points[j].x - x_mean) / scale;
				Pool.points[j].y = (Pool.points[j].y - y_mean) / scale;
				Pool.points[j].z = (Pool.points[j].z - z_mean) / scale;
			}
		}
	}

	// 归一化
	void normalization(StreamlinePool& Pool,vector<vec4f>& TandS) {
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

	// 获取训练集
	void getData(vector<float>& data, vector<float>& label, vector<vector<vector<vec3f>>>& After_Rotate, int theta_interval) {
		//const float PI = 3.141592;
		for (int i = 0; i < After_Rotate.size(); ++i) {
			//当前流线的点池
			vector<vector<vec3f>> oneStreamlinePointPool = After_Rotate[i];
			for (int j = 0; j < oneStreamlinePointPool.size(); ++j) {
				//当前旋转角度下的流线上的点,数据1
				vector<vec3f> StreamlinePointData1 = oneStreamlinePointPool[j];
				for (int k = j + 1; k < oneStreamlinePointPool.size(); ++k) {
					vector<vec3f> StreamlinePointData2 = oneStreamlinePointPool[k];
					//数据1
					for (int r = 0; r < StreamlinePointData1.size(); ++r) {
						data.push_back(StreamlinePointData1[r].x);
						data.push_back(StreamlinePointData1[r].y);
						data.push_back(StreamlinePointData1[r].z);
					}
					//数据2
					for (int r = 0; r < StreamlinePointData2.size(); ++r) {
						data.push_back(StreamlinePointData2[r].x);
						data.push_back(StreamlinePointData2[r].y);
						data.push_back(StreamlinePointData2[r].z);
					}
					float currentLabel = (k - j) * theta_interval * PI / 180;
					label.push_back(currentLabel);
				}
			}
		}
		write_array(data.data(), data.size(), theta_dataset_path.c_str());
		write_array(label.data(), label.size(), theta_label_path.c_str());
	}

	//数据集
	void gettestData(vector<vector<vector<vec3f>>>& After_Rotate, StreamlinePool Pool, int test_num, string datasetpath, string labelpath) {
		//const float PI = 3.141592f;
		vector<vector<float>>ratatetheta;
		for (int i = 0; i < Pool.streamlines.size(); ++i) {
			const Streamline& s = Pool.streamlines[i];
			vec3f start = Pool.points[s.start];
			vec3f end = Pool.points[s.start + s.numPoint - 1];
			vector<vector<vec3f>> v;
			vector<float> theta_of_streamline;
			for (int j = 0; j < test_num; ++j) {
				vector<vec3f> points;
				float theta = (rand() % 180) * PI / 180;
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
			ratatetheta.push_back(theta_of_streamline);
			After_Rotate.push_back(v);
		}
		vector<float> test_data;
		vector<float> test_label;
		for (int i = 0; i < After_Rotate.size(); ++i) {
			vector<vector<vec3f>> oneStreamlinePointPool = After_Rotate[i];
			for (int j = 0; j < oneStreamlinePointPool.size(); ++j) {
				vector<vec3f> StreamlinePointData1 = oneStreamlinePointPool[j];
				for (int k = j + 1; k < oneStreamlinePointPool.size(); ++k) {
					vector<vec3f> StreamlinePointData2 = oneStreamlinePointPool[k];
					//数据1
					for (int r = 0; r < StreamlinePointData1.size(); ++r) {
						test_data.push_back(StreamlinePointData1[r].x);
						test_data.push_back(StreamlinePointData1[r].y);
						test_data.push_back(StreamlinePointData1[r].z);
					}
					//数据2
					for (int r = 0; r < StreamlinePointData2.size(); ++r) {
						test_data.push_back(StreamlinePointData2[r].x);
						test_data.push_back(StreamlinePointData2[r].y);
						test_data.push_back(StreamlinePointData2[r].z);
					}
					float currentLabel = abs(ratatetheta[i][j] - ratatetheta[i][k]);
					test_label.push_back(currentLabel);
				}
			}
		}
		write_array(test_data.data(), test_data.size(), datasetpath.c_str());
		write_array(test_label.data(), test_label.size(), labelpath.c_str());
	}

	// 计算流线上的点绕端点连线旋转theta角的坐标
	void computeRatate(vector<vector<vector<vec3f>>>& After_Rotate, StreamlinePool Pool, int theta_interval) {
		//const float PI = 3.141592;
		for (int i = 0; i < Pool.streamlines.size(); ++i) {
			const Streamline& s = Pool.streamlines[i];
			vec3f start = Pool.points[s.start];
			vec3f end = Pool.points[s.start + s.numPoint - 1];
			// 一条流线旋转得到的流线
			vector<vector<vec3f>> v;
			for (int j = 0; j < 360; j = j + theta_interval) {
				// 一条流线上的点
				vector<vec3f> points;
				float theta = j * PI / 180;
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
			After_Rotate.push_back(v);
		}
	}
	
	// 流线上的点归一化
	void computeRatate_normalization(vector<vector<vector<vec3f>>>& After_Rotate, StreamlinePool Pool, int train_num) {
		// const float PI = 3.141592;
		vector<vector<float>>ratatetheta;
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
		// 旋转
		for (int i = 0; i < Pool.streamlines.size(); ++i) {
			const Streamline& s = Pool.streamlines[i];
			vec3f start = Pool.points[s.start];
			vec3f end = Pool.points[s.start + s.numPoint - 1];
			//一条流线旋转得到的流线
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
			ratatetheta.push_back(theta_of_streamline);
			After_Rotate.push_back(v);
		}
		vector<float> test_data;
		vector<float> test_label;
		for (int i = 0; i < After_Rotate.size(); ++i) {
			vector<vector<vec3f>> oneStreamlinePointPool = After_Rotate[i];
			for (int j = 0; j < oneStreamlinePointPool.size(); ++j) {
				vector<vec3f> StreamlinePointData1 = oneStreamlinePointPool[j];
				for (int k = j + 1; k < oneStreamlinePointPool.size(); ++k) {
					vector<vec3f> StreamlinePointData2 = oneStreamlinePointPool[k];
					//数据1
					for (int r = 0; r < StreamlinePointData1.size(); ++r) {
						test_data.push_back(StreamlinePointData1[r].x);
						test_data.push_back(StreamlinePointData1[r].y);
						test_data.push_back(StreamlinePointData1[r].z);
					}
					//数据2
					for (int r = 0; r < StreamlinePointData2.size(); ++r) {
						test_data.push_back(StreamlinePointData2[r].x);
						test_data.push_back(StreamlinePointData2[r].y);
						test_data.push_back(StreamlinePointData2[r].z);
					}
					float currentLabel = abs(ratatetheta[i][j] - ratatetheta[i][k]);
					test_label.push_back(currentLabel);
				}
			}
		}
		write_array(test_data.data(), test_data.size(), theta_dataset_path_test.c_str());
		write_array(test_label.data(), test_label.size(), theta_label_path_test.c_str());
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

	// 流线的起始点和终止点
	void get_startpoint_and_endpoint(StreamlinePool Pool, vector<vec3f>& StartAndEnd, vector<float>& StartAndEnd_coordinate) {
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
		write_array(StartAndEnd_coordinate.data(), StartAndEnd_coordinate.size(), start_and_end_path.c_str());
	}
	
	// 每条流线采样相同的点数sample_numpoint
	void getEqualPointStreamlinePool(StreamlinePool& Pool, const int& sample_numpoint) {
		StreamlinePool tempPool = Pool;
		Pool.points.clear();
		Pool.streamlines.clear();
		Pool.line_ids.clear();
		Pool.streamlines.reserve(tempPool.streamlines.size());
		for (int i = 0; i < tempPool.streamlines.size(); ++i) {
			const Streamline& s = tempPool.streamlines[i];
			Streamline new_line = makeStreamline(0, Pool.points.size(), 0);
			// 采样相同的点数
			float interval = s.numPoint * 1.0 / sample_numpoint;
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
	}
	void getRandomEqualPointStreamlinePool(StreamlinePool& Pool, const int& sample_numpoint) {
		StreamlinePool tmpPool = Pool;
		Pool.points.clear();
		Pool.streamlines.clear();
		Pool.line_ids.clear();
		Pool.streamlines.reserve(tmpPool.streamlines.size());
		for (int i = 0; i < tmpPool.streamlines.size(); ++i) {
			const Streamline& s = tmpPool.streamlines[i];
			Streamline new_line = makeStreamline(0, Pool.points.size(), 0);
			// 随机采样相同的点数
			// 随机取流线的总长（随机点数），保证numPoint>=sample_numpoint
			int num = sample_numpoint + rand() % (s.numPoint - sample_numpoint + 1);
			// 确定随机起点: num个点数的滑动窗口
			int begin = rand() % (s.numPoint - num + 1);
			float interval = num * 1.0 / sample_numpoint;
			for (int j = 0; j < sample_numpoint; ++j) {
				if (j != sample_numpoint) {
					int position = floor(begin + j * interval);
					Pool.points.push_back(tmpPool.points[position]);
				}
				else {
					Pool.points.push_back(tmpPool.points[begin + sample_numpoint - 1]);
				}
			}
			// 流线上点的数量
			new_line.numPoint = Pool.points.size() - new_line.start;
			// 每个点所属的流线id
			Pool.line_ids.insert(Pool.line_ids.end(), new_line.numPoint, i);
			// 将新构造的流线加入流线池
			Pool.streamlines.push_back(new_line);
		}
	}
	void resample(StreamlinePool& Pool, const float& thresh) {
		StreamlinePool tempPool = Pool;
		Pool.points.clear();
		Pool.streamlines.clear();
		Pool.line_ids.clear();
		Pool.streamlines.reserve(tempPool.streamlines.size());
		for (int i = 0; i < tempPool.streamlines.size(); ++i) {
			const Streamline& s = tempPool.streamlines[i];
			Streamline resample_line = makeStreamline(0, Pool.points.size(), 0);
			vec3f* line_points = tempPool.points.data() + s.start;
			arc_resample(line_points, s.numPoint, thresh, Pool.points);
			// 计算当前流线上的点的个数
			resample_line.numPoint = Pool.points.size() - resample_line.start;
			// 保存流线上的点的line_ids
			Pool.line_ids.insert(Pool.line_ids.end(), resample_line.numPoint, i);
			// 将流线加入采样流线池
			Pool.streamlines.push_back(resample_line);
		}
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
	void toresample(MultiPool<float>&tmpResamplePool, const float& thresh, std::vector<vec3f>& sample_points) {
		Pool<float>& resample_pool = tmpResamplePool.resample_pools[(int)arc_length][thresh];
		resample_pool.org_pool = tmpResamplePool.pool;
		resample_pool.streamlines.reserve(tmpResamplePool.pool->streamlines.size());
		for (int i = 0; i < tmpResamplePool.pool->streamlines.size(); ++i) {
			const Streamline& s = tmpResamplePool.pool->streamlines[i];
			//初始化单条流线，sid=0,start=resample_pool.points.size(),numPoint=0
			Streamline resample_line = makeStreamline(0, resample_pool.points.size(), 0);
			 //流线第一个点的指针
			vec3f* line_points = tmpResamplePool.pool->points.data() + s.start;
			//开始弧长采样
			arc_resample(resample_pool.points, line_points, s.numPoint, thresh, sample_points);
			//计算当前流线上的点的个数
			resample_line.numPoint = resample_pool.points.size() - resample_line.start;
			//保存流线上的点的line_ids
			resample_pool.line_ids.insert(resample_pool.line_ids.end(), resample_line.numPoint, i);
			//将流线加入采样流线池
			
			resample_pool.streamlines.push_back(resample_line);
		}
	}
	void arc_resample(std::vector<float>& ret,
		vec3f* points, const int& num, const float& thresh, std::vector<vec3f>&sample_points)
	{
		std::vector<float> line_resample_indices;
		std::vector<vec3f> &line_resample_points = sample_points;
		line_resample_indices.reserve(num);
		line_resample_indices.push_back(0.0f);
		line_resample_points.push_back(points[0]);
		vec3f p = points[0];
		float di1 = 0.0f, di, fac;
		for (int i = 1; i < num; ++i) {
			di = length(p - points[i]);
			if (di > thresh) {
				fac = interpolate(di1, di, thresh, 0.0f, 1.0f);
				p = interpolate(points[i - 1], points[i], fac);
				line_resample_points.push_back(p);
				line_resample_indices.push_back(i - 1 + fac);
				di1 = 0.0f;
			}
			else {
				di1 = di;
			}
		}
		ret.insert(ret.end(), line_resample_indices.begin(), line_resample_indices.end());
	}
	
	void allocateData(StreamlinePool& mPool) {
		sample_steps_num = sample_steps.size();
		filter_scales_num = filter_scales.size();
		resample_scales_num = resample_scales.size();
		latent_feature_num = sample_steps_num + filter_scales_num + resample_scales_num;
		if (curvature_sample_rate > 1e-10) ++latent_feature_num;
		latent_feature_full_dim = latent_feature_dim * latent_feature_num;
		int data_size = 0;
		for (int i = 0; i < mPool.streamlines.size(); ++i) {
			//n为流线上点的数量
			int n = mPool.streamlines[i].numPoint;
			//所有流线邻接矩阵大小
			data_size += n * n;
		}
		int num_points = mPool.points.size();
		int num_lines = mPool.streamlines.size();
		mDistData.resize(data_size);
		mDegreeData.resize((num_points)*dist_thresh_num);
		mLatentDisplayData.reserve(16 * latent_feature_full_dim);//should be large enough
		allocateVisGraphs(mPool);
	}
	void allocateVisGraphs(StreamlinePool& mPool) {
		int data_offset = 0;
		mGraphs.resize(mPool.streamlines.size());
		for (int i = 0; i < mPool.streamlines.size(); ++i) {
			const Streamline& s = mPool.streamlines[i];
			//n为流线上点的数目
			int n = s.numPoint;
			//create graphs
			//每条streamline对应一个mGraph,其中data_offset表示给条流线和最开始数据尺度(最开始为0)上的间隔，其大小为累计的流线的邻接矩阵的大小之和
			int latent_dim = (mLatentDisplayMode == latent_raw_data) ? (latent_feature_full_dim) : latent_feature_num;
			mGraphs[i] = new VisibilityGraph(n, data_offset, &mDistData[data_offset], &mDegreeData[s.start * dist_thresh_num], dist_thresh_num, mLatentDisplayData.data(), latent_dim);

			data_offset += n * n;
		}
	}
	void computeAndSaveCurvatures(StreamlinePool& mPool, const StreamlineTraceParameter& pars,
		cudaStreamlineTracer* tracer, const string& curvature_path, vector<float>& ret_cur) {
		vector<vec3f>seeds;
		mPool.getSeeds(seeds);
		tracer->getAccCurvature(ret_cur, seeds, pars);
		//write_array(mCurvatures.data(), mCurvatures.size(), curvature_path.c_str());
	}
	void computeAndSaveDistanceMatrix(StreamlinePool& mPool, cudaStreamlineRenderer* mRenderer) {
		nTimer.start();
		printf("Timing: Computing all Visibility Graphs.");
		std::vector<VisGraphCompTask> tasks;
		tasks.reserve((mDistData.size() - mPool.points.size()) / 2);
		VisGraphCompTask t;
		//create tasks
		//把streamline上邻接矩阵上的每个节点作为VisGraphCompTask
		int data_offset, n;
		for (int i = 0; i < mPool.streamlines.size(); ++i) {
			const Streamline& s = mPool.streamlines[i];
			n = s.numPoint;
			data_offset = getVisGraphOfStreamline(i)->matrix_offset;
			for (int j = 0; j < n; ++j) {
				t.uid = s.start + j;
				//t-uid表示当前计算的是第i条流线的第j个点
				for (int k = j + 1; k < n; ++k) {
					//t-vid表示计算的是邻接矩阵上第j行第k列
					//t-dist_loc表示其在所有邻接矩阵上的数据位置
					t.vid = s.start + k;
					t.dist_loc = data_offset + j * n + k;
					tasks.push_back(t);
				}
			}
		}
		//compute the first half of the matrix visibility graph
		//distance matrix保存到mDistData中
		computeVisibiltyGraphs_h(mDistData.data(), tasks.data(), mRenderer->getPoints_d(), mDistData.size(), 16, tasks.size());

		//fill the other half of the matricies
		//另一半直接对称
		for (int i = 0; i < mGraphs.size(); ++i) {
			float** dist_mat = mGraphs[i]->adj_mat.getMatrixPointer();
			int n = mGraphs[i]->n;
			for (int j = 0; j < n; ++j) {
				dist_mat[j][j] = 0.0f;
				for (int k = 0; k < j; ++k) {
					dist_mat[j][k] = dist_mat[k][j];
				}
			}
		}
		PRINT_TIME(" Finish in %5.3f.\n\n", nTimer.end());
		//write_array(mDistData.data(), mDistData.size(), distance_matrix_path.c_str());
	}
	bool toSampleCurvature() {
		return (curvature_sample_rate > 1e-10);
	}
	VisibilityGraph* getVisGraphOfStreamline(const int& streamline_id) {
		return mGraphs[streamline_id];
	}
	void compute(vector<int>& theta, vector<float> one_streamline_x, vector<float> one_streamline_y) {
		const int half = 180;
		const double pi = 3.141592653;
		int theta_for_max_variance = 0;
		float max_variance = 0.0f;
		for (int d = 0; d < 2 * half; d += 2) {
			vector<float> f1, f2;
			for (int j = 0; j < one_streamline_x.size(); j++) {
				f1.push_back(one_streamline_x[j] * cos(d * pi / half) + one_streamline_y[j] * sin(d * pi / half));
				f2.push_back(-one_streamline_x[j] * sin(d * pi / half) + one_streamline_y[j] * cos(d * pi / half));
			}
			float expectation = 0.0f;
			for (int i = 0; i < f1.size(); i++) {
				expectation += f1[i];
			}
			expectation /= f1.size() * 1.0f;
			float variance = 0.0f;
			for (int j = 0; j < f1.size(); j++) {
				variance += (f1[j] - expectation) * (f1[j] - expectation);
			}
			if (variance > max_variance) {
				max_variance = variance;
				theta_for_max_variance = d;
			}
		}
		float k = max_variance;
		theta.push_back(theta_for_max_variance);
	}
	void PCA(vector<float>points_data, vector<int>& theta, int sample_numpoint) {
		vector<float> streamline_x;
		vector<float> streamline_y;
		for (int i = 0, j = 1; i < points_data.size() && j < points_data.size(); i += 3, j += 3) {
			streamline_x.push_back(points_data[i]);
			streamline_y.push_back(points_data[j]);
		}
		const double pi = 3.141592653;
		const int half = 180;
		int index = 0;
		while (index < streamline_x.size()) {
			vector<float> one_streamline_x, one_streamline_y;
			for (int i = 0; i < sample_numpoint; i++) {
				one_streamline_x.push_back(streamline_x[i + index]);
				one_streamline_y.push_back(streamline_y[i + index]);
			}
			index += sample_numpoint;
			compute(theta, one_streamline_x, one_streamline_y);
		}
	}



	WindowsTimer nTimer;
	vector<StreamlinePool>  ResamplePool;
	StreamlinePool VelocityFromBinormalPool;


	vector<float>mDistData;
	vector<VisibilityGraph*>mGraphs;
	vector<float> mDegreeData;
	vector<float> mLatentDisplayData;
	std::vector<float> mCurvatures; //for MultiPool
	MultiPool<float>mmp;
	cudaStreamlineTracer* mTracer;
	vec3i mDim;
	//cudaStreamlineTracer* mBinormalTracer;
	//StreamlineTraceParameter mVelocityPars;


	float thresh = 1.0f;

	string curvature_path = "curvature.dat";
	string streamline_path = "streamline.stl";
	string all_vg_path = "visgraph-all.dat";
	string vis_graph_path = "./data/visibility_graph.dat";
	string left_or_right_path = "./data/left_or_right.dat";
	string points_data_path = "./data/points.dat";
	string dataset_path = "./data/dataset.dat";
	string theta_dataset_path = "./data/theta/test/dataset/two_swirl.dat";
	string theta_label_path = "./data/theta/test/label/two_swirl.dat";
	string theta_dataset_path_test = "./data/theta_dataset_test_normalization1.dat";
	string theta_label_path_test = "./data/theta_label_test_normalization1.dat";
	string vec_hdr_path = "e:/data/flow/abc.hdr";
	string vec_field_path = "e:/data/flow/abc.vec";
	string pool_path = "e:/data/flow/abc_streamline.stl";

	string distance_matrix_path = "./data/stl_dataset/plume_distance_matrix.dat";
	string start_and_end_path = "./data/stl_dataset/plume_start_and_end.dat";
	string points_path = "./data/stl_dataset/plume_points.dat";
	string label_path = "./data/stl_dataset/plume_label.dat";

	/*
	"resample scales": [4.0, 8.0],
	"sample steps": [1.0, 2.0, 3.0, 4.0],
	"filter scales": [2.0],
	*/
	vector<float> sample_steps = { 1.0 };
	vector<float> filter_scales = { 2.0 };
	vector<float> resample_scales = { 1.0 };

	//VisGraphMetaInfo
	float radius = 1.0;
	float segment_length = 0.1;
	float curvature_sample_rate = 0.0;
	int latent_feature_dim = 128;
	int sample_steps_num;
	int filter_scales_num;
	int resample_scales_num;
	int latent_feature_num;
	int latent_feature_full_dim;
	int dist_thresh_num = 32;
	int sample_numpoint = 32;
	int theta_interval = 5;
	
	LatentDisplayMode mLatentDisplayMode;
};