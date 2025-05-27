#pragma once
//#include"Communication.h"
#include "WebSocketClient.h"
#include <cudaStreamlineRenderer.h>
#include <iostream>
#include <string>
#include "DataUser.h"
#include "DataManager.h"
#include "MessageCenter.h"
#include <algorithm>
#include <unordered_set>
#include <cmath>
#include <QDir>
#include <QCoreApplication>
#include <ctime>
using namespace std;

#define M_PI (3.1415926535)
class BrainDataManager : public DataUser

{
public:
	BrainDataManager(string name="Brain Data") {

		//processAllData();
	
		//getConnectiveFibersFeatures();
		//calculateAndSaveGeometricFeature("057_S_4909");
		
		//Filter();
		
		setData();
		
		//make_fibers(160);

	}
	void setData() {
		
		string path = streamline_path + mSubject + ".stl";
		//string path = "D:/DATA/brain/display_fibers.stl";
		readBrainStreamlinePool(mPool, path.c_str());
		readROIFile(mSubject);

		mDim = makeVec3i(250, 200, 100);
		int render_bottleneck = 30000;
		int render_number =  mPool.streamlines.size() < render_bottleneck ? mPool.streamlines.size() : render_bottleneck;
		mRenderer = new cudaStreamlineRenderer(mPool.streamlines.data(), mPool.points.data(), render_number); //mPool.streamlines.size()
		
	}

	void setSubject(string sub) { mSubject = sub; }
	void setROI(string roi) { mCurrent_ROI = roi; }

	void setROIs(vector<int> array) {
		mROIs.clear();
		mROIs.reserve(array.size());
		for (auto item : array) {
			mROIs.push_back(item + 1);
		}

		unordered_set<int> mROIs_set(mROIs.begin(), mROIs.end());
		//for region color
		/*processColorMask(mROIs_set);
		processSegmentMask();*/
		

	}

	void setEdges(vector<vector<int>> Edges) {
		mEdges.clear();
		mEdges = Edges;

		//mEdges.erase(mEdges.begin() + 0);


		processFibers();

	}

	
	void getConnectiveFibersFeatures() {
		std::ifstream file("D:/DATA/brain/subject_name.txt");

		if (file.is_open()) {
			std::string line;
			while (getline(file, line)) {
				Subject_names.push_back(line);
			}
			file.close();
		}
		else {
			std::cout << "Unable to open the subject_name file" << std::endl;
		}

		for (auto subject : Subject_names) {
			string path = "D:/DATA/brain/original_data/streamline_file/" + subject + ".stl";

			string file_path = "D:/DATA/brain/geometric_features/" + subject + "/";
			QDir dir;
			QString directoryPath = QString::fromStdString(file_path);
			//if (dir.exists(directoryPath)) { continue; }

			readBrainStreamlinePool(mPool, path.c_str());
			useOriginalData = true;
			readROIFile(subject);
			useOriginalData = false;
			calculateAndSaveGeometricFeature(subject);
		}

	}

	struct ZeroRange {
		int start;
		int end;
		int leftNonZero; // ���ķ���Ԫ��
		int rightNonZero; // �Ҳ�ķ���Ԫ��
	};
	vector<ZeroRange> findZeroRanges(const vector<int>& input) {
		vector<ZeroRange> zeroRanges; // �洢����0��Χ�Լ�������������Ԫ��

		int start = -1; // ����0����ʼ����
		int leftNonZero = -1; // ���ķ���Ԫ��
		for (int i = 0; i < input.size(); i++) {
			if (input[i] == 0) {
				if (start == -1) {
					start = i; // ������ʼ����
				}
			}
			else {
				if (start != -1) {
					ZeroRange range;
					range.start = start;
					range.end = i - 1;
					range.leftNonZero = leftNonZero;
					range.rightNonZero = input[i];
					zeroRanges.push_back(range); // ����Χ��������������Ԫ����ӵ������
					start = -1; // ������ʼ����
				}
				leftNonZero = input[i]; // �������ķ���Ԫ��
			}
		}

		// ��������0ֱ��������������
		if (start != -1) {
			ZeroRange range;
			range.start = start;
			range.end = input.size() - 1;
			range.leftNonZero = leftNonZero;
			range.rightNonZero = -1; // ĩβ��0û���Ҳ�ķ���Ԫ��
			zeroRanges.push_back(range);
		}

		return zeroRanges;
	}
	vector<ZeroRange> findSingleZero(const vector<int>& input) {
		std::vector<ZeroRange> result;
		int n = input.size();

		for (int i = 0; i < n; ++i) {
			if (input[i] == 0 && i > 0 && i < n - 1 && input[i - 1] != 0 && input[i + 1] != 0 && input[i - 1] != input[i + 1]) {
				// Single zero with different non-zero neighbors
				ZeroRange range;
				range.start = i - 1;
				range.end = i + 1;
				range.leftNonZero = input[i - 1];
				range.rightNonZero = input[i + 1];
				result.push_back(range);
			}
			else if (i > 0 && input[i] != 0 && input[i - 1] != 0 && input[i] != input[i - 1]) {
				// Different non-zero adjacent elements
				ZeroRange range;
				range.start = i - 1;
				range.end = i;
				range.leftNonZero = input[i - 1];
				range.rightNonZero = input[i];
				result.push_back(range);
			}
		}

		return result;
	}
	vector<ZeroRange> findNonAdjacentNonZeroRanges(const vector<int>& input) {
		vector<ZeroRange> result;
		int n = input.size();

		for (int i = 0; i < n; ++i) {
			if (input[i] != 0) {  // �ҵ���һ������Ԫ��
				for (int j = i + 2; j < n; ++j) { // ��i����һ����ֱ������λ�ÿ�ʼѰ��
					if (input[j] != 0 && input[j] != input[i]) { // �ҵ��ڶ�������Ԫ��

						// �����߶�
						ZeroRange range;
						range.start = i;
						range.end = j;
						range.leftNonZero = input[i];
						range.rightNonZero = input[j];
						result.push_back(range);
						// �������Ҫ����ÿ������Ԫ��֮��ĵ�һ������Ԫ����ɵ��߶Σ�
						// ���������break;�������ڲ�ѭ��
					}
				}
			}
		}

		return result;
	}
	void calculateAndSaveGeometricFeature(string subject) {
		cout << "Processing :" << subject << "...";
		std::clock_t start = std::clock();
		vector<Feature> connection[71][71];
		int idx_row = 0;
		for (auto& row : mROI_MAP) {
			bool allZeros = all_of(row.begin(), row.end(), [](int num) {return num == 0; });
			if (!allZeros) {
				vector<ZeroRange> zeroRanges = findZeroRanges(row);
				vector<ZeroRange> tmp = findSingleZero(row);
				vector<ZeroRange> tmp2=findNonAdjacentNonZeroRanges(row);

				zeroRanges.insert(zeroRanges.end(), tmp.begin(), tmp.end());
				zeroRanges.insert(zeroRanges.end(), tmp2.begin(), tmp2.end());
				Streamline stl = mPool.streamlines[idx_row];
				for (auto zeroRange : zeroRanges) {
					if (zeroRange.leftNonZero == -1 || zeroRange.rightNonZero == -1) continue;
					if (zeroRange.leftNonZero < zeroRange.rightNonZero) {
						connection[zeroRange.leftNonZero][zeroRange.rightNonZero].push_back(getGeometricFeature(makeStreamlineSegment(stl.sid, stl.start+zeroRange.start, stl.start + zeroRange.end)));
					}
					else {
						connection[zeroRange.rightNonZero][zeroRange.leftNonZero].push_back(getGeometricFeature(makeStreamlineSegment(stl.sid, stl.start + zeroRange.start, stl.start + zeroRange.end)));
					}
				}

			}
			idx_row++;
		}
		std::clock_t end = std::clock();
		double duration = (double)(end - start) / CLOCKS_PER_SEC;
		string file_path = "D:/DATA/brain/geometric_features/" + subject+"/";
		QDir dir;
		QString directoryPath = QString::fromStdString(file_path);
		if (!dir.exists(directoryPath)) { dir.mkpath(directoryPath); }

		cout << "Start saving: " << subject<<endl;

		for (int i = 1; i < 71; i++) {
			for (int j = i + 1; j < 71; j++) {
				vector<Feature> features = connection[i][j];
				if (features.empty()) continue;

				ofstream outputFile(file_path+to_string(i)+"_"+ to_string(j)+".dat", std::ios::binary);
				
				if (outputFile.is_open()) {
					// ������ά���鲢������д���ļ�
					for (auto feature : features) {
						if (checkFeature(feature)) { break; }
						outputFile << feature.length << " ";
						outputFile << feature.curvature << " ";
						outputFile << feature.torsion << " ";
						outputFile << feature.tortuosity << " ";
						outputFile << feature.entropy << " ";
						outputFile << "\n"; // ��ÿ��ĩβ��ӻ��з�
					}

					// �ر��ļ�
					outputFile.close();
					
				}
				else {
					std::cerr << "saving error  \n";
				}
			}
		}
		

	}

	/*calculate geometric feature*/
	float calculateLength(const StreamlineSegment seg) {
		vec3f* p1 = &mPool.points[seg.segment.lower];
		vec3f* p2 = &mPool.points[seg.segment.lower + 1];
		float dist = 0;
		while (p2 <= &mPool.points[seg.segment.upper]) {
			dist += dist3d(*p1, *p2);
			p1++, p2++;
		}
		return dist;
	}

	vec3f crossProduct(const vec3f v1, const vec3f v2) {
		vec3f result;
		result.x = v1.y * v2.z - v1.z * v2.y;
		result.y = v1.z * v2.x - v1.x * v2.z;
		result.z = v1.x * v2.y - v1.y * v2.x;
		return result;
	}

	float calculateCurvature(const StreamlineSegment seg) {
		vec3f* p = &mPool.points[seg.segment.lower+1];
		vector<float> curvatures;
		while (p < &mPool.points[seg.segment.upper]) {
			vec3f* p1 = p - 1;
			vec3f* p2 = p + 1;
			
			vec3f AB = (*p) - (*p1);
			vec3f BC = (*p2) - (*p);
			

			// ���� AB �� BC �����Ĳ��
			vec3f crossProduct = {
				AB.y * BC.z - AB.z * BC.y,
				AB.z * BC.x - AB.x * BC.z,
				AB.x * BC.y - AB.y * BC.x
			};

			// ���� AB �� BC ������ģ
			float ABLength = dist3d((*p1), (*p));
			float BCLength = dist3d((*p), (*p2));
			float crossProductLength = std::sqrt(crossProduct.x * crossProduct.x + crossProduct.y * crossProduct.y + crossProduct.z * crossProduct.z);

			// �������ʲ���ӵ�����������
			if (crossProductLength > 0.0f) {
				float epsilon = 1e-6;
				float curvatureValue = 2.0f * crossProductLength / (ABLength * BCLength * (ABLength + BCLength)+epsilon);
				curvatures.push_back(curvatureValue);
			}
			else {
				curvatures.push_back(0.0f); // ������Ϊ�㣬������Ϊ��
			}

			p++;
		}
		if (curvatures.empty()) { return 0; }
		return accumulate(curvatures.begin(), curvatures.end(), 0.0f) / curvatures.size();
	}

	float calculateTorsion(const StreamlineSegment seg) {
		vec3f* A = &mPool.points[seg.segment.lower];
		vec3f* B = &mPool.points[seg.segment.lower+1];
		vec3f* C = &mPool.points[seg.segment.lower + 2];
		vec3f* D = &mPool.points[seg.segment.lower + 3];
		vector<float> torsions;
		while (D <= &mPool.points[seg.segment.upper]) {
			// �������� AB, BC, CD
			vec3f AB = (*A) - (*B);
			vec3f BC = (*B) - (*C);
			vec3f CD = (*C) - (*D);

			// ��������������
			vec3f N1 = crossProduct(AB, BC);
			vec3f N2 = crossProduct(BC, CD);
			
			// ����Ť���Ƕȣ����ȣ�
			float torsionAngle = atan2f(N2.y, N2.x) - atan2f(N1.y, N1.x);

			// ������ת��Ϊ�Ƕ�
			torsionAngle *= 180.0f / M_PI;

			torsions.push_back(torsionAngle);

			A++, B++, C++, D++;

		}
		if (torsions.empty()) { return 0; }
		float res= accumulate(torsions.begin(), torsions.end(), 0.0f) / torsions.size();
		return abs(res);
	}

	float calculateTortuosity(const StreamlineSegment seg) {
		float s = calculateLength(seg);
		float d = dist3d(mPool.points[seg.segment.lower], mPool.points[seg.segment.upper]);
		float epsilon = 1e-6;
		return s / (d + epsilon);
	}

	float calculateEntropy(const StreamlineSegment seg) {
		float entropy = 0.0;
		int totalCount = seg.segment.upper - seg.segment.lower + 1;

		// �����֮��ľ��벢ͳ��
		vector<float> distances;
		vec3f* p1 = &mPool.points[seg.segment.lower];
		vec3f* p2 = &mPool.points[seg.segment.lower+1];
		while (p1 <= &mPool.points[seg.segment.upper]) {
			while (p2 <= &mPool.points[seg.segment.upper]) {
				distances.push_back(dist3d(*p1, *p2));
				p2++;
			}
			p1++;
		}

		// ����������ݵ���
		for (float distance : distances) {
			float probability = 1.0 / (1.0 + distance); // һ�ּ򵥵ľ��뵽���ʵ�ӳ��
			entropy -= probability * std::log2(probability);
		}

		return entropy;
	}

	struct Feature
	{
		float length, curvature, torsion, tortuosity, entropy;
	};
	Feature getGeometricFeature(const StreamlineSegment seg) {
		Feature ret;
		ret.length = calculateLength(seg);
		ret.curvature = calculateCurvature(seg);
		ret.torsion = calculateTorsion(seg);
		ret.tortuosity = calculateTortuosity(seg);
		ret.entropy = calculateEntropy(seg);
		return ret;
	}
	bool checkFeature(const Feature& feature) {
		return (feature.length == 0.0f) &&
			(feature.curvature == 0.0f) &&
			(feature.torsion == 0.0f) &&
			(feature.tortuosity == 0.0f) &&
			(feature.entropy == 0.0f);
	}

	//for update segments color
	void clearROIInvolved() {
		mROIs.clear();
		mSegMask.clear();
		mColorMask.clear();
	}
	struct Segment {
		int start;
		int end;
	};
	std::vector<Segment> findContiguousSegments(const std::vector<int>& numbers) {
		std::vector<Segment> contiguousSegments;

		int currentStart = -1;

		for (int i = 0; i < numbers.size(); ++i) {
			if (numbers[i] != 0) {
				if (currentStart == -1) {
					currentStart = i;
				}
			}
			else {
				if (currentStart != -1) {
					Segment segment = { currentStart, i - 1 };
					contiguousSegments.push_back(segment);
					currentStart = -1;
				}
			}
		}

		if (currentStart != -1) {
			Segment segment = { currentStart, static_cast<int>(numbers.size()) - 1 };
			contiguousSegments.push_back(segment);
		}

		return contiguousSegments;
	}

	void processSegmentMask() {
		mSegMask.clear();
		int idx = 0;

		for (auto streamline : mColorMask) {
			
			bool allZeros = all_of(streamline.begin(), streamline.end(), [](int num) {return num == 0; });
			if (!allZeros) {
				vector<Segment> segments=findContiguousSegments(streamline);
				for (auto seg : segments) {
					mSegMask.push_back(makeStreamlineSegment(idx, seg.start, seg.end));
				}

			}
			idx++;
			
		}
	}

	void processColorMask(unordered_set<int> set) {
		mColorMask.clear();
		mColorMask = mROI_MAP;

		for (auto& streamline : mColorMask) {
			bool allZeros = all_of(streamline.begin(), streamline.end(), [](int num) {return num == 0;});
			if (!allZeros) {
				for (auto& searchElement : streamline) {
					if (set.find(searchElement) == set.end()) {
						searchElement = 0;
					}
				}

			}
		}

	}

	bool containsEdge(const std::vector<std::vector<int>>& mEdges, const std::vector<int>& edge) {
		return std::find(mEdges.begin(), mEdges.end(), edge) != mEdges.end();
	}
	void processFibers() { //mEdges -> mFibers : fill arrays of segments for coloring
		mFibers.clear();

		int idx_row = 0;
		for (auto& row : mROI_MAP) {
			bool allZeros = all_of(row.begin(), row.end(), [](int num) {return num == 0; });
			if (!allZeros) {
				vector<ZeroRange> zeroRanges = findZeroRanges(row);
				vector<ZeroRange> tmp = findSingleZero(row);
				vector<ZeroRange> tmp2 = findNonAdjacentNonZeroRanges(row);

				zeroRanges.insert(zeroRanges.end(), tmp.begin(), tmp.end());
				zeroRanges.insert(zeroRanges.end(), tmp2.begin(), tmp2.end());
				Streamline stl = mPool.streamlines[idx_row];
				for (auto zeroRange : zeroRanges) {
					int start = zeroRange.leftNonZero; 
					int end = zeroRange.rightNonZero;
					if (start == -1 || end == -1) continue;
					
					if (containsEdge(mEdges, { start-1,end-1 })) { mFibers.push_back(makeStreamlineSegment(stl.sid, stl.start + zeroRange.start, stl.start + zeroRange.end)); }
					if (containsEdge(mEdges, { end-1,start-1 })) { mFibers.push_back(makeStreamlineSegment(stl.sid, stl.start + zeroRange.start, stl.start + zeroRange.end)); }
				}

			}
			idx_row++;
		}
		

	}

	

	//process data
	void processAllData() {
		std::ifstream file("D:/DATA/brain/subject_name.txt");

		if (file.is_open()) {
			std::string line;
			while (getline(file, line)) {
				Subject_names.push_back(line);
			}
			file.close();
		}
		else {
			std::cout << "Unable to open the file" << std::endl;
		}

		for (auto subject : Subject_names) {
			cout << "Start processing: " << subject << endl;
			processMATData(subject);
		}
	}

	void processMATData(string name) {
		string path = "D:/DATA/brain/original_data/original_file/";
		string file_path = path + name + ".bin";

		std::ifstream file(file_path, std::ios::binary | std::ios::ate);
		if (!file.is_open()) {
			std::cout << "Error opening binary file" << std::endl;
		}

		std::streamsize fileSize = file.tellg();
		file.seekg(0, std::ios::beg);

		// ��ȡ����
		double* data = new double[fileSize / sizeof(double)];
		file.read(reinterpret_cast<char*>(data), fileSize);
		
		//���ñ���
		int stl_num = static_cast<int>(data[0]);
		int sample_num = static_cast<int>(data[1]);
		int points_num = stl_num * sample_num;

		/*Points.resize(points_num);
		for (int i = 0; i < sample_num; ++i) {
			for (int s = 0; s < stl_num; ++s) {
				double* p = &data[(i * stl_num + s) * 3 + 2];
				Points[s * sample_num + i] = makeVec3f(p[0], p[1], p[2]);
			}
		}*/

		Points.clear();
		Streamlines.clear();

		for (int i = 2; i < points_num * 3+2; i = i + 3) {
			float x = static_cast<float>(data[i]);
			float y = static_cast<float>(data[i + 1]);
			float z = static_cast<float>(data[i + 2]);
			Points.push_back(makeVec3f(x, y, z));
		}

		for (int i = 0; i < stl_num; i++) {
			Streamlines.push_back(makeStreamline(i, i * sample_num, sample_num));
		}


		// �ͷ���Դ
		delete[] data;
		file.close();

		//�������ļ�
		ofstream output_file("D:/DATA/brain/original_data/streamline_file/" + name + ".stl", std::ios::binary);

		if (!output_file.is_open()) {
			std::cerr << "Error opening output file" << std::endl;
		}

		output_file.write((char*)(&points_num), sizeof(int));
		output_file.write((char*)(&stl_num), sizeof(int));
		output_file.write((char*)Points.data(), sizeof(vec3f) * points_num);
		output_file.write((char*)Streamlines.data(), sizeof(Streamline) * stl_num);

		output_file.close();
	}

	void processROIFile(string subject) {
		string path = streamline_path + mSubject + ".stl";
		readBrainStreamlinePool(mPool, path.c_str());

		string file_path = "D:/DATA/brain/ROI_MAP/" + subject + ".dat";
		vector<vector<int>> ROI_MAP;
		//mROI_MAP.clear();
		//mROI_MAP.reserve(getNumStreamlines());

		std::ifstream file(file_path, std::ios::binary | std::ios::ate);
		if (!file.is_open()) {
			std::cout << "Error opening ROI file" << std::endl;
		}
		std::streamsize fileSize = file.tellg();
		file.seekg(0, std::ios::beg);
		/*int* data = new int[fileSize / sizeof(int)];
		file.read(reinterpret_cast<char*>(data), fileSize);*/

		std::shared_ptr<int> data(new int[fileSize / sizeof(int)], std::default_delete<int[]>());

		file.read(reinterpret_cast<char*>(data.get()), fileSize);


		int sample_num = getNumPoints() / getNumStreamlines();

		for (int i = 0; i < getNumStreamlines(); i++) {
			vector <int> tmp;
			for (int j = 0; j < sample_num; j++) {
				tmp.push_back(data.get()[i * sample_num + j]);
			}
			ROI_MAP.push_back(tmp);
		}

		file.close();

		//�������ļ�
		ofstream output_file("D:/DATA/brain/ROI_MAP/ROI/" + subject + ".roi", std::ios::binary);

		if (!output_file.is_open()) {
			std::cerr << "Error opening output file" << std::endl;
		}

		output_file.write((char*)ROI_MAP.data(), sizeof(int) * getNumPoints());

		output_file.close();

		
	}

	//Filter the pool 

	template <typename T>
	bool hasConsecutiveZerosBetweenNonZeros(const std::vector<T>& vec) {

		for (size_t i = 0; i < vec.size(); ++i) {

			if (vec[i] != 0) {
				continue; // �����ǰԪ�ز����㣬����
			}

			// �������Ƿ��з���Ԫ��
			if (i == 0 || vec[i - 1] != 0) {
				size_t j = i;
				// ����ɨ��������������
				while (j < vec.size() && vec[j] == 0) {
					j++;
				}

				// ����Ҳ��Ƿ��з���Ԫ��
				if (j < vec.size() && vec[j] != 0) {
					return true; // �ҵ���һ���������㣬�����඼�Ƿ���Ԫ��
				}
			}
		}

		return false; // û���ҵ��������������
	}

	int find_max_length() {
		float maxv=0;
		for (int i = 0; i < getNumStreamlines() - 1; i++) {
			Streamline stl = mPool.streamlines[i];
			float temp;
			temp = calculateLength(makeStreamlineSegment(stl.sid, stl.start, stl.start + stl.numPoint - 1));
			maxv = maxv > temp ? maxv : temp;

		}
		return maxv;
	}

	void Filter() {
		std::ifstream file("D:/DATA/brain/subject_name.txt");

		if (file.is_open()) {
			std::string line;
			while (getline(file, line)) {
				Subject_names.push_back(line);
			}
			file.close();
		}
		else {
			std::cout << "Unable to open the subject_name file" << std::endl;
		}

		for (auto subject : Subject_names) {
			string path = "D:/DATA/brain/original_data/streamline_file/" + subject + ".stl";
			readBrainStreamlinePool(mPool, path.c_str());
			useOriginalData = true;
			readROIFile(subject);
			useOriginalData = false;

			float thresh_ratio = 0.35; //cut off lines #thresh_ratio of max length  
			current_max_length = find_max_length();
			int thresh = floor(thresh_ratio * current_max_length);
			filterStreamlinePool_length(subject, thresh);
		}
	}

	void make_fibers(int threshold) {
		StreamlinePool sampled_pool;
		vector<int> sampledIndices;
		for (int i = 0; i < mPool.streamlines.size(); ++i) {
			Streamline stl = mPool.streamlines[i];
			if (calculateLength(makeStreamlineSegment(stl.sid, stl.start, stl.start + stl.numPoint - 1)) > threshold) {
				
				sampledIndices.push_back(i);
				int current_size = sampled_pool.streamlines.size();
				sampled_pool.streamlines.push_back(stl);
				sampled_pool.streamlines[current_size].start = current_size * sampled_pool.streamlines[current_size].numPoint;
				sampled_pool.points.insert(sampled_pool.points.end(), mPool.points.begin() + stl.start, mPool.points.begin() + stl.start + stl.numPoint);

			}
		}

		sampled_pool.fillLineIds();

		string path = "D:/DATA/brain/";

		int points_num = sampled_pool.points.size();
		int lines_num = sampled_pool.streamlines.size();

		//export streamlinePool
		ofstream output_file(path + "display_fibers" + ".stl", std::ios::binary);
		if (!output_file.is_open()) {
			std::cerr << "Error opening output file" << std::endl;
		}
		output_file.write((char*)(&points_num), sizeof(int));
		output_file.write((char*)(&lines_num), sizeof(int));
		output_file.write((char*)sampled_pool.points.data(), sizeof(vec3f) * points_num);
		output_file.write((char*)sampled_pool.streamlines.data(), sizeof(Streamline) * lines_num);
		output_file.close();

	}

	void filterStreamlinePool_length(string subject, float threshold) {
		cout << "Sampling the pool of :" << subject << "...";
		//sample
		vector<int> sampledIndices;
		StreamlinePool sampled_pool;
		vector<vector<int>> sampled_map;

		for (int i = 0; i < mPool.streamlines.size(); ++i) {
			Streamline stl = mPool.streamlines[i];
			if (calculateLength(makeStreamlineSegment(stl.sid,stl.start,stl.start+stl.numPoint-1)) > threshold) { 
				bool all_zeros = std::all_of(mROI_MAP[i].begin(), mROI_MAP[i].end(), [](int i) { return i == 0; });
				if (all_zeros) continue;
				if (!hasConsecutiveZerosBetweenNonZeros(mROI_MAP[i])) continue;
				sampledIndices.push_back(i); 
				sampled_map.push_back(mROI_MAP[i]);
				int current_size = sampled_pool.streamlines.size();
				sampled_pool.streamlines.push_back(stl);
				sampled_pool.streamlines[current_size].start = current_size * sampled_pool.streamlines[current_size].numPoint;
				sampled_pool.points.insert(sampled_pool.points.end(), mPool.points.begin() + stl.start, mPool.points.begin() + stl.start + stl.numPoint);

			}
		}
		
		sampled_pool.fillLineIds();

		cout << "Saving the pool of :" << subject << endl;
		string path = "D:/DATA/brain/filtered_data/";

		int points_num = sampled_pool.points.size();
		int lines_num= sampled_pool.streamlines.size();

		//export streamlinePool
		ofstream output_file(path+"streamline_file/" + subject + ".stl", std::ios::binary);
		if (!output_file.is_open()) {
			std::cerr << "Error opening output file" << std::endl;
		}
		output_file.write((char*)(&points_num), sizeof(int));
		output_file.write((char*)(&lines_num), sizeof(int));
		output_file.write((char*)sampled_pool.points.data(), sizeof(vec3f) * points_num);
		output_file.write((char*)sampled_pool.streamlines.data(), sizeof(Streamline) * lines_num);
		output_file.close();

		//export ROI_MAP
		ofstream file(path + "ROI_MAP/" + subject + ".dat", std::ios::binary);
		if (!file.is_open()) {
			std::cerr << "Error opening output file" << std::endl;
		}
		for (const auto& row : sampled_map) {
			for (int value : row) {
				file.write(reinterpret_cast<const char*>(&value), sizeof(int));
			}
		}
		file.close();

	}

	//read file
	bool readBrainStreamlinePool(StreamlinePool& ret_pool, const char* file_path) {

		std::ifstream input_file;
		if (!open_file(input_file, file_path, true)) {
			return false;
		}
		ret_pool.line_ids.clear();
		ret_pool.points.clear();
		ret_pool.streamlines.clear();

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

	bool readROIFile(string Subject) {
		string file_path=useOriginalData? "D:/DATA/brain/original_data/ROI_MAP/" + Subject + ".dat": "D:/DATA/brain/filtered_data/ROI_MAP/" + Subject + ".dat";

		mROI_MAP.clear();
		mROI_MAP.reserve(getNumStreamlines());

		std::ifstream file(file_path, std::ios::binary | std::ios::ate);
		if (!file.is_open()) {
			std::cout << "Error opening ROI file" << std::endl;
		}
		std::streamsize fileSize = file.tellg();
		file.seekg(0, std::ios::beg);
		/*int* data = new int[fileSize / sizeof(int)];
		file.read(reinterpret_cast<char*>(data), fileSize);*/

		std::shared_ptr<int> data(new int[fileSize / sizeof(int)], std::default_delete<int[]>());

		file.read(reinterpret_cast<char*>(data.get()), fileSize);


		int sample_num = getNumPoints() / getNumStreamlines();

		for (int i = 0; i < getNumStreamlines(); i++) {
			vector <int> tmp;
			for (int j = 0; j < sample_num; j++) {
				tmp.push_back(data.get()[i * sample_num + j]);
			}
			mROI_MAP.push_back(tmp);
		}

		file.close();
		return true;
	}

	int getNumPoints() {
		return mPool.points.size();
	}

	int getNumStreamlines() {
		return mPool.streamlines.size();
	}
	
	string& getName() { return mName; }

	void onDataItemChanged(const std::string& name) {
		
		
	}

	

	//��Ա����
	bool useOriginalData=false;
	string mName;
	vector<vec3f> Points;
	vector<Streamline> Streamlines;
	StreamlinePool mPool;
	vec3i mDim;//width height depth
	cudaStreamlineRenderer* mRenderer;
	string streamline_path = "D:/DATA/brain/filtered_data/streamline_file/";

	//global parameters
	int mStreamlineId;
	//vector<string> mSubject_names;
	string mSubject= "057_S_4909";
	vector<string> Subject_names;

	vector<vector<int>> mROI_MAP;//ROI MAP
	vector <int> mROIs; //ROI involved
	vector<vector<int>> mEdges;//edges involved 

	string mCurrent_ROI="1: lh-insula";

	//for update segments color
	//region color
	vector <StreamlineSegment> mSegMask;
	vector<vector<int>> mColorMask;
	//fiber color
	vector<StreamlineSegment> mFibers;

	//for filter
	float current_max_length;

	
	


private:
};