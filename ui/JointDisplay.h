#pragma once
#include "typeOperation.h"
#include "ColorMap.h"
#include "MatrixData.h"
#include "DisplayWidget.h"
#include "BarDisplay.h"
#include "MatrixDisplay.h"
#include "LineChartDisplay.h"
#include "ParallelCoordinateDisplay.h"
#include "EmbeddingWidget.h"

class JointDisplay {
public:

	struct Bundle {
		std::string subject;
		int start;
		int end;

		Bundle(std::string sub, int s, int e) {
			subject = sub;
			start = s;
			end = e;
		}
	};

	JointDisplay() {
		mData = NULL;
		mJointData = NULL;
		/*mFlagX = 0;
		mFlagY = 1;*/
		mDataNum = 0;
		mNumBin = 10;
		mCutoffRatio = 0.0f;
		mMargin = 5;
		read_subject_names(mSubject_names);


	}

	void read_subject_names(std::vector<std::string>& mSubject_names) {
		std::ifstream file("D:/DATA/brain/subject_name.txt");

		if (file.is_open()) {
			std::string line;
			while (getline(file, line)) {
				mSubject_names.push_back(line);
			}
			file.close();
		}
		else {
			std::cout << "Unable to open the file" << std::endl;
		}
		mSubject_names.push_back("Cohort");
	}

	bool loadData(int start, int end) {
		string path = "D:/DATA/brain/dti_features/";
		std::string file_path = path + to_string(start)+"_"+ to_string(end) + ".dat";

		dti_data.clear();
		int rows = 5;
		int cols = 198;

		std::ifstream file(file_path, std::ios::binary | std::ios::ate);
		if (!file.is_open()) {
			std::cout << "Error opening DTI DATA file" << std::endl;
		}
		std::streamsize fileSize = file.tellg();
		file.seekg(0, std::ios::beg);

		float* data = new float[fileSize / sizeof(float)];
		file.read(reinterpret_cast<char*>(data), fileSize);

		for (int i = 0; i < rows; i++) {
			std::vector<vec2f> temp;
			for (int j = 0; j < cols; j++) {
				vec2f p = makeVec2f((float)i, (float)data[i * rows + j]);
				temp.push_back(p);
			}
			dti_data.push_back(temp);
		}

		file.close();
		return true;
	}

	bool loadData(std::string path) { 
		delete mData;
		std::ifstream file(path);
		std::vector<std::vector<float>> data;

		if (file.is_open()) {
			std::string line;
			while (getline(file, line)) {
				std::istringstream iss(line);
				std::vector<float> row;
				float value;
				while (iss >> value) {
					row.push_back(value);
				}
				data.push_back(row);
			}
			file.close();
		}
		else {
			printf("Unable to load geometric data file\n");
			return false;
		}

		int height = data.size();
		int width = data.empty() ? 0 : data[0].size();
		float* arrayData = new float[width * height];
		// 将二维向量数据复制到一维数组中
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				arrayData[i * width + j] = data[i][j];
			}
		}

		mData = new MatrixData<float>(width, height, arrayData);
		return true;
	}

	vector<int> histogram2D(const std::vector<float>& data1, const std::vector<float>& data2, int bins) {
		
		std::vector<int> histogram;

		float min1 = *std::min_element(data1.begin(), data1.end());
		float max1 = *std::max_element(data1.begin(), data1.end());
		float min2 = *std::min_element(data2.begin(), data2.end());
		float max2 = *std::max_element(data2.begin(), data2.end());

		float binSize1 = (max1 > min1) ? (max1 - min1) / bins : 1.0f;
		float binSize2 = (max2 > min2) ? (max2 - min2) / bins : 1.0f;

		histogram.resize(bins * bins, 0); 

		for (size_t i = 0; i < data1.size(); i++) {
			int bin1 = std::min(static_cast<int>((data1[i] - min1) / binSize1), bins - 1);
			int bin2 = std::min(static_cast<int>((data2[i] - min2) / binSize2), bins - 1);

			int index = bin1 * bins + bin2; 
			histogram[index] += 1; 
		}

		return histogram;
	}

	Bundle extractDataFromPath(const std::string& path) {
		// 找到subject部分
		size_t subject_start = path.find("geometric_features/") + 19; // "geometric_features/"的长度加上其在路径中的起始位置
		size_t subject_end = path.find('/', subject_start);
		std::string subject = path.substr(subject_start, subject_end - subject_start);

		// 找到整型变量部分
		size_t numbers_start = path.rfind('/') + 1; // 最后一个'/'后面的位置
		size_t dot_position = path.rfind('.');      // '.'的位置
		std::string numbers = path.substr(numbers_start, dot_position - numbers_start);

		// 分割整型变量
		std::istringstream iss(numbers);
		int num1, num2;
		char underscore;
		iss >> num1 >> underscore >> num2;

		return Bundle(subject, num1, num2);
	}

	void display() {
		mMarginalDisplay_X.display();
		mMarginalDisplay_Y.display();
		mJointDisplay.display();
		dti_view.display();
	}

	void setData(std::string path, int flag1, int flag2, std::vector<int> comp_selected) {

		Bundle bundle = extractDataFromPath(path);

		//set display area
		setLayout();

		setFlags(flag1, flag2);
		bool bData=loadData(path);
		if (!bData) { return; }

		std::vector<float> columnData1, columnData2;
		for (int i = 0; i < mData->height(); i++) {
			columnData1.push_back(mData->getValueQuick(mFlagX, i));
			columnData2.push_back(mData->getValueQuick(mFlagY, i));
			
		}

		//************************************ Marginal Distribution ***********************************************
		
		mDataNum = columnData1.size();
		float* data_X = columnData1.data();
		float* data_Y = columnData2.data();

		//histogram assign
		std::vector<int> hist_X;
		std::vector<int> hist_Y;
		Range bound_X = compute_bound(data_X, mDataNum, mCutoffRatio);
		Range bound_Y = compute_bound(data_Y, mDataNum, mCutoffRatio);
		histogram(hist_X, data_X, mDataNum, mNumBin, bound_X, true);
		histogram(hist_Y, data_Y, mDataNum, mNumBin, bound_Y, true);
		std::reverse(hist_Y.begin(), hist_Y.end());

		mMarginalDisplay_X.set_data(hist_X);
		mMarginalDisplay_Y.set_data(hist_Y);
		mMarginalDisplay_X.set_range(MINMAX);
		mMarginalDisplay_Y.set_range(MINMAX);
		mMarginalDisplay_X.set_domain(bound_X);
		mMarginalDisplay_Y.set_domain(bound_Y);


		
		//************************************ Joint Distribution ***********************************************
		vector<int> histogram_data = histogram2D(columnData1, columnData2, mNumBin);
		int* histogram_ptr = histogram_data.data();
		MatrixData<int> histogram_matrix_int(mNumBin, mNumBin, histogram_ptr);
		
		mJointData = histogram_matrix_int.convert<float>();
		
		mJointDisplay.setData(mJointData);
		mJointDisplay.set_range(MINMAX);

		//************************************ DTI View ***********************************************

		loadData(bundle.start, bundle.end);

		dti_view.setData(dti_data, false);
		dti_view.set_range(MINMAX);

		auto it = find(mSubject_names.begin(), mSubject_names.end(), bundle.subject);
		int index = std::distance(mSubject_names.begin(), it);
		dti_view.clear_selected();
		dti_view.set_selected(index);

		dti_view.set_comp_selected(comp_selected);


	}

	void setFlags(int flag1, int flag2) {
		mFlagX = flag1;
		mFlagY = flag2;
	}

	void setBins(const int& bins) { mNumBin = bins;}

	void setArea(const RectDisplayArea& area) {mArea = area;}

	void setLayout() {
		mMarginalDisplay_X.set_area(makeRectDisplayArea(makeVec2f(mArea.origin.x , mArea.origin.y + 200 + mMargin), makeVec2f(200.0f, 0.0f), makeVec2f(0.0f, 120.0f)));
		mMarginalDisplay_Y.set_area(makeRectDisplayArea(makeVec2f(mArea.origin.x + 200 + mMargin, mArea.origin.y + 200 ), makeVec2f(0.0f, -200.0f), makeVec2f(120.0f, 0.0f)));
		mJointDisplay.setArea(makeRectDisplayArea(mArea.origin, makeVec2f(200.0f, 0.0f), makeVec2f(0.0f, 200.0f)));
		dti_view.setArea(makeRectDisplayArea(makeVec2f(mArea.origin.x+460, mArea.origin.y), makeVec2f(500.0f, 0.0f), makeVec2f(0.0f, 300.0f)));
	}

	//private:
	//data
	std::vector<std::string> mSubject_names;

	MatrixData<float>* mData;
	MatrixData<float>* mJointData;
	//int* histogram_ptr;
	std::vector<int> mMarginalData_X , mMarginalData_Y;

	std::vector<std::vector<vec2f>> dti_data;

	//view
	MatrixDisplay<float> mJointDisplay;
	BarDisplay mMarginalDisplay_X, mMarginalDisplay_Y;
	ParallelCoordinateDisplay dti_view;
	

	//variable flag
	int mFlagX , mFlagY;

	//parameters
	int mNumBin;
	int mDataNum;
	float mCutoffRatio;

	//for display layout
	RectDisplayArea mArea;
	float mMargin;

};