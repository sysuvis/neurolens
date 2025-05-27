#pragma once
#include "typeOperation.h"
#include "ColorMap.h"
#include "MatrixData.h"
#include "DisplayWidget.h"
#include "BarDisplay.h"
#include "MatrixDisplay.h"
#include "GraphDisplay.h"
#include "NodetrixDisplay.h"
#include "GlobalDataManager.h"

class EmbeddingDisplay2{
public:

	EmbeddingDisplay2(){
		read_subject_names(mSubject_names);
		selected_left.push_back(0);
		selected_right.push_back(0);
		mLeft.mGraph.setColorScheme(COLOR_MAP_PURPLE_SCALE);
		mRight.mGraph.setColorScheme(COLOR_MAP_YELLOW_SCALE);

		mLeft.pSelectedROIs= &mSelectedROIs;
		mRight.pSelectedROIs = &mSelectedROIs;

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

	bool readConnectionFile(string Subject) {
		string file_path = "D:/DATA/brain/connections/" + Subject + ".adj";

		connection.clear();

		std::ifstream file(file_path, std::ios::binary | std::ios::ate);
		if (!file.is_open()) {
			std::cout << "Error opening connection file" << std::endl;
		}
		std::streamsize fileSize = file.tellg();
		file.seekg(0, std::ios::beg);
		std::shared_ptr<int> data(new int[fileSize / sizeof(int)], std::default_delete<int[]>());
		file.read(reinterpret_cast<char*>(data.get()), fileSize);

		int column = 70;
		int row = 70;

		for (int i = 0; i < row; i++) {
			vector <int> tmp;
			for (int j = 0; j < column; j++) {
				tmp.push_back(data.get()[i * column + j]);
			}
			connection.push_back(tmp);
		}

		file.close();
		return true;
	}

	bool readConnectionFile2(string Subject) {
		string file_path = "D:/DATA/brain/connections/" + Subject + ".adj";

		connection2.clear();

		std::ifstream file(file_path, std::ios::binary | std::ios::ate);
		if (!file.is_open()) {
			std::cout << "Error opening connection file" << std::endl;
		}
		std::streamsize fileSize = file.tellg();
		file.seekg(0, std::ios::beg);
		std::shared_ptr<int> data(new int[fileSize / sizeof(int)], std::default_delete<int[]>());
		file.read(reinterpret_cast<char*>(data.get()), fileSize);

		int column = 70;
		int row = 70;

		for (int i = 0; i < row; i++) {
			vector <int> tmp;
			for (int j = 0; j < column; j++) {
				tmp.push_back(data.get()[i * column + j]);
			}
			connection2.push_back(tmp);
		}

		file.close();
		return true;
	}

	// brain load
	bool read_brain_encoding_file(std::vector<vec2f>& coordinate) {
		std::string file_path = "D:/DATA/brain/matrix_positions.dat";

		coordinate.clear();

		std::ifstream file(file_path, std::ios::binary | std::ios::ate);
		if (!file.is_open()) {
			std::cout << "Error opening matrix_positions file" << std::endl;
		}
		std::streamsize fileSize = file.tellg();
		file.seekg(0, std::ios::beg);

		float* data = new float[fileSize / sizeof(float)];
		file.read(reinterpret_cast<char*>(data), fileSize);

		for (int i = 0; i < 70 * 2; i = i + 2) {
			float x = (float)data[i];
			float y = (float)data[i + 1];
			coordinate.push_back(makeVec2f(x, y));
		}

		file.close();
		return true;
	}

	void load_brain_color(GraphDisplay<float>& mGraph, std::vector<int> left, std::vector<int> right, bool isMean = true) {
		std::vector<float> ret(70);
		std::vector<vec2f> left_data;
		std::vector<vec2f> right_data;
		if (left.size() == 0 || right.size() == 0) { ret.assign(70, 0); mGraph.mNodeDatas = ret; return; }
		if (left.size() == 1) {
			read_encoding_file(mSubject_names[left[0]], left_data);
		}
		if (right.size() == 1) {
			read_encoding_file(mSubject_names[right[0]], right_data);
		}
		if (left.size() > 1) {
			vector<vector<vec2f>> total_coordinate;
			for (auto item : left) {
				vector<vec2f> tmp;
				read_encoding_file(mSubject_names[item], tmp);
				total_coordinate.push_back(tmp);

			}

			//average coordinate
			size_t numCoords = total_coordinate[0].size();

			// Resize ret to match the size of one coordinate vector
			left_data.resize(numCoords);

			// Calculate average coordinate
			for (size_t i = 0; i < numCoords; ++i) {
				float sumX = 0.0f, sumY = 0.0f;
				for (const auto& coords : total_coordinate) {
					sumX += coords[i].x;
					sumY += coords[i].y;
				}
				left_data[i].x = sumX / total_coordinate.size();
				left_data[i].y = sumY / total_coordinate.size();
			}
		}
		if (right.size() > 1) {
			vector<vector<vec2f>> total_coordinate;
			for (auto item : right) {
				vector<vec2f> tmp;
				read_encoding_file(mSubject_names[item], tmp);
				total_coordinate.push_back(tmp);

			}

			//average coordinate
			size_t numCoords = total_coordinate[0].size();

			// Resize ret to match the size of one coordinate vector
			right_data.resize(numCoords);

			// Calculate average coordinate
			for (size_t i = 0; i < numCoords; ++i) {
				float sumX = 0.0f, sumY = 0.0f;
				for (const auto& coords : total_coordinate) {
					sumX += coords[i].x;
					sumY += coords[i].y;
				}
				right_data[i].x = sumX / total_coordinate.size();
				right_data[i].y = sumY / total_coordinate.size();
			}
		}

		for (int i = 0; i < ret.size(); i++) {
			ret[i] = dist2d(left_data[i], right_data[i]);
		}
		normailize_array(ret);
		mGraph.mNodeDatas = ret;
	}

	//utils for load coord and color
	bool read_encoding_file(string subject, std::vector<vec2f>& coordinate) {
		string path = data_path;
		std::string file_path = path + "coordinate/" + subject + ".dat";

		coordinate.clear();

		std::ifstream file(file_path, std::ios::binary | std::ios::ate);
		if (!file.is_open()) {
			std::cout << "Error opening mGraph file" << std::endl;
		}
		std::streamsize fileSize = file.tellg();
		file.seekg(0, std::ios::beg);

		float* data = new float[fileSize / sizeof(float)];
		file.read(reinterpret_cast<char*>(data), fileSize);

		for (int i = 0; i < 70 * 2; i = i + 2) {
			float x = (float)data[i];
			float y = (float)data[i + 1];
			coordinate.push_back(makeVec2f(x, y));
		}

		file.close();
		return true;
	}

	bool read_color_file(string subject, std::vector<float>& color_values) {
		string path = data_path;
		std::ifstream inputFile(path + "attention/" + subject + ".dat");
		float value;

		color_values.clear();

		while (inputFile >> value) {
			color_values.push_back(value);
		}
		inputFile.close();
		return true;
	}

	//load encodings and color
	void load_coordinate(GraphDisplay<float>& mGraph, std::vector<int> selected) {
		vector<vec2f> ret;
		if (selected.empty()) {
			read_encoding_file(mSubject_names[0], ret);
		}
		if (selected.size() == 1) {
			read_encoding_file(mSubject_names[selected[0]], ret);
		}
		if (selected.size() > 1) {
			vector<vector<vec2f>> total_coordinate;
			for (auto item : selected) {
				vector<vec2f> tmp;
				read_encoding_file(mSubject_names[item], tmp);
				total_coordinate.push_back(tmp);

			}

			//average coordinate
			size_t numCoords = total_coordinate[0].size();

			// Resize ret to match the size of one coordinate vector
			ret.resize(numCoords);

			// Calculate average coordinate
			for (size_t i = 0; i < numCoords; ++i) {
				float sumX = 0.0f, sumY = 0.0f;
				for (const auto& coords : total_coordinate) {
					sumX += coords[i].x;
					sumY += coords[i].y;
				}
				ret[i].x = sumX / total_coordinate.size();
				ret[i].y = sumY / total_coordinate.size();
			}


		}
		mGraph.getNodePositions() = ret;
	}

	void load_color(GraphDisplay<float>& mGraph, std::vector<int> selected) {
		std::vector<float> ret;
		if (selected.empty()) {
			read_color_file(mSubject_names[0], ret);
		}
		if (selected.size() == 1) {
			read_color_file(mSubject_names[selected[0]], ret);
		}
		if (selected.size() > 1) {
			vector<vector<vec2f>> total_coordinate;
			for (auto item : selected) {
				vector<vec2f> tmp;
				read_encoding_file(mSubject_names[item], tmp);
				total_coordinate.push_back(tmp);
			}

			vector<vector<vec2f>> total_coordinate_T = transpose(total_coordinate);
			for (auto row : total_coordinate_T) {
				ret.push_back(calculate_variance(row));
			}
			normailize_array(ret);
		}


		mGraph.mNodeDatas = ret;
	}



	std::vector<std::vector<vec2f>> transpose(const std::vector<std::vector<vec2f>> original) {
		if (original.empty() || original[0].empty()) {
			// 原始数据为空，返回空向量
			return {};
		}

		// 获取原始数据的维度
		size_t original_rows = original.size();
		size_t original_cols = original[0].size();

		// 创建转置后的数据结构
		std::vector<std::vector<vec2f>> transposed(original_cols, std::vector<vec2f>(original_rows));

		// 进行转置
		for (size_t i = 0; i < original_rows; ++i) {
			for (size_t j = 0; j < original_cols; ++j) {
				transposed[j][i] = original[i][j];
			}
		}

		return transposed;
	}

	float calculate_variance(const std::vector<vec2f> coordinates) {
		size_t n = coordinates.size();
		if (n <= 1) {
			// 如果样本数小于等于1，方差未定义
			return 0.0f;
		}

		// 计算 x 和 y 的均值
		float mean_x = 0.0f, mean_y = 0.0f;
		for (const auto& point : coordinates) {
			mean_x += point.x;
			mean_y += point.y;
		}
		mean_x /= n;
		mean_y /= n;

		// 计算方差
		float variance_x = 0.0f, variance_y = 0.0f;
		for (const auto& point : coordinates) {
			variance_x += (point.x - mean_x) * (point.x - mean_x);
			variance_y += (point.y - mean_y) * (point.y - mean_y);
		}
		variance_x /= n - 1; // 使用 n-1 作为分母，这是无偏方差的计算方式
		variance_y /= n - 1;

		// 返回 x 和 y 方差的平均值
		return (variance_x + variance_y) / 2.0f;
	}

	void processEdge(std::vector<int> nodes) {
		edges.clear();

		if (!nodes.empty()) {
			for (auto element : nodes) {
				std::vector<int> row = connection[element];
				bool allZeros = all_of(row.begin(), row.end(), [](int num) {return num == 0; });
				if (allZeros) { continue; }
				for (int j = element + 1; j < row.size(); ++j) {
					//check edge
					if (row[j] == 1) {
						edges.push_back({ element, j });
					}
				}
			}
		}


	}

	void processEdge2(std::vector<int> nodes) {
		edges2.clear();

		if (!nodes.empty()) {
			for (auto element : nodes) {
				std::vector<int> row = connection2[element];
				bool allZeros = all_of(row.begin(), row.end(), [](int num) {return num == 0; });
				if (allZeros) { continue; }
				for (int j = element + 1; j < row.size(); ++j) {
					//check edge
					if (row[j] == 1) {
						edges2.push_back({ element, j });
					}
				}
			}
		}
	}

	void update_edge_selected() {
		mLeft.mGraph.clearSelectionBox();
		mLeft.mGraph.finishSelection(mSelectedROIs);
		mLeft.mGraph.clearSelection();

		processEdge(mSelectedROIs);
		mLeft.mGraph.setEdgeDatas(edges);

		edgesSelected.clear();
	}

	void display() {
		mLeft.display();
		mRight.display();
		mBrain.display();
		drawBox();
		
		
	}

	void drawBox() {
		mLeft.mGraph.drawSelectionBox();
		mLeft.mGraph.drawClickBox();
		mRight.mGraph.drawSelectionBox();
		mRight.mGraph.drawClickBox();
	}

	void setData(vector<int> left, vector<int> right, int conv, string pooling) {

		mSelectedROIs.clear();
		//set display area
		setLayout();

		//set parametrs
		mConv = conv;
		mPooling = pooling;
		data_path = "D:/DATA/brain/encodings/encoding_" + mPooling + "_" + to_string(mConv) + "/";

		//************************************ Left View ***********************************************
		selected_left = left;
		subject_left = left.size() == 1 ? mSubject_names[left[0]] : mSubject_names.back();
		load_coordinate(mLeft.mGraph, left);
		load_color(mLeft.mGraph, left);

		mLeft.mGraph.clearSelection();
		mLeft.mGraph.updateNodePositionRange();
		mLeft.mGraph.updateNodeDisplayPosition();
		mLeft.mGraph.updateNodeColorByData();

		mLeft.setConnection(subject_left);

		mLeft.setLayoutUpdate(true);

		//************************************ Right View ***********************************************
		selected_right = right;
		subject_right = right.size() == 1 ? mSubject_names[right[0]] : mSubject_names.back();
		load_coordinate(mRight.mGraph, right);
		load_color(mRight.mGraph, right);

		mRight.mGraph.clearSelection();
		mRight.mGraph.updateNodePositionRange();
		mRight.mGraph.updateNodeDisplayPosition();
		mRight.mGraph.updateNodeColorByData();

		mRight.setConnection(subject_right);

		mRight.setLayoutUpdate(false);

		//************************************ brain View ***********************************************

		vector<vec2f> roi_positions;
		read_brain_encoding_file(roi_positions);
		mBrain.getNodePositions().clear();
		mBrain.getNodePositions() = roi_positions;

		mBrain.clearSelection();
		mBrain.updateNodePositionRange();
		mBrain.updateNodeDisplayPosition();

		load_brain_color(mBrain, selected_left, selected_right);
		mBrain.updateNodeColorByData();

		


	}

	/*void set_left_data(vector<int> left) {
		selected_left = left;
		subject_left = left.size() == 1 ? mSubject_names[left[0]] : mSubject_names.back();
		load_coordinate(mLeft, left);
		load_color(mLeft, left);

		mLeft.clearSelection();
		mLeft.updateNodePositionRange();
		mLeft.updateNodeDisplayPosition();
		mLeft.updateNodeColorByData();
	}

	void set_right_data(vector<int> right) {
		selected_right = right;
		subject_right = right.size() == 1 ? mSubject_names[right[0]] : mSubject_names.back();
		load_coordinate(mRight, right);
		load_color(mRight, right);

		mRight.clearSelection();
		mRight.updateNodePositionRange();
		mRight.updateNodeDisplayPosition();
		mRight.updateNodeColorByData();
	}*/

	void setArea(const RectDisplayArea& area) { mArea = area; }

	void setLayout() {
		float margin = 80;
		float width = 250;
		mLeft.setArea(makeRectDisplayArea(makeVec2f(mArea.origin.x-30, mArea.origin.y), makeVec2f(width + 30, 0.0f), makeVec2f(0.0f, 250)));
		mBrain.setArea(makeRectDisplayArea(makeVec2f(mArea.origin.x + width + margin, mArea.origin.y), makeVec2f(220, 0.0f), makeVec2f(0.0f, 1.35 * 220)));
		mRight.setArea(makeRectDisplayArea(makeVec2f(mArea.origin.x + 2 * width + 100, mArea.origin.y), makeVec2f(width + 30, 0.0f), makeVec2f(0.0f, 250)));
		
	}

	void set_data_pointer(GlobalDataManager* data) {
		gdata = data;
	}

	inline bool inDisplayArea(const vec2f& p) {
		return inRectDisplayArea(p, mArea);
	}

	void press_right(const vec2f& p) {
		if (mBrain.inDisplayArea(p)) {
			check_ROI = mBrain.get_id_by_pos(p);
			MessageCenter::sharedCenter()->processMessage("Check ROI", "overview");
		}
	}

	void press(const vec2f& p) {

		if (mLeft.mGraph.inDisplayArea(p)) {
			readConnectionFile(subject_left);
			mLeft.mGraph.setSelectionAnchor(p);
		}
		if (mRight.mGraph.inDisplayArea(p)) {
			readConnectionFile2(subject_right);
			mRight.mGraph.setSelectionAnchor(p);
		}
		if (mBrain.inDisplayArea(p)) {

			mBrain.setSelectionAnchor(p);
		}
	}


	void move(const vec2f& p) {
		if (mLeft.mGraph.inDisplayArea(p)) {
			if (mLeft.mGraph.inSelection()) {
				mLeft.mGraph.updateSelectionBox(p);
				mLeft.mGraph.drawSelectionBox();
			}
		}
		else {
			if (mRight.mGraph.inSelection()) {
				mRight.mGraph.updateSelectionBox(p);
				mRight.mGraph.drawSelectionBox();
			}
		}


	}
	//brush to select nodes
	void left_mouse_release(const vec2f& p) {
		if (mLeft.mGraph.inDisplayArea(p)) {
			mLeft.mGraph.updateSelectionBox(p);
			mLeft.mGraph.finishSelection(mSelectedROIs);
			mLeft.mGraph.setSelectedNodes(mSelectedROIs);

			processEdge(mSelectedROIs);
			mLeft.mGraph.setEdgeDatas(edges);

			mRight.mGraph.setSelectedNodes(mSelectedROIs);
			mBrain.setSelectedNodes(mSelectedROIs);
			//mBrain.setEdgeDatas(edges);
			
		}
		if (mRight.mGraph.inDisplayArea(p)) {
			mRight.mGraph.updateSelectionBox(p);
			mRight.mGraph.finishSelection(mSelectedROIs);
			mRight.mGraph.setSelectedNodes(mSelectedROIs);

			processEdge2(mSelectedROIs);
			mRight.mGraph.setEdgeDatas(edges2);

			mLeft.mGraph.setSelectedNodes(mSelectedROIs);
			mBrain.setSelectedNodes(mSelectedROIs);
			
		}
		if (mBrain.inDisplayArea(p)) {
			readConnectionFile(subject_left);
			readConnectionFile2(subject_right);

			mBrain.updateSelectionBox(p);
			mBrain.finishSelection(ROIs_temp);
			mSelectedROIs.insert(mSelectedROIs.end(), ROIs_temp.begin(), ROIs_temp.end());
			mBrain.setSelectedNodes(mSelectedROIs);
			mLeft.mGraph.setSelectedNodes(mSelectedROIs);
			mRight.mGraph.setSelectedNodes(mSelectedROIs);

			processEdge(mSelectedROIs);
			mLeft.mGraph.setEdgeDatas(edges);
			processEdge2(mSelectedROIs);
			mRight.mGraph.setEdgeDatas(edges2);
			//mBrain.setEdgeDatas(edges);
			
		}
		MessageCenter::sharedCenter()->processMessage("NodeTrix Set","EmbeddingDisplay2");
	}
	// click to select edges
	void right_mouse_release(const vec2f& p) {
		if (mLeft.mGraph.inDisplayArea(p)) {

			mLeft.mGraph.setClickPosition(p);
			mLeft.mGraph.checkEdges();
			edgesSelected = mLeft.mGraph.getEdgeSelected();

			mRight.mGraph.setEdgesSelected(edgesSelected[0]);
			MessageCenter::sharedCenter()->processMessage("SET SUBJECT BY EDGE", "EmbeddingDisplay2");
		}
		else {

			mRight.mGraph.setClickPosition(p);
			mRight.mGraph.checkEdges();
			edgesSelected = mRight.mGraph.getEdgeSelected();

			mLeft.mGraph.setEdgesSelected(edgesSelected[0]);
		}
	}

	//private:
	//data
	std::vector<std::string> mSubject_names;
	std::string subject_left, subject_right;
	std::vector<int> selected_left, selected_right;
	GlobalDataManager* gdata;
	

	std::vector<int> mSelectedROIs;
	std::vector<int> ROIs_temp;
	int check_ROI;

	std::vector<std::vector<int>> connection;
	std::vector<std::vector<int>> connection2;
	std::vector<std::vector<int>> edges;
	std::vector<std::vector<int>> edges2;
	std::vector<std::vector<int>> edgesSelected;

	//view
	GraphDisplay<float> mBrain;
	
	NodeTrixDisplay mLeft;
	NodeTrixDisplay mRight;

	//for display layout
	RectDisplayArea mArea;

	//parameters
	string data_path;

	int mConv;
	string mPooling;

};