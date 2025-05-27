#pragma once
#include "typeOperation.h"
#include "ColorMap.h"
#include "MatrixData.h"
#include "DisplayWidget.h"
#include "BarDisplay.h"
#include "MatrixDisplay.h"
#include "GraphDisplay.h"
#include "GlobalDataManager.h"

class EmbeddingDisplay {
public:

	EmbeddingDisplay() {

		read_subject_names(mSubject_names);
		selected_left.push_back(0);
		selected_right.push_back(0);
		mLeft.setColorScheme(COLOR_MAP_PURPLE_SCALE);
		mRight.setColorScheme(COLOR_MAP_YELLOW_SCALE);

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

	bool read_encoding_file(string subject,std::vector<vec2f>& coordinate) {
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

	bool read_color_file( string subject, std::vector<float>& color_values) {
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

	void load_brain_color(GraphDisplay<float>& mGraph, std::vector<int> left, std::vector<int> right ,bool isMean=true) {
		std::vector<float> ret(70);
		std::vector<vec2f> left_data;
		std::vector<vec2f> right_data;
		if (left.size() == 0 || right.size() == 0) { ret.assign(70, 0); mGraph.mNodeDatas = ret; return; }
		if(left.size() == 1){
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


	std::vector<std::vector<vec2f>> transpose(const std::vector<std::vector<vec2f>> original) {
		if (original.empty() || original[0].empty()) {
			// ԭʼ����Ϊ�գ����ؿ�����
			return {};
		}

		// ��ȡԭʼ���ݵ�ά��
		size_t original_rows = original.size();
		size_t original_cols = original[0].size();

		// ����ת�ú�����ݽṹ
		std::vector<std::vector<vec2f>> transposed(original_cols, std::vector<vec2f>(original_rows));

		// ����ת��
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
			// ���������С�ڵ���1������δ����
			return 0.0f;
		}

		// ���� x �� y �ľ�ֵ
		float mean_x = 0.0f, mean_y = 0.0f;
		for (const auto& point : coordinates) {
			mean_x += point.x;
			mean_y += point.y;
		}
		mean_x /= n;
		mean_y /= n;

		// ���㷽��
		float variance_x = 0.0f, variance_y = 0.0f;
		for (const auto& point : coordinates) {
			variance_x += (point.x - mean_x) * (point.x - mean_x);
			variance_y += (point.y - mean_y) * (point.y - mean_y);
		}
		variance_x /= n - 1; // ʹ�� n-1 ��Ϊ��ĸ��������ƫ����ļ��㷽ʽ
		variance_y /= n - 1;

		// ���� x �� y �����ƽ��ֵ
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

	void processEdge_truss(std::vector<int> nodes) {
		edges.clear();

		if (!nodes.empty()) {
			for (size_t i = 0; i < nodes.size(); ++i) {
				int element = nodes[i];
				std::vector<int> row = connection[element];

				// ����nodes�е�ǰ��֮������е�
				for (size_t j = i + 1; j < nodes.size(); ++j) {
					int target = nodes[j];
					// ���element��target֮���Ƿ�������
					if (row[target] == 1) {
						edges.push_back(  { element, target });
					}
				}
			}
		}


	}

	void processEdge_truss2(std::vector<int> nodes) {
		edges2.clear();

		if (!nodes.empty()) {
			for (size_t i = 0; i < nodes.size(); ++i) {
				int element = nodes[i];
				std::vector<int> row = connection2[element];

				// ����nodes�е�ǰ��֮������е�
				for (size_t j = i + 1; j < nodes.size(); ++j) {
					int target = nodes[j];
					// ���element��target֮���Ƿ�������
					if (row[target] == 1) {
						edges2.push_back({ element, target });
					}
				}
			}
		}


	}

	void update_edge_selected() {
		mLeft.clearSelectionBox();
		mLeft.finishSelection(mSelectedROIs);
		mLeft.clearSelection();

		processEdge(mSelectedROIs);
		mLeft.setEdgeDatas(edges);

		edgesSelected.clear();
	}

	void display() {
		mLeft.display();
		mRight.display();
		mBrain.display();
		drawBox();

	}

	void drawBox() {
		mLeft.drawSelectionBox();
		mLeft.drawClickBox();
		mRight.drawSelectionBox();
		mRight.drawClickBox();
	}

	void setData(vector<int> left, vector<int> right,int conv,string pooling) {

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
		load_coordinate(mLeft, left);
		load_color(mLeft, left);

		mLeft.clearSelection();
		mLeft.updateNodePositionRange();
		mLeft.updateNodeDisplayPosition();
		mLeft.updateNodeColorByData();
		
		//************************************ Right View ***********************************************
		selected_right = right;
		subject_right = right.size() == 1 ? mSubject_names[right[0]] : mSubject_names.back();
		load_coordinate(mRight, right);
		load_color(mRight, right);

		mRight.clearSelection();
		mRight.updateNodePositionRange();
		mRight.updateNodeDisplayPosition();
		mRight.updateNodeColorByData();

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

	void set_left_data(vector<int> left) {
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
	}

	void setArea(const RectDisplayArea& area) { mArea = area; }

	void setLayout() {
		float margin = 80;
		float width = 250;
		mLeft.setArea(makeRectDisplayArea(makeVec2f(mArea.origin.x, mArea.origin.y), makeVec2f(width+30, 0.0f), makeVec2f(0.0f, 250)));
		mBrain.setArea(makeRectDisplayArea(makeVec2f(mArea.origin.x + width+ margin, mArea.origin.y), makeVec2f(220, 0.0f), makeVec2f(0.0f, 1.35*220)));
		mRight.setArea(makeRectDisplayArea(makeVec2f(mArea.origin.x+ 2*width+ 140, mArea.origin.y), makeVec2f(width+30, 0.0f), makeVec2f(0.0f, 250)));

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

		if (mLeft.inDisplayArea(p)) {
			readConnectionFile(subject_left);
			mLeft.setSelectionAnchor(p);
		}
		if (mRight.inDisplayArea(p)) {
			readConnectionFile2(subject_right);
			mRight.setSelectionAnchor(p); 
		}
		if (mBrain.inDisplayArea(p)) {
			
			mBrain.setSelectionAnchor(p);
		}
	}


	void move(const vec2f& p) {
		if (mLeft.inDisplayArea(p)) {
			if (mLeft.inSelection()) {
				mLeft.updateSelectionBox(p);
				mLeft.drawSelectionBox();
			}
		}
		else {
			if (mRight.inSelection()) {
				mRight.updateSelectionBox(p);
				mRight.drawSelectionBox();
			}
		}


	}
	//brush to select nodes
	void left_mouse_release(const vec2f& p) {
		if (mLeft.inDisplayArea(p)) {
			mLeft.updateSelectionBox(p);
			mLeft.finishSelection(mSelectedROIs);
			mLeft.setSelectedNodes(mSelectedROIs);

			processEdge(mSelectedROIs);
			mLeft.setEdgeDatas(edges);

			mRight.setSelectedNodes(mSelectedROIs);
			mBrain.setSelectedNodes(mSelectedROIs);
		}
		if (mRight.inDisplayArea(p)) {
			mRight.updateSelectionBox(p);
			mRight.finishSelection(mSelectedROIs);
			mRight.setSelectedNodes(mSelectedROIs);

			processEdge2(mSelectedROIs);
			mRight.setEdgeDatas(edges2);

			mLeft.setSelectedNodes(mSelectedROIs);
			mBrain.setSelectedNodes(mSelectedROIs);
		}
		if (mBrain.inDisplayArea(p)) {
			readConnectionFile(subject_left);
			readConnectionFile2(subject_right);

			mBrain.updateSelectionBox(p);
			mBrain.finishSelection(ROIs_temp);
			mSelectedROIs.insert(mSelectedROIs.end(), ROIs_temp.begin(), ROIs_temp.end());
			mBrain.setSelectedNodes(mSelectedROIs);
			mLeft.setSelectedNodes(mSelectedROIs);
			mRight.setSelectedNodes(mSelectedROIs);

			processEdge(mSelectedROIs);
			mLeft.setEdgeDatas(edges);
			processEdge2(mSelectedROIs);
			mRight.setEdgeDatas(edges2);
		}
	}
	// click to select edges
	void right_mouse_release(const vec2f& p) {
		if (mLeft.inDisplayArea(p)) {

			mLeft.setClickPosition(p);
			mLeft.checkEdges();
			edgesSelected = mLeft.getEdgeSelected();
			
		}
		else{

			mRight.setClickPosition(p);
			mRight.checkEdges();
			edgesSelected = mRight.getEdgeSelected();
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
	GraphDisplay<float> mLeft;
	GraphDisplay<float> mRight;
	GraphDisplay<float> mBrain;

	//for display layout
	RectDisplayArea mArea;

	//parameters
	string data_path;

	int mConv;
	string mPooling;

};