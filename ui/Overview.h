#pragma once
#include "typeOperation.h"
#include "ColorMap.h"
#include "MatrixData.h"
#include "DisplayWidget.h"
#include "BarDisplay.h"
#include "MatrixDisplay.h"
#include "GraphDisplay.h"
#include "GraphDisplay2.h"

class Overview {
public:

	Overview() {

		readInfo();

		mDataNum = 0;
		mNumBin = 20;
		mCutoffRatio = 0.0f;
		mForeground.assign(198, 1);
		confusion_matrix_view.setColorScheme(COLOR_MAP_PIYG);
		
		
	}

	void display() {
		age_view.display();
		educ_view.display();
		confusion_matrix_view.display();
		scatter_view.drawNodes_with_depth();
		scatter_view.drawSelectionBox();

	}

	template <typename T>
	Range compute_minmax(std::vector<T> arr) {
		Range ret;
		float maxv = *std::max_element(arr.begin(), arr.end());
		float minv = *std::min_element(arr.begin(), arr.end());
		ret = makeRange(minv, maxv);
		return ret;
	}

	bool readInfo() {
		//load label
		std::string label_file_path = "D:/DATA/brain/labels.dat";
		labels.clear();
		// 打开文本文件
		std::ifstream label_file(label_file_path);
		if (!label_file.is_open()) {
			return false;
		}
		int label_data;
		while (label_file >> label_data) {
			labels.push_back(label_data);
		}
		label_file.close();

		//load pred_label
		std::string pred_label_file_path = "D:/DATA/brain/pred_labels.dat";
		pred_labels.clear();
		// 打开文本文件
		std::ifstream pred_label_file(pred_label_file_path);
		if (!pred_label_file.is_open()) {
			return false;
		}
		int pred_label_data;
		while (pred_label_file >> pred_label_data) {
			pred_labels.push_back(pred_label_data);
		}
		pred_label_file.close();

		//load Age
		std::string Age_file_path = "D:/DATA/brain/Age.dat";
		Age.clear();
		// 打开文本文件
		std::ifstream Age_file(Age_file_path);
		if (!Age_file.is_open()) {
			return false;
		}
		float Age_data;
		while (Age_file >> Age_data) {
			Age.push_back(Age_data);
		}
		Age_file.close();

		//load Educ
		std::string Educ_file_path = "D:/DATA/brain/Educ.dat";
		Educ.clear();
		// 打开文本文件
		std::ifstream Educ_file(Educ_file_path);
		if (!Educ_file.is_open()) {
			return false;
		}
		float Educ_data;
		while (Educ_file >> Educ_data) {
			Educ.push_back(Educ_data);
		}
		Educ_file.close();

		std::string cls_file_path = "D:/DATA/brain/cls_tokens.dat";
		cls_tokens.clear();
		std::ifstream cls_file(cls_file_path);
		if (!cls_file.is_open()) {
			return false;
		}
		float x,y;
		while (cls_file >> x>>y) {
			cls_tokens.push_back(makeVec2f(x,y));
		}
		cls_file.close();

		return true;
	}

	template <typename T>
	vector<T> filter(const std::vector<int>& indices, const std::vector<T>& allData) {
		size_t size = indices.size();

		vector<T> selectedElements;

		for (size_t i = 0; i < size; ++i) {
			int index = indices[i];
			if (index < 0 || index >= static_cast<int>(allData.size())) {
				throw std::out_of_range("Index out of range");
			}
			selectedElements.push_back(allData[index]);
		}

		return selectedElements;
	}
	
	template <typename T>
	vector<T> filter(const Range& bound, const std::vector<T>& allData) {
		std::vector<T> selectedElements;

		for (const T& element : allData) {
			if (element >= bound.lower && element <= bound.upper) {
				selectedElements.push_back(element);
			}
		}

		// 返回包含选定元素的 Selected 结构
		return selectedElements;
	}

	template <typename T>
	vector<T> copy_data(const Range& ratio, const std::vector<T>& allData) {

		// 计算起始索引和结束索引
		size_t start = static_cast<size_t>(ratio.lower * allData.size());
		size_t end = static_cast<size_t>(ratio.upper * allData.size());

		// 使用 std::vector 构造函数拷贝指定范围的数据
		return std::vector<T>(allData.begin() + start, allData.begin() + end);
	}

	std::vector<float> GenerateConfusionMatrix(const std::vector<int>& labels, const std::vector<int>& pred_labels) {
		// 检查向量的大小是否一致
		if (labels.size() != pred_labels.size()) {
			throw std::invalid_argument("Input vectors must have the same size");
		}

		const int num_classes = 4;
		std::vector<float> confusionMatrix(num_classes * num_classes, 0.0f);

		for (size_t i = 0; i < labels.size(); ++i) {
			if (labels[i] < 0 || labels[i] >= num_classes || pred_labels[i] < 0 || pred_labels[i] >= num_classes) {
				throw std::invalid_argument("Label values should be between 0 and 3");
			}

			confusionMatrix[labels[i] * num_classes + pred_labels[i]]++;
		}

		return confusionMatrix;
	}


	bool containsXY(const std::vector<vec2i>& selected_cells, int x, int y) {
		for (const auto& cell : selected_cells) {
			if (cell.x == x && cell.y == y) {
				return true;
			}
		}
		return false;
	}

	template <typename T>
	std::vector<T> set_scatter_selected(std::vector<vec2i> selected_cells, std::vector<T> arr) {
		std::vector<T> ret = arr;
		for (int i = 0; i< ret.size(); i++) {
			if (ret[i] == 1) {
				int x = pred_labels[i];
				int y = labels[i];
				if (!containsXY(selected_cells, x, y)) { ret[i] = 0; }
			}
			
		}
		return ret;
	}

	float get_color_by_XY(int x, int y) {
		return confusion_matrix_view.get_value_by_cord(makeVec2i(x, y));
	}

	template <typename T>
	std::vector<T> set_scatter_color(std::vector<vec2i> selected_cells, std::vector<T> arr) {
		std::vector<T> ret_colors= arr;
		for (int i = 0; i < ret_colors.size(); i++) {
			if (arr[i] == 1) {
				int x = pred_labels[i];
				int y = labels[i];
				if (containsXY(selected_cells, x, y)) { ret_colors[i] = get_color_by_XY(x,y); }
				else { ret_colors[i] = 0; }
			}
		}
		return ret_colors;
	}
	template <typename T>
	void set_scatter_border(std::vector<vec2i> selected_cells, std::vector<int>& border_marks,int mark, std::vector<T> arr) {
		for (int i = 0; i < border_marks.size(); i++) {
			if (arr[i] == 1) {
				int x = pred_labels[i];
				int y = labels[i];
				if (containsXY(selected_cells, x, y)) { border_marks[i] = mark; }
				//else { border_marks[i] = 0; }
			}
		}
		
	}

	template <typename T>
	void set_scatter_selected(const Range& ratio, std::vector<T>& arr) {
		// 计算起始索引和结束索引
		size_t start = static_cast<size_t>(ratio.lower * arr.size());
		size_t end = static_cast<size_t>(ratio.upper * arr.size());
		// 确保范围在有效的索引范围内
		start = std::min(start, arr.size());
		end = std::min(end, arr.size());

		// 将范围内的元素置为1
		for (size_t i = start; i < end; ++i) {
			arr[i] = 1;
		}

	}

	void setData() {
		//set display area
		setLayout();

		clear_selected();

		//************************************ Bar Display ***********************************************

		// get selected data
		vector<float> age_data_ori = Age;
		vector<float> educ_data_ori = Educ;

		float* age_data = age_data_ori.data();
		float* educ_data = educ_data_ori.data();
		mDataNum = Age.size();

		//histogram assign
		std::vector<int> age_hist;
		std::vector<int> educ_hist;
		Range age_bound = compute_bound(age_data, mDataNum, mCutoffRatio);
		Range educ_bound = compute_bound(educ_data, mDataNum, mCutoffRatio);
		histogram(age_hist, age_data, mDataNum, mNumBin, age_bound, true);
		histogram(educ_hist, educ_data, mDataNum, mNumBin, educ_bound, true);

		age_view.set_data(age_hist);
		educ_view.set_data(educ_hist);
		age_view.set_range(MINMAX);
		educ_view.set_range(MINMAX);
		age_view.set_domain(age_bound);
		educ_view.set_domain(educ_bound);

		//************************************ Confusion Matrix ***********************************************

		vector<int> label_data = labels;
		vector<int> pred_label_data = pred_labels;
		vector<float> cm_data = GenerateConfusionMatrix(label_data, pred_label_data);
		float* confusion_matrix_ptr = cm_data.data();
		MatrixData<float> confusion_matrix(4, 4, confusion_matrix_ptr);
		confusion_matrix_data = confusion_matrix.convert<float>();

		confusion_matrix_view.setData(confusion_matrix_data);
		confusion_matrix_view.set_range(MINMAX);
		

		//************************************ Global Embedding ***********************************************

		scatter_view.setNumber(198);
		std::vector<float> label_data_float;
		for (int label : label_data) {
	        label_data_float.push_back(static_cast<float>(label));
		}
		std::vector<float> foreground(198, 1);
		scatter_view.setNodeDatas(foreground);
		scatter_view.updateNodeColorByForeground();

		scatter_view.getNodePositions()=cls_tokens;
		scatter_view.updateNodePositionRange();
		scatter_view.updateNodeDisplayPosition();
		



	}
	

	void setArea(const RectDisplayArea& area) { mArea = area; }

	void setLayout() {
		float BarDisplay_margin = 250;
		float Matrix_margin = 150;
		age_view.set_area(makeRectDisplayArea(makeVec2f(mArea.origin.x , mArea.origin.y + BarDisplay_margin), makeVec2f(200.0f, 0.0f), makeVec2f(0.0f, 120.0f)));
		educ_view.set_area(makeRectDisplayArea(makeVec2f(mArea.origin.x, mArea.origin.y + 80), makeVec2f(200.0f, 0.0f), makeVec2f(0.0f, 120.0f)));
		confusion_matrix_view.setArea(makeRectDisplayArea(makeVec2f(mArea.origin.x +age_view.getArea().row_axis.x+ Matrix_margin, mArea.origin.y + 100), makeVec2f(200.0f, 0.0f), makeVec2f(0.0f, 200.0f)));
		scatter_view.setArea(makeRectDisplayArea(makeVec2f(mArea.origin.x + 600, mArea.origin.y + 80), makeVec2f(250.0f, 0.0f), makeVec2f(0.0f, 220.0f)));
	}

	std::string get_selected_container() {
		if (selected_left.empty()) { return "left"; }
		else { return "right"; }
	}

	//std::vector<int> get_selected() { return data; }

	Range get_absolute_range(const Range& seg, const Range& entire) {
		Range ret;
		ret.lower = interpolate(0.0f, 1.0f, seg.lower, entire.lower, entire.upper);
		ret.upper = interpolate(0.0f, 1.0f, seg.upper, entire.lower, entire.upper);
		return ret;
	}

	inline bool inDisplayArea(const vec2f& p) {
		return inRectDisplayArea(p, mArea);
	}

	bool in_interaction() {
		return (age_view.in_interaction() || educ_view.in_interaction());
	}

	void clear_selected() {
		confusion_matrix_view.clear_selected();
		selected_left.clear();
		selected_right.clear();
		scatter_view.clearSelection();
	}

	bool mouse_over(const vec2f& p) {
		if (!age_view.mouse_over(p)) {
			return educ_view.mouse_over(p);
		}
		return true;
	}

	bool press(const vec2f& p) {
		if (scatter_view.inDisplayArea(p)) {
			scatter_view.setSelectionAnchor(p);
		}
		if (confusion_matrix_view.inDisplayArea(p)) {
			/*confusion_matrix_view.mouse_left_click(p);
			std::vector<vec2i> selected_cells = confusion_matrix_view.selected_coords_left;
			std::vector<float> values = set_scatter_color(selected_cells, mForeground);
			scatter_view.setNodeDatas(values);
			scatter_view.updateNodeColorByForeground();*/

			confusion_matrix_view.mouse_left_click(p);
			std::vector<int> borders(198, 0);
			std::vector<vec2i> selected_cells_left = confusion_matrix_view.selected_coords_left;
			std::vector<vec2i> selected_cells_right = confusion_matrix_view.selected_coords_right;

			set_scatter_border(selected_cells_left, borders, 1, mForeground);
			set_scatter_border(selected_cells_right, borders, 2, mForeground);
			scatter_view.setNodeBorders(borders);

			selected_cells_left.insert(selected_cells_left.end(), selected_cells_right.begin(), selected_cells_right.end());
			std::vector<float> values = set_scatter_color(selected_cells_left, mForeground);
			scatter_view.setNodeDatas(values);
			scatter_view.updateNodeColorByForeground();
			
		}
		if (age_view.inDisplayArea(p)) { age_view.press(p); }
		if (educ_view.inDisplayArea(p)) { educ_view.press(p); }
		/*if (!age_view.press(p)) {
			return educ_view.press(p);
		}*/
		
		return true;
	}

	bool press_right(const vec2f& p) {
		if (scatter_view.inDisplayArea(p)) {
			scatter_view.setSelectionAnchor(p);
		}
		if (confusion_matrix_view.inDisplayArea(p)) {
			confusion_matrix_view.mouse_right_click(p);
			std::vector<int> borders(198, 0);
			std::vector<vec2i> selected_cells_left = confusion_matrix_view.selected_coords_left;
			std::vector<vec2i> selected_cells_right = confusion_matrix_view.selected_coords_right;

			set_scatter_border(selected_cells_left, borders, 1,mForeground);
			set_scatter_border(selected_cells_right, borders, 2, mForeground);
			scatter_view.setNodeBorders(borders);
			selected_cells_left.insert(selected_cells_left.end(), selected_cells_right.begin(), selected_cells_right.end());
			std::vector<float> values = set_scatter_color(selected_cells_left, mForeground);
			scatter_view.setNodeDatas(values);
			scatter_view.updateNodeColorByForeground();

		}
		return true;
	}

	void drag(const vec2f& p) {
		if (age_view.in_interaction()) {
			age_view.drag(p);
			
			Range selected_ratio = age_view.get_selected_relative();
			Range selected_domain = get_absolute_range(selected_ratio, compute_minmax(Educ));
			vector<float> selected_data = filter(selected_domain, Educ);

			educ_view.set_domain(selected_domain);
			
			std::vector<int> hist;
			histogram(hist, selected_data.data(), selected_data.size(), mNumBin, selected_domain, true);
			educ_view.set_data(hist);
			educ_view.set_range(MINMAX);

			vector<int> label_data = copy_data(selected_ratio, labels);
			vector<int> pred_label_data = copy_data(selected_ratio, pred_labels);
			vector<float> cm_data = GenerateConfusionMatrix(label_data, pred_label_data);
			float* confusion_matrix_ptr = cm_data.data();
			MatrixData<float> confusion_matrix(4, 4, confusion_matrix_ptr);
			confusion_matrix_data = confusion_matrix.convert<float>();

			confusion_matrix_view.setData(confusion_matrix_data);
			confusion_matrix_view.set_range(MINMAX);

			vector<float> foreground(198);
			set_scatter_selected(selected_ratio, foreground);
			scatter_view.setNodeDatas(foreground);
			scatter_view.updateNodeColorByForeground();

			mForeground = foreground;
			
		}
		else if (educ_view.in_interaction()) {
			educ_view.drag(p);

			Range selected_ratio = educ_view.get_selected_relative();
			Range selected_domain = get_absolute_range(selected_ratio, compute_minmax(Age));
			vector<float> selected_data = filter(selected_domain, Age);

			age_view.set_domain(selected_domain);

			std::vector<int> hist;
			histogram(hist, selected_data.data(), selected_data.size(), mNumBin, selected_domain, true);
			age_view.set_data(hist);
			age_view.set_range(MINMAX);

			vector<int> label_data = copy_data(selected_ratio, labels);
			vector<int> pred_label_data = copy_data(selected_ratio, pred_labels);
			vector<float> cm_data = GenerateConfusionMatrix(label_data, pred_label_data);
			float* confusion_matrix_ptr = cm_data.data();
			MatrixData<float> confusion_matrix(4, 4, confusion_matrix_ptr);
			confusion_matrix_data = confusion_matrix.convert<float>();

			confusion_matrix_view.setData(confusion_matrix_data);
			confusion_matrix_view.set_range(MINMAX);

			vector<float> foreground(198);
			set_scatter_selected(selected_ratio, foreground);
			scatter_view.setNodeDatas(foreground);
			scatter_view.updateNodeColorByForeground();

			mForeground = foreground;
		}
	}

	void move(const vec2f& p) {
		if (scatter_view.inDisplayArea(p)) {
			if (scatter_view.inSelection()) {
				scatter_view.updateSelectionBox(p);
				scatter_view.drawSelectionBox();
			}
		}
	}

	void release(const vec2f& p) {
		if (age_view.in_interaction()) {
			age_view.release(p);
		}
		else if (educ_view.in_interaction()) {
			educ_view.release(p);
		}
		if (scatter_view.inDisplayArea(p)) {
			scatter_view.updateSelectionBox(p);
			std::vector<int> left_temp;
			scatter_view.finishSelection_with_foreground(left_temp);
			selected_left.insert(selected_left.end(), left_temp.begin(), left_temp.end());
			scatter_view.setSelectedNodes_left(selected_left);
			MessageCenter::sharedCenter()->processMessage("Select Individuals", "overview");
		}
	}

	void release_right(const vec2f& p) {
		
		if (scatter_view.inDisplayArea(p)) {
			scatter_view.updateSelectionBox(p);
			
			std::vector<int> right_temp;
			scatter_view.finishSelection_with_foreground(right_temp);
			selected_right.insert(selected_right.end(), right_temp.begin(), right_temp.end());
			scatter_view.setSelectedNodes_right(selected_right);
			MessageCenter::sharedCenter()->processMessage("Select Individuals", "overview");
		}
	}

	//private:
	//data
	std::vector<int> labels;
	std::vector<int> pred_labels;
	std::vector<float> Age;
	std::vector<float> Educ;
	std::vector<vec2f> cls_tokens;

	//selected data
	std::vector<float> mForeground;
	std::vector<int> selected_left; 
	std::vector<int> selected_right;

	//std::vector<int> age_data, educ_data;
	MatrixData<float>* confusion_matrix_data;

	//view
	BarDisplay age_view, educ_view;
	MatrixDisplay<float> confusion_matrix_view;
	GraphDisplay2<float> scatter_view;

	//for display layout
	RectDisplayArea mArea;
	

	//histogram parameter
	/*Range age_range, educ_range;
	float age_interval;
	float educ_interval;*/

	int mNumBin;
	int mDataNum;
	float mCutoffRatio;

};