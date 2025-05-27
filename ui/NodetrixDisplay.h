#include <unordered_map>
#include "typeOperation.h"
#include "DisplayWidget.h"
#include "ColorMap.h"
#include "GraphDisplay.h"
#include "MatrixDisplay.h"
#include "FRLayout.h"
#include <algorithm>
#include <cmath>


class NodeTrixDisplay {
public:
	NodeTrixDisplay() {
		brainRegions = {
			// 额叶（Frontal Lobe）
			{4, 13, 15, 19, 20, 21, 25, 27, 28, 29, 33, 39, 48, 50, 54, 55, 56, 60, 62, 63, 64, 68},

			// 顶叶（Parietal Lobe）
			{9, 18, 23, 26, 30, 32, 44, 53, 58, 61, 65, 67},

			// 颞叶（Temporal Lobe）
			{2, 7, 8, 10, 16, 17, 31, 34, 35, 37, 42, 43, 45, 51, 52, 66, 69, 70},

			// 枕叶（Occipital Lobe）
			{6, 12, 14, 22, 41, 47, 49, 57}
		};

		region_matrix = {
			//{0, 1, 1, 0}, // 额叶（Frontal Lobe）
			//{1, 0, 1, 1}, // 顶叶（Parietal Lobe）
			//{1, 1, 0, 1}, // 颞叶（Temporal Lobe）
			//{0, 1, 1, 0}  // 枕叶（Occipital Lobe）
			{1, 1, 1, 1}, 
			{1, 1, 1, 1}, 
			{1, 1, 1, 1}, 
			{1, 1, 1, 1}  
		};

		m_width = 50;

		
		
	}

	void setData() {
		mGraph.set_visibility();
		generate_cluster(brainDivisions);
		generate_matrix();

		//update layout of matrix
		update_layout();

		generate_link();

		setBorderVis(true);
	}


	void draw_matrix() {

		matrix1.setArea(makeRectDisplayArea(matrix_pos[0], makeVec2f(m_width, 0.0f), makeVec2f(0.0f, m_width)));
		matrix2.setArea(makeRectDisplayArea(matrix_pos[1], makeVec2f(m_width, 0.0f), makeVec2f(0.0f, m_width)));
		matrix3.setArea(makeRectDisplayArea(matrix_pos[2], makeVec2f(m_width, 0.0f), makeVec2f(0.0f, m_width)));
		matrix4.setArea(makeRectDisplayArea(matrix_pos[3], makeVec2f(m_width, 0.0f), makeVec2f(0.0f, m_width)));

		matrix1.display();
		matrix2.display();
		matrix3.display();
		matrix4.display();

		
	}
	
	//utils
	void rotatePositions(std::vector<vec2f>& matrix_pos, int angel,int side) {
		if (side == 0) {
			for (int idx = 0; idx < 4; idx++) {
				vec2f element = matrix_pos[idx];
				auto it = std::find(matrix_layout_left.begin(), matrix_layout_left.end(), element);
				int pos;
				if (it != matrix_layout_left.end()) {
					pos = std::distance(matrix_layout_left.begin(), it); // 返回元素的索引
				}
				else {
					pos = -1;
				}
				int new_pos = (pos + angel) % 4;
				matrix_pos[idx] = matrix_layout_left[new_pos];
			}
		}
		if (side == 1) {
			for (int idx = 0; idx < 4; idx++) {
				vec2f element = matrix_pos[idx];
				auto it = std::find(matrix_layout_right.begin(), matrix_layout_right.end(), element);
				int pos;
				if (it != matrix_layout_right.end()) {
					pos = std::distance(matrix_layout_right.begin(), it); // 返回元素的索引
				}
				else {
					pos = -1;
				}
				int new_pos = (pos + angel) % 4;
				matrix_pos[idx] = matrix_layout_right[new_pos];
			}
		}
		
	}

	void swapPositions(std::vector<vec2f>& matrix_pos, int index1,int index2) {
		if (matrix_pos.size() != 4) {
			std::cerr << "Matrix must have 4 elements." << std::endl;
			return;
		}

		std::swap(matrix_pos[index1], matrix_pos[index2]);
	}
	void FRLayout(std::vector<vec2f>& positions, const std::vector<std::vector<int>>& adjacencyMatrix,RectDisplayArea range, int iterations = 100, float area = 10000.0, float k = 0.0) {
		float xmin = range.origin.x;
		float ymin = range.origin.y;
		float xmax = range.row_axis.x + xmin;
		float ymax = range.col_axis.y + ymin;
		float minDistance = 20;

		int numNodes = positions.size();
		if (k == 0.0) {
			k = std::sqrt(area / numNodes);
		}

		float temperature = area; // 初始温度

		for (int iter = 0; iter < iterations; ++iter) {
			std::vector<vec2f> displacements(numNodes, { 0.0f, 0.0f });

			// 计算排斥力
			for (int i = 0; i < numNodes; ++i) {
				for (int j = i + 1; j < numNodes; ++j) {
					if (i != j) {
						float dist = dist2d(positions[i], positions[j]);
						if (dist > 0.0f) {
							float repulsiveForce = (k * k) / dist;
							vec2f direction = { (positions[i].x - positions[j].x) / dist, (positions[i].y - positions[j].y) / dist };
							displacements[i].x += direction.x * repulsiveForce;
							displacements[i].y += direction.y * repulsiveForce;
							displacements[j].x -= direction.x * repulsiveForce;
							displacements[j].y -= direction.y * repulsiveForce;
						}
					}
				}
			}

			// 计算吸引力
			for (int i = 0; i < numNodes; ++i) {
				for (int j = 0; j < numNodes; ++j) {
					if (adjacencyMatrix[i][j] == 1) {
						float dist = dist2d(positions[i], positions[j]);
						if (dist > 0.0f) {
							float attractiveForce = (dist * dist) / k;
							vec2f direction = { (positions[j].x - positions[i].x) / dist, (positions[j].y - positions[i].y) / dist };
							displacements[i].x -= direction.x * attractiveForce;
							displacements[i].y -= direction.y * attractiveForce;
							displacements[j].x += direction.x * attractiveForce;
							displacements[j].y += direction.y * attractiveForce;
						}
					}
				}
			}

			// 更新节点位置
			for (int i = 0; i < 4; ++i) {
				float dispLength = dist2d(makeVec2f(0,0), displacements[i]);
				if (dispLength > 0.0f) {
					positions[i].x += (displacements[i].x / dispLength) * std::min(dispLength, temperature);
					positions[i].y += (displacements[i].y / dispLength) * std::min(dispLength, temperature);
				}

				// 限制坐标范围
				positions[i].x = std::max(xmin, std::min(positions[i].x, xmax));
				positions[i].y = std::max(ymin, std::min(positions[i].y, ymax));
			}

			// 降低温度
			temperature *= 0.9;
		}
	}
	void BLayout(std::vector<vec2f>& positions, RectDisplayArea range) {
		float xmin = range.origin.x;
		float ymin = range.origin.y;
		float xmax = range.row_axis.x + xmin;
		float ymax = range.col_axis.y + ymin;

		positions[0] = makeVec2f(xmin, ymax); 
		positions[1] = makeVec2f(xmax, ymax); 
		positions[2] = makeVec2f(xmax, ymin); 
		positions[3] = makeVec2f(xmin, ymin); 
	}

	std::vector<std::vector<int>> generateAdjacencyMatrix(const std::vector<int>& nodes, const std::vector<std::vector<int>>& edges) {
		int size = nodes.size();
		std::vector<std::vector<int>> adjacencyMatrix(size, std::vector<int>(size, 0));

		// 创建节点ID到矩阵索引的映射
		std::unordered_map<int, int> nodeIndex;
		for (int i = 0; i < size; ++i) {
			nodeIndex[nodes[i]] = i;
		}

		// 填充邻接矩阵
		for (const auto& edge : edges) {
			int node1 = edge[0];
			int node2 = edge[1];

			// 检查节点是否在映射中
			if (nodeIndex.find(node1) != nodeIndex.end() && nodeIndex.find(node2) != nodeIndex.end()) {
				int index1 = nodeIndex[node1];
				int index2 = nodeIndex[node2];
				adjacencyMatrix[index1][index2] = 1;
				adjacencyMatrix[index2][index1] = 1; // 无向图
			}
		}

		return adjacencyMatrix;
	}

	std::vector<std::vector<int>> mergeMatrices(const std::vector<std::vector<int>>& matrix1, const std::vector<std::vector<int>>& matrix2) {
		int size1 = matrix1.size();
		int size2 = matrix2.size();
		int newSize = size1 + size2;

		// 创建新矩阵，并初始化为0
		std::vector<std::vector<int>> mergedMatrix(newSize, std::vector<int>(newSize, 0));

		// 复制第一个矩阵到新矩阵的左上角
		for (int i = 0; i < size1; ++i) {
			for (int j = 0; j < size1; ++j) {
				mergedMatrix[i][j] = matrix1[i][j];
			}
		}

		// 复制第二个矩阵到新矩阵的右下角
		for (int i = 0; i < size2; ++i) {
			for (int j = 0; j < size2; ++j) {
				mergedMatrix[i + size1][j + size1] = matrix2[i][j];
			}
		}

		return mergedMatrix;
	}
	
	vector<float> calculate_distance_matrix(vector<int> IDs) {
		vector<float> ret;
		int n = IDs.size();

		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				ret.push_back(mGraph.getNodePositions()[IDs[i]]* mGraph.getNodePositions()[IDs[j]]);
			}
		}
		return ret;
	}

	vec2f calculate_mean_position(vector<int> IDs) {
		vec2f ret;
		int n = IDs.size();

		for (int i = 0; i < n; i++) {
			ret = ret + mGraph.getNodePositions()[IDs[i]];
		}

		return ret/n;
	}

	struct Link {
		vec2f start;   // 起点
		vec2f end;     // 终点
		int connectivity; // 连通性

		// 构造函数
		Link(const vec2f& s, const vec2f& e, int conn)
			: start(s), end(e), connectivity(conn) {}
	};

	int countConnections(int id, const std::vector<int>& idArray) {
		int count = 0;

		for (int i = 0; i < idArray.size(); ++i) {
			int checkId = idArray[i];
			if (connection[id][checkId] != 0) { // 非零表示存在连通性
				++count;
			}
		}

		return count;
	}

	// data
	void generate_cluster(std::vector<std::vector<int>>& brainDivisions) {
		brainDivisions.clear();
		visibility_nodes.clear();
		for (const auto& region : brainRegions) {
			std::vector<int> newRegion;
			for (int id : region) {
				newRegion.push_back(id - 1); // 将每个编号减1
			}
			brainDivisions.push_back(newRegion);
		}
		// 遍历 0-69 的编号，如果 mark 为 true，则从 brainDivisions 中删除该编号
		for (int i = 0; i < 70; ++i) {
			if (mGraph.mNodeVisbleMarks[i]) {
				visibility_nodes.push_back(i);
				for (auto& region : brainDivisions) {
					// 使用 remove-erase 习惯用法来删除编号 i
					region.erase(std::remove(region.begin(), region.end(), i), region.end());
				}
			}
		}
	}
	
	void generate_matrix() {
		//m1
		int size1 = brainDivisions[0].size();
		vector<float> dis_matrix1 = calculate_distance_matrix(brainDivisions[0]);
		matrix_pos[0] = mGraph.convertToScreenCoordinates(calculate_mean_position(brainDivisions[0]));
		
		float* dm1_ptr = dis_matrix1.data();
		MatrixData<float> dm1(size1, size1, dm1_ptr);
		matrix1_data = dm1.convert<float>();
		matrix1.setData(matrix1_data);
		matrix1.set_range(MINMAX);

		//m2
		int size2 = brainDivisions[1].size();
		vector<float> dis_matrix2 = calculate_distance_matrix(brainDivisions[1]);
		matrix_pos[1] = mGraph.convertToScreenCoordinates(calculate_mean_position(brainDivisions[1]));
		
		float* dm2_ptr = dis_matrix2.data();
		MatrixData<float> dm2(size2, size2, dm2_ptr);
		matrix2_data = dm2.convert<float>();
		matrix2.setData(matrix2_data);
		matrix2.set_range(MINMAX);

		//m3
		int size3 = brainDivisions[2].size();
		vector<float> dis_matrix3 = calculate_distance_matrix(brainDivisions[2]);
		matrix_pos[2] = mGraph.convertToScreenCoordinates(calculate_mean_position(brainDivisions[2]));
		
		float* dm3_ptr = dis_matrix3.data();
		MatrixData<float> dm3(size3, size3, dm3_ptr);
		matrix3_data = dm3.convert<float>();
		matrix3.setData(matrix3_data);
		matrix3.set_range(MINMAX);

		//m4
		int size4 = brainDivisions[3].size();
		vector<float> dis_matrix4 = calculate_distance_matrix(brainDivisions[3]);
		matrix_pos[3] = mGraph.convertToScreenCoordinates(calculate_mean_position(brainDivisions[3]));
		
		float* dm4_ptr = dis_matrix4.data();
		MatrixData<float> dm4(size4, size4, dm4_ptr);
		matrix4_data = dm4.convert<float>();
		//adjustMatrix(matrix4_data, "-");
		matrix4.setData(matrix4_data);
		matrix4.set_range(MINMAX);
		
		//matrix4.set_range(MAXMIN);

	}

	void generate_link() {
		
		//links.clear();
		//outside link
		for (auto node : (*pSelectedROIs)) {
			for (int i = 0; i < 4; i++) {
				int cnt = countConnections(node, brainDivisions[i]);
				vec2f end = makeVec2f(matrix_pos[i].x + m_width/2, matrix_pos[i].y + m_width/2);
				Link edge(mGraph.getNodeDisplayPosition()[node], end, cnt);
				links.push_back(edge);
			}
		}
		//inside link
		/*for (int i = 0; i < region_matrix.size(); ++i) {
			for (int j = 0; j < region_matrix[i].size(); ++j) {
				if (region_matrix[i][j] != 1) continue;
				vec2f start = makeVec2f(matrix_pos[i].x + m_width/2, matrix_pos[i].y + m_width/2);
				vec2f end = makeVec2f(matrix_pos[j].x + m_width/2, matrix_pos[j].y + m_width/2);
				Link edge(start, end, 1);
				links.push_back(edge);
			}
			
		}*/

		//dashed link
		dashed_links.clear();
		for (const auto& edge : mGraph.mEdgeDatas) {
			int node1 = edge[0];
			int node2 = edge[1];

			if (std::find((*pSelectedROIs).begin(), (*pSelectedROIs).end(), node1) == (*pSelectedROIs).end()) {
				//check region
				for (size_t i = 0; i < brainRegions.size(); ++i) {
					if (std::find(brainRegions[i].begin(), brainRegions[i].end(), node1+1) != brainRegions[i].end()) {
						vec2f end = makeVec2f(matrix_pos[i].x + m_width / 2, matrix_pos[i].y + m_width / 2);
						Link edge(mGraph.getNodeDisplayPosition()[node1], end, 1);
						dashed_links.push_back(edge);
					}
					
				}
			
			}

			if (std::find((*pSelectedROIs).begin(), (*pSelectedROIs).end(), node2) == (*pSelectedROIs).end()) {
				//check region
				for (size_t i = 0; i < brainRegions.size(); ++i) {
					if (std::find(brainRegions[i].begin(), brainRegions[i].end(), node2+1) != brainRegions[i].end()) {
						vec2f end = makeVec2f(matrix_pos[i].x + m_width / 2, matrix_pos[i].y + m_width / 2);
						Link edge(mGraph.getNodeDisplayPosition()[node2], end, 1);
						dashed_links.push_back(edge);
					}

				}

			}
		}


		update_LinkWidth(2, 15);
		update_LinkColor(0.4, 0.8);

	}

	void update_LinkWidth(float target_min, float target_max) {
		// 确保 link_width 的大小与 links 相同
		link_width.resize(links.size());

		// 找到连接强度的最小值和最大值
		int source_min = 0;
		int source_max = 30;

		for (const auto& link : links) {
			source_min = std::min(source_min, link.connectivity);
			source_max = std::max(source_max, link.connectivity);
		}

		// 遍历链接并插值
		for (size_t i = 0; i < links.size(); ++i) {
			link_width[i] = interpolate(
				static_cast<float>(source_min),  // a1
				static_cast<float>(source_max),  // b1
				static_cast<float>(links[i].connectivity), // c1
				target_min, // a2
				target_max  // b2
			);
		}
	}

	void update_LinkColor(float target_min, float target_max) {
		vec4f base_color = ColorMap::getColorByName(ColorMap::Air_Force_blue);
		// 确保 link_width 的大小与 links 相同
		link_color.resize(links.size());

		// 找到连接强度的最小值和最大值
		int source_min = 0;
		int source_max = 30;

		for (const auto& link : links) {
			source_min = std::min(source_min, link.connectivity);
			source_max = std::max(source_max, link.connectivity);
		}

		// 遍历链接并插值
		for (size_t i = 0; i < links.size(); ++i) {
			float a = interpolate(
				static_cast<float>(source_min),  // a1
				static_cast<float>(source_max),  // b1
				static_cast<float>(links[i].connectivity), // c1
				target_min, // a2
				target_max  // b2
			);
			vec4f temp_color = base_color;
			temp_color.a = a;
			link_color[i] = temp_color;
		}
	}

	void update_layout() {

		FRLayout(matrix_pos, region_matrix,mArea);
		

		if (layout_update) {

		}

	}

	void display() {
		
		draw_links();
		mGraph.visibility_display();
		draw_matrix();

		if (border_visibility) {
			DisplayWidget::drawRectBorder(matrix_pos[0], makeVec2f(matrix_pos[0].x + m_width, matrix_pos[0].y + m_width), 3, matrix1_color);
			DisplayWidget::drawRectBorder(matrix_pos[1], makeVec2f(matrix_pos[1].x + m_width, matrix_pos[1].y + m_width), 3, matrix2_color);
			DisplayWidget::drawRectBorder(matrix_pos[2], makeVec2f(matrix_pos[2].x + m_width, matrix_pos[2].y + m_width), 3, matrix3_color);
			DisplayWidget::drawRectBorder(matrix_pos[3], makeVec2f(matrix_pos[3].x + m_width, matrix_pos[3].y + m_width), 3, matrix4_color);
		}
		

	}

	void draw_links() {
		vec4f color = ColorMap::getColorByName(ColorMap::Air_Force_blue);
		size_t idx = 0;
		for (auto edge : links) {
			//if (edge.connectivity == 0) continue;
			//DisplayWidget::drawLine(edge.start, edge.end, 2, color);
			DisplayWidget::drawCurve(edge.start, edge.end,1,0,0,link_width[idx], link_color[idx]);
			idx++;
		}

		for (auto edge : dashed_links) {
			if (edge.connectivity == 0) continue;
			//DisplayWidget::drawLine(edge.start, edge.end, 2, color);
			DisplayWidget::drawDashedLine(edge.start, edge.end, 1, color);
		}

	}

	

	void setArea(const RectDisplayArea& area) {
		mArea = area;
		mGraph.setArea(area);
	}

	void setBorderVis(bool flag) {
		border_visibility = flag;
	}

	void setLayoutUpdate(bool flag) { layout_update = flag; }

	void setConnection(std::string subject) {
		string file_path = "D:/DATA/brain/connections/" + subject + ".adj";

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
	}


//private:
	//display
	RectDisplayArea mArea;
	GraphDisplay<float> mGraph;
	bool layout_update = false;

	MatrixDisplay<float> matrix1;
	MatrixDisplay<float> matrix2;
	MatrixDisplay<float> matrix3;
	MatrixDisplay<float> matrix4;

	//data
	std::vector<vec2f> mData;
	std::vector<std::vector<int>> brainRegions;
	std::vector<std::vector<int>> brainDivisions;
	std::vector<std::vector<int>> region_matrix;
	std::vector<std::vector<int>> connection;
	std::vector<int> visibility_nodes;//index
	std::vector<int>* pSelectedROIs;
	std::vector<Link> links;
	std::vector<float> link_width;
	std::vector<vec4f> link_color;
	std::vector<Link> dashed_links;

	//matrix data
	MatrixData<float>* matrix1_data;
	MatrixData<float>* matrix2_data;
	MatrixData<float>* matrix3_data;
	MatrixData<float>* matrix4_data;

	vector<vec2f> matrix_pos= {
		makeVec2f(0, 0),
		makeVec2f(3.0f, 4.0f),
		makeVec2f(5.0f, 6.0f),
		makeVec2f(7.0f, 8.0f)
	};

	vector<vec2f> matrix_layout_left = {
		makeVec2f(30, 510),
		makeVec2f(30, 760),
		makeVec2f(310, 760),
		makeVec2f(310, 510)
	};

	vector<vec2f> matrix_layout_right = {
		makeVec2f(660, 510),
		makeVec2f(660, 760),
		makeVec2f(940, 760),
		makeVec2f(940, 510)
	};

	//para
	float m_width;
	bool border_visibility = false;
	//color matrix
	vec4f matrix1_color = ColorMap::getColorByName(ColorMap::Bright_lavender);
	vec4f matrix2_color = ColorMap::getColorByName(ColorMap::Bright_turquoise);
	vec4f matrix3_color = ColorMap::getColorByName(ColorMap::Black);
	vec4f matrix4_color = ColorMap::getColorByName(ColorMap::Stil_de_grain_yellow);


};

