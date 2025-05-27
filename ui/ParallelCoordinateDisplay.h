#pragma once
#pragma once

#include "typeOperation.h"
#include "DisplayWidget.h"
#include "ColorMap.h"

class ParallelCoordinateDisplay {
public:
	ParallelCoordinateDisplay() {
		mRadius = 8.0f;
		mNodeColor = ColorMap::getColorByName(ColorMap::Bottle_green);
		mLineColor = ColorMap::getColorByName(ColorMap::Royal_purple);
		mAxColor = ColorMap::getColorByName(ColorMap::Black);
		mBackgoundColor = ColorMap::getColorByName(ColorMap::Lavender_gray);
		mCompColor= ColorMap::getColorByName(ColorMap::Orange_Yellow);
		mLineWidth = 4;
		mAxWidth = 4;
	}

	void setData(const std::vector<std::vector<vec2f>>& data, const bool& b_sort) {
		mData.assign(data.begin(), data.end());
		if (b_sort) {
			for (int i = 0; i < mData.size(); ++i) {
				std::sort(mData[i].begin(), mData[i].end());
			}
		}
		num_attr = mData.size();
	}

	vec2f getNodePosition(const int& i, const int& j) {
		float x = interpolate(0, num_attr, i, 0.0f, 1.0f);
		float y = interpolate(mRanges[i].lower, mRanges[i].upper, mData[i][j].y, 0.0f, 1.0f);
		return mArea.origin + x * mArea.row_axis + y * mArea.col_axis;
	}

	void drawNodes() {
		for (int i = 0; i < mData.size(); ++i) {
			for (int j = 0; j < mData[i].size(); ++j) {
				DisplayWidget::drawCircle(getNodePosition(i, j), mRadius, mNodeColor, 2.0f, makeVec4f(1.0f));
			}
		}
	}

	void drawAttrAxes() {
		for (int i = 0; i < mData.size(); ++i) {
			for (int j = 0; j < mData[i].size() - 1; ++j) {
				vec2f p1 = getNodePosition(i, j);    
				vec2f p2 = getNodePosition(i, j + 1); 
				DisplayWidget::drawLine(p1, p2, mAxWidth, mAxColor);
			}
		}
	}

	void drawLines() {
		for (int j = 0; j < mData[0].size(); ++j) { 
			//draw background lines
			for (int i = 0; i < mData.size() - 1; ++i)if (find(selected_indices.begin(), selected_indices.end(), j) == selected_indices.end()) {
				vec2f p1 = getNodePosition(i, j);    
				vec2f p2 = getNodePosition(i + 1, j); 
				DisplayWidget::drawLine(p1, p2, 2, mBackgoundColor);
				
			}
		}

		for (int j = 0; j < mData[0].size(); ++j) {
			//draw compare lines
			for (int i = 0; i < mData.size() - 1; ++i)if (find(comp_indices.begin(), comp_indices.end(), j) != comp_indices.end()) {
				vec2f p1 = getNodePosition(i, j);
				vec2f p2 = getNodePosition(i + 1, j);
				DisplayWidget::drawLine(p1, p2, mLineWidth, mCompColor);

			}
		}

		for (int j = 0; j < mData[0].size(); ++j) {
			//draw selected lines
			for (int i = 0; i < mData.size() - 1; ++i)if (find(selected_indices.begin(), selected_indices.end(), j) != selected_indices.end()) {
				vec2f p1 = getNodePosition(i, j);
				vec2f p2 = getNodePosition(i + 1, j);
				DisplayWidget::drawLine(p1, p2, mLineWidth, mLineColor);

			}
		}


	}

	void display() {
		drawAttrAxes();
		drawLines();
		//drawNodes();
	}

	void set_selected(int idx) {selected_indices.push_back(idx);}

	void set_selected(std::vector<int> arr) {selected_indices=arr;}

	void set_comp_selected(std::vector<int> arr) { comp_indices = arr; }

	void setRange(const float& lower, const float& upper)
	{
		for (int i = 0; i < num_attr; i++) {
			mRanges[i] = makeRange(lower, upper);
		}
	}

	void set_range(NormScheme scheme) {
		mRanges.clear();
		if (scheme == MINMAX) {
			for (int i = 0; i < num_attr; i++) {
				std::vector<vec2f> data_array = mData[i];
				std::vector<float> y_values;
				int size = data_array.size();
				float minv, maxv;
				y_values.reserve(size); 
				for (const vec2f& vec : data_array) {
					y_values.push_back(vec.y);
				}

				float* y_ptr = y_values.data();
				computeMinMax(y_ptr, size, minv, maxv);
				mRanges.push_back(makeRange(minv, maxv));
			}
			
			
		}

	}

	void setArea(const RectDisplayArea& area) {
		mArea = area;
	}

	void clear_selected() {selected_indices.clear();}

private:
	vec4f mNodeColor;
	vec4f mLineColor;
	vec4f mAxColor;
	vec4f mBackgoundColor;
	vec4f mCompColor;

	float mLineWidth;
	float mAxWidth;
	float mRadius;

	RectDisplayArea mArea;

	int num_attr;
	std::vector<int> selected_indices;
	std::vector<int> comp_indices;

	std::vector<Range> mRanges;
	std::vector<std::vector<vec2f>> mData;
};

