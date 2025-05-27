#pragma once

#include "typeOperation.h"
#include "DisplayWidget.h"
#include "ColorMap.h"

class LineChartDisplay {
public:
	LineChartDisplay() {
		mRadius = 8.0f;
		mNodeColor = ColorMap::getColorByName(ColorMap::Bottle_green);
		mLineColor= ColorMap::getColorByName(ColorMap::Black);
	}

	void setData(const std::vector<std::vector<vec2f>>& data, const bool& b_sort) {
		mData.assign(data.begin(), data.end());
		if (b_sort) {
			for (int i = 0; i < mData.size(); ++i) {
				std::sort(mData[i].begin(), mData[i].end());
			}
		}
	}

	vec2f getNodePosition(const int& i, const int& j) {
		float x = interpolate(mXRange.lower, mXRange.upper, mData[i][j].x, 0.0f, 1.0f);
		float y = interpolate(mYRange.lower, mYRange.upper, mData[i][j].y, 0.0f, 1.0f);
		return mArea.origin + x * mArea.row_axis + y * mArea.col_axis;
	}

	void drawNodes() {
		for (int i = 0; i < mData.size(); ++i) {
			for (int j = 0; j < mData[i].size(); ++j) {
				DisplayWidget::drawCircle(getNodePosition(i, j), mRadius, mNodeColor, 2.0f, makeVec4f(1.0f));
			}
		}
	}

	void drawLines() {
		glPushAttrib(GL_CURRENT_BIT);
		glColor(mLineColor);
		for (int i = 0; i < mData.size(); ++i) {
			glBegin(GL_LINE_STRIP);
			for (int j = 0; j < mData[i].size(); ++j) {
				glVertex(getNodePosition(i, j));
			}
			glEnd();
		}
	}

	void display() {
		drawLines();
		drawNodes();
	}

	void setRange(const float& x_lower, const float& x_upper,
		const float& y_lower, const float& y_upper)
	{
		mXRange = makeRange(x_lower, x_upper);
		mYRange = makeRange(y_lower, y_upper);
	}

	void setArea(const RectDisplayArea& area) {
		mArea = area;
	}

private:
	vec4f mNodeColor;
	vec4f mLineColor;
	float mRadius;
	RectDisplayArea mArea;
	Range mXRange, mYRange;
	std::vector<std::vector<vec2f>> mData;
};


//project the line chart on the horizontal axis
//the y-coordinate map to color
//therefore, each line chart becomes a line (long rectangle) with varying color

class LineChartMarginDisplay : public DisplayBase {
public:
	LineChartMarginDisplay() {
		mColorScheme = COLOR_MAP_SPECTRUAL;
		mSelectedColor = ColorMap::getColorByName(ColorMap::Harvard_crimson);
		mSelectedRangeColor = makeVec4f(1.0f, 1.0f, 1.0f, 0.2f);
		mSelectedYRange = makeRange(1e30, -1e30);
		mMargin = 0.1f;
	}

	void display() {
		glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT);

		glDisable(GL_LIGHTING);

		vec2f col_unit = getColumnUnit();
		vec2f col_axis = (1.0f - 2.0f * mMargin) * col_unit;
		vec2f origin = mArea.origin + mMargin * col_unit;

		for (int i = 0; i < mData.size(); ++i) {
			glBegin(GL_QUAD_STRIP);
			for (int j = 0; j < mData[i].size(); ++j) {
				if (inRange(mXRange, mData[i][j].x)) {
					float x = interpolate(mXRange.lower, mXRange.upper, mData[i][j].x, 0.0f, 1.0f);
					float y = interpolate(mYRange.lower, mYRange.upper, mData[i][j].y, 0.0f, 1.0f);
					glColor(ColorMap::getLinearColor(y, mColorScheme));
					vec2f p = origin + x * mArea.row_axis;
					glVertex(p);
					glVertex(p + col_axis);
				}
			}
			glEnd();
			origin += col_unit;
		}

		RectDisplayArea area = makeRectDisplayArea(mArea.origin, mArea.row_axis, col_unit);
		for (int i = 0; i < mSelectedLines.size(); ++i) {
			area.origin = mSelectedLines[i] * col_unit + mArea.origin;
			DisplayWidget::drawRect(area, makeVec4f(0.0f), 2.0f, mSelectedColor);
		}

		if (isYRangeSelected()) {
			float y_low = interpolate(mYRange.lower, mYRange.upper, mSelectedYRange.lower, 0.0f, 1.0f);
			float y_up = interpolate(mYRange.lower, mYRange.upper, mSelectedYRange.upper, 0.0f, 1.0f);
			area = makeRectDisplayArea(mArea.origin + mArea.row_axis * y_low, (y_up - y_low) * mArea.row_axis, mArea.col_axis);
			DisplayWidget::drawRect(area, mSelectedRangeColor, 1.0f, makeVec4f(0.5f, 0.5f, 0.5f, 1.0f));
		}

		glPopAttrib();
	}

	vec2f getColumnUnit() {
		return mArea.col_axis / numColumnElements();
	}

	int getLineIdWithPos(const vec2f& pos) {
		vec2f rp = pos - mArea.origin;
		vec2f col_unit = getColumnUnit();
		int id = (int)((rp * col_unit) / (col_unit * col_unit));
		float y = rp * mArea.row_axis / (mArea.row_axis * mArea.row_axis);
		if (isLineIDValid(id) && y >= 0.0f && y <= 1.0f) {
			return id;
		}
		return -1;
	}

	int getClickElement(const vec2f& p) {
		return getLineIdWithPos(p);
	}

	void setSelectElement(const int& eid) {
		selectLine(eid);
	}

	float getYValueWithPos(const vec2f& pos) {
		vec2f rp = pos - mArea.origin;
		float y = rp * mArea.row_axis / (mArea.row_axis * mArea.row_axis);
		return interpolate(0.0f, 1.0f, y, mYRange.lower, mYRange.upper);
	}

	void setData(const std::vector<std::vector<vec2f>>& data, const bool& b_sort = true) {
		mData.assign(data.begin(), data.end());
		if (b_sort) {
			for (int i = 0; i < mData.size(); ++i) {
				std::sort(mData[i].begin(), mData[i].end());
			}
		}
	}

	bool isLineIDValid(const int& line_id) {
		return (line_id >= 0 && line_id < mData.size());
	}

	void selectLine(const int& line_id) {
		mSelectedLines.clear();
		if (isLineIDValid(line_id)) {
			mSelectedLines.push_back(line_id);
		}
	}

	void selectLines(const std::vector<int>& selected_lines) {
		mSelectedLines.assign(selected_lines.begin(), selected_lines.end());
	}

	void selectYRange(Range r) {
		if (r.lower > r.upper) std::swap(r.lower, r.upper);
		r.lower = clamp(r.lower, mYRange.lower, mYRange.upper);
		r.upper = clamp(r.upper, mYRange.lower, mYRange.upper);

		if ((r.upper - r.lower) / (mYRange.upper - mYRange.lower) > 0.02f) {
			mSelectedYRange = r;
		}
	}

	Range getSelectedYRange() { return mSelectedYRange; }

	Range getYRange() { return mYRange; }
	Range getXRange() { return mXRange; }

	int numRowElements() {
		if (mData.empty()) return 0;
		return mData[0].size();
	}

	int numColumnElements() { return mData.size(); }

	bool isYRangeSelected() { return (mSelectedYRange.lower < mSelectedYRange.upper); }

	void clearSelection() {
		mSelectedLines.clear();
	}

	void setColorScheme(const LinearColorMapType& color_map) {
		mColorScheme = color_map;
	}

	void setRange(const float& x_lower, const float& x_upper,
		const float& y_lower, const float& y_upper)
	{
		mXRange = makeRange(x_lower, x_upper);
		mYRange = makeRange(y_lower, y_upper);
	}

	void setArea(const RectDisplayArea& area) {
		mArea = area;
	}

	void setMargin(const float& margin) {
		mMargin = margin;
	}

private:
	float mMargin;
	LinearColorMapType mColorScheme;
	Range mSelectedYRange;
	Range mXRange, mYRange;
	std::vector<std::vector<vec2f>> mData;
	std::vector<int> mSelectedLines;
	vec4f mSelectedColor;
	vec4f mSelectedRangeColor;
};