#pragma once
#pragma once

#include "typeOperation.h"
#include "ColorMap.h"
#include "DisplayWidget.h"
#include "InteractiveItem.h"

class BarDisplay {
public:

	typedef enum {
		OUT_OF_SELECT = 0,
		LEFT_SELECT_BORDER,
		RIGHT_SELECT_BORDER,
		CENTER_OF_SELECT
	} ClickType;

	BarDisplay() {
		mBarColor = ColorMap::getColorByName(ColorMap::Baby_blue);
		mSelectedBarColor = ColorMap::getColorByName(ColorMap::Bubble_gum, 0.7f);
		mSelectedBarBorderColor = ColorMap::getColorByName(ColorMap::Ash_grey);
		mMargin = 0.05f;
	}

	void set_data(const std::vector<Range>& data) {
		mData.assign(data.begin(), data.end());
	}

	template <typename T>
	void set_data(const std::vector<T>& data, const float& lower = 0.0f) {
		mData.assign(data.size(), makeRange(lower, lower));
		for (int i = 0; i < data.size(); ++i) {
			mData[i].upper = data[i];
		}
	}

	void set_domain(const Range& domain) {
		mDomain = domain;
		mSelectedDomain.set_range(makeRange(1.0f, -1.0f));
	}

	void set_selected_domain(const Range& selected_domain) {
		mSelectedDomain.set_range(get_relative_range(selected_domain, mDomain));
	}

	void set_range(const float& lower, const float& upper) {
		mRange = makeRange(lower, upper);
	}

	void set_range(NormScheme scheme) {
		if (scheme == MINMAX) {
			//computeMinMax()
			float minv = 0;
			int maxv = INT_MIN;
			for (const auto& range : mData) {
				maxv = maxv > range.upper ? maxv : range.upper;
			}
			mRange = makeRange(minv, maxv);
		}

	}

	void set_area(const RectDisplayArea& area) {
		mArea = area;
		mSelectedDomain.set_area(area);
	}

	void set_margin(const float& margin) {
		mMargin = margin;
	}

	void clear() {
		mData.clear();
	}

	bool empty() {
		return mData.empty();
	}

	bool selection_reach_lower() {
		return mSelectedDomain.reach_lower();
	}

	bool selection_reach_upper() {
		return mSelectedDomain.reach_upper();
	}

	Range get_relative_range(const Range& seg, const Range& entire) {
		Range ret;
		ret.lower = interpolate(entire.lower, entire.upper, seg.lower, 0.0f, 1.0f);
		ret.upper = interpolate(entire.lower, entire.upper, seg.upper, 0.0f, 1.0f);
		return ret;
	}

	Range get_absolute_range(const Range& seg, const Range& entire) {
		Range ret;
		ret.lower = interpolate(0.0f, 1.0f, seg.lower, entire.lower, entire.upper);
		ret.upper = interpolate(0.0f, 1.0f, seg.upper, entire.lower, entire.upper);
		return ret;
	}

	Range get_selected_relative() {
		return mSelectedDomain.get_range();
	}


	Range get_selected_domain() {
		return get_absolute_range(mSelectedDomain.get_range(), mDomain);
	}

	RectDisplayArea get_selection_area() {
		RectDisplayArea area = mArea;
		Range range = mSelectedDomain.get_range();
		area.origin = mArea.norm2display(makeVec2f(range.lower, 0));
		area.row_axis = mArea.norm2display(makeVec2f(range.upper, 0)) - area.origin;
		return area;
	}

	RectDisplayArea getArea() {
		RectDisplayArea ret;
		ret.origin = mArea.origin;
		ret.row_axis = mArea.row_axis;
		ret.col_axis = mArea.col_axis;
		return ret;
	}

	void display() {
		glPushAttrib(GL_ENABLE_BIT);
		glDisable(GL_DEPTH_TEST);
		float fac = 1.0f - 2.0f * mMargin;
		vec2f inc = (1.0f / mData.size()) * mArea.row_axis;
		RectDisplayArea area;
		area.origin = mArea.origin + 0.5f * mMargin * mArea.col_axis;
		area.row_axis = fac * inc;
		area.col_axis = (1.0f - mMargin) * mArea.col_axis;
		for (int i = 0; i < mData.size(); ++i) {
			Range r = get_relative_range(mData[i], mRange);
			RectDisplayArea a = area;
			a.origin += r.lower * area.col_axis;
			a.col_axis = (r.upper - r.lower) * area.col_axis;
			DisplayWidget::drawRect(a, mBarColor, 0.0f);
			area.origin += inc;
		}
		mSelectedDomain.display();
		glPopAttrib();
	}

	void update_range() {
		if (mData.empty()) return;
		mRange = mData[0];
		for (int i = 1; i < mData.size(); ++i) {
			if (mRange.lower > mData[i].lower) mRange.lower = mData[i].lower;
			if (mRange.upper < mData[i].upper) mRange.upper = mData[i].upper;
		}
	}

	bool mouse_over(const vec2f& p) {
		mSelectedDomain.mouse_over(p);
		return mArea.is_in(p);
	}

	bool press(const vec2f& p) {
		return mSelectedDomain.press(p);
	}

	void drag(const vec2f& p) {
		mSelectedDomain.drag(p);
	}

	void release(const vec2f& p) {
		mSelectedDomain.release(p);
	}

	bool in_interaction() {
		return mSelectedDomain.in_interaction();
	}

	bool has_selected_domain() {
		Range r = mSelectedDomain.get_range();
		return (r.lower < r.upper);
	}

	RectDisplayArea area() {
		return mArea;
	}

	inline bool inDisplayArea(const vec2f& p) {
		return inRectDisplayArea(p, mArea);
	}

private:
	float mMargin;
	vec4f mBarColor;
	vec4f mSelectedBarColor;
	vec4f mSelectedBarBorderColor;
	RectDisplayArea mArea;
	Range mDomain, mRange;
	LinearRangeItem mSelectedDomain;
	std::vector<Range> mData;
};