#pragma once
#pragma once

#include "typeOperation.h"
#include "DisplayWidget.h"
#include "ColorMap.h"

inline bool in_range(const float& f, const float& center, const float& radius) {
	return (f > center - radius && f < center + radius);
}

class FilterItem {
public:
	FilterItem() {
		mBorderColor = ColorMap::getColorByName(ColorMap::Ash_grey);
		mFillColor = ColorMap::getColorByName(ColorMap::Bubble_gum, 0.5f);
		mHighlightBorderColor = ColorMap::getColorByName(ColorMap::Harvard_crimson);
		mHighlightFillColor = ColorMap::getColorByName(ColorMap::Coral_pink, 0.5f);
		mbInInteraction = false;
		mClickThresh = 5.0f;
	}

	vec4f border_color() { return mBorderColor; }
	vec4f fill_color() { return mFillColor; }
	vec4f highlight_border_color() { return mHighlightBorderColor; }
	vec4f highlight_fill_color() { return mHighlightFillColor; }
	bool in_interaction() { return mbInInteraction; }
	void set_click_thresh(const float& thresh) { mClickThresh = thresh; }

	virtual bool mouse_over(const vec2f& p) = 0;
	virtual bool press(const vec2f& p) = 0;
	virtual void drag(const vec2f& p) = 0;
	virtual void release(const vec2f& p) = 0;

protected:
	vec4f mBorderColor;
	vec4f mFillColor;
	vec4f mHighlightFillColor;
	vec4f mHighlightBorderColor;
	float mClickThresh;
	bool mbInInteraction;
};

class LinearRangeItem : public FilterItem {
public:
	typedef enum {
		OUTSIDE = 0,
		INSIDE,
		LEFT_BOUNDARY,
		RIGHT_BOUNDARY
	} Component;

	LinearRangeItem(const RectDisplayArea& area = makeRectDisplayArea(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
		const Range& range = makeRange(0.0f, 1.0f))
		:FilterItem()
	{
		mRange = range;
		set_area(area);
		mClickThresh = 0.02f;
		mbInInteraction = false;
		mDragComponent = OUTSIDE;
	}

	void display() {
		if (mRange.lower >= mRange.upper) return;

		RectDisplayArea ra = mArea;
		ra.origin += mRange.lower * ra.row_axis;
		ra.row_axis *= mRange.upper - mRange.lower;
		if (mDragComponent == OUTSIDE) {
			DisplayWidget::drawRect(ra, fill_color(), 2.0f, border_color());
		}
		else if (mDragComponent == INSIDE) {
			DisplayWidget::drawRect(ra, highlight_fill_color(), 2.0f, highlight_border_color());
		}
		else {
			DisplayWidget::drawRect(ra, highlight_fill_color(), 2.0f, border_color());
			if (mDragComponent == LEFT_BOUNDARY || mDragComponent == RIGHT_BOUNDARY) {
				if (mDragComponent == RIGHT_BOUNDARY) ra.origin += ra.row_axis;
				ra.row_axis = makeVec2f(0.0f, 0.0f);
				DisplayWidget::drawRect(ra, fill_color(), 4.0f, highlight_border_color());
			}
		}
	}

	void set_area(const RectDisplayArea& area) {
		mArea = area;
		update_unit_axes();
	}

	void set_range(const Range& range) {
		mRange = range;
		mRange.lower = clamp(mRange.lower, 0.0f, 1.0f);
		mRange.upper = clamp(mRange.upper, 0.0f, 1.0f);
	}

	Range get_range() {
		return mRange;
	}

	bool reach_lower() {
		return in_range(mRange.lower, 0.0f, mClickThresh);
	}

	bool reach_upper() {
		return in_range(mRange.upper, 1.0f, mClickThresh);
	}

	void update_unit_axes() {
		mUnitRow = unit_vec(mArea.row_axis);
		mUnitCol = unit_vec(mArea.col_axis);
	}

	void update_drag_component(const vec2f& n) {//take normalized point as input
		mDragComponent = OUTSIDE;
		if (n.y < 0.0f || n.y > 1.0f) return; //out of vertical range, do nothing
		if (in_range(n.x, mRange.lower, mClickThresh)) {//click left boundary
			mDragComponent = LEFT_BOUNDARY;
		}
		else if (in_range(n.x, mRange.upper, mClickThresh)) {//click right boundary
			mDragComponent = RIGHT_BOUNDARY;
		}
		else if (n.x > mRange.lower + mClickThresh && n.x < mRange.upper - mClickThresh) {//click inside
			mDragComponent = INSIDE;
			mDragInsideRange = makeRange(n.x - mRange.lower, 1.0f - (mRange.upper - n.x));
		}
	}

	bool mouse_over(const vec2f& p) {
		vec2f n = mArea.display2norm(p);
		update_drag_component(n);
		return (mDragComponent != OUTSIDE);
	}

	bool press(const vec2f& p) {
		mbInInteraction = false;
		vec2f n = mArea.display2norm(p);
		update_drag_component(n);
		if (mDragComponent == OUTSIDE && (n.x > 0.0f && n.x < 1.0f && n.y >0.0f && n.y < 1.0f)) {//change range
			mDragComponent = LEFT_BOUNDARY;
			mRange.lower = mRange.upper = n.x;
		}
		if (mDragComponent != OUTSIDE) mbInInteraction = true;
		return mbInInteraction;
	}

	void drag(const vec2f& p) {
		if (!in_interaction()) return;
		vec2f n = mArea.display2norm(p);
		if (mDragComponent == LEFT_BOUNDARY) {
			mRange.lower = clamp(n.x, 0.0f, 1.0f);
		}
		else if (mDragComponent == RIGHT_BOUNDARY) {
			mRange.upper = clamp(n.x, 0.0f, 1.0f);
		}
		else if (mDragComponent == INSIDE) {
			float x = clamp(n.x, mDragInsideRange.lower, mDragInsideRange.upper);
			mRange.lower = x - mDragInsideRange.lower;
			mRange.upper = x + (1.0f - mDragInsideRange.upper);
		}

		if (mRange.lower > mRange.upper) {//swap boundary
			std::swap(mRange.lower, mRange.upper);
			mDragComponent = (mDragComponent == LEFT_BOUNDARY) ? RIGHT_BOUNDARY : LEFT_BOUNDARY;
		}
	}

	void release(const vec2f& p) {
		if (!in_interaction()) return;
		drag(p);
		mDragComponent = OUTSIDE;
		mbInInteraction = false;
	}

private:
	Range mRange;
	RectDisplayArea mArea;
	vec2f mUnitRow, mUnitCol;

	//status
	Component mDragComponent;
	Range mDragInsideRange;
};

class CircleFilterItem : public FilterItem {
public:
	typedef enum {
		OUTSIDE = 0,
		INSIDE,
		BOUNDARY
	} Component;

	CircleFilterItem(const vec2f& center, const float& radius) {
		mDragComponent = OUTSIDE;
		mClickThresh = 3.0f;
		mCenter = center;
		mRadius = radius;
	}

	void display() {
		if (mDragComponent == OUTSIDE) {//draw circle
			DisplayWidget::drawCircle(mCenter, mRadius, fill_color(), 2.0f, border_color());
		}
		else if (mDragComponent == INSIDE) {//highlight the entire circle and draw the center
			DisplayWidget::drawCircle(mCenter, mRadius, highlight_fill_color(), 2.0f, highlight_border_color());
			DisplayWidget::drawCircle(mCenter, mClickThresh, makeVec4f(0.0f), 2.0f, border_color());
		}
		else if (mDragComponent == BOUNDARY) {//highlight the entire circle with thicker border
			DisplayWidget::drawCircle(mCenter, mRadius, highlight_fill_color(), 4.0f, highlight_border_color());
			DisplayWidget::drawCircle(mCenter, mClickThresh, makeVec4f(0.0f), 2.0f, border_color());
		}
	}

	void update_drag_component(const vec2f& p) {//take point in widget space
		mDragComponent = OUTSIDE;
		float d = length(p - mCenter);
		if (d < mRadius - mClickThresh) {//click the center
			mDragComponent = INSIDE;
		}
		else if (d < mRadius + mClickThresh) {//click boundary
			mDragComponent = BOUNDARY;
		}
	}

	void reset_status() {
		mDragComponent = OUTSIDE;
	}

	bool is_in(const vec2f& p) {
		float d = length(p - mCenter);
		return (d < mRadius + mClickThresh);
	}

	bool mouse_over(const vec2f& p) {
		update_drag_component(p);
		return (mDragComponent != OUTSIDE);
	}

	bool press(const vec2f& p) {
		if (mouse_over(p)) {
			if (mDragComponent == INSIDE) {
				mDragCenterOffset = mCenter - p;
			}
			mbInInteraction = true;
		}
		else {
			mbInInteraction = false;//not necessary if release was called
		}
		return mbInInteraction;
	}

	void drag(const vec2f& p) {
		if (mDragComponent == INSIDE) {//move the entire filter
			mCenter = p + mDragCenterOffset;
		}
		else if (mDragComponent == BOUNDARY) {//move the boundary
			mRadius = length(p - mCenter);
		}
	}

	void release(const vec2f& p) {
		mbInInteraction = false;
	}

	vec2f center() { return mCenter; }
	float radius() { return mRadius; }
	void set_center(const vec2f& center) { mCenter = center; }
	void set_radius(const float& radius) { mRadius = radius; }

private:
	vec2f mCenter;
	float mRadius;
	Component mDragComponent;
	vec2f mDragCenterOffset;
};