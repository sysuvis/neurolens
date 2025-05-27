#ifndef _MATRIX_DISPLAY_H
#define _MATRIX_DISPLAY_H

#include "typeOperation.h"
#include "ColorMap.h"
#include "MatrixData.h"
#include "DisplayWidget.h"

template<typename T>
class MatrixDisplay:public DisplayBase{
public:
	enum SelectType {
		Column = 0,
		Row
	};
	MatrixDisplay() {
		mMargin = 0.05f;
		mbTranspose = false;
		mbHighlightCells = false;
		mColorSheme = COLOR_MAP_PERCEPTUAL;
		mCellColor = NULL;
		mData = NULL;
		mbHighlightSize = true;
		mHighlightSizeThresh = 1e30;
		mSelectFillColor = makeVec4f(0.0f);
		mHighlightColor = ColorMap::getColorByName(ColorMap::Black);
		mSelectType = Column;
		//mMatrixBorderColor = makeVec4f(0.0f);
	}

	~MatrixDisplay() {
		if (mCellColor) {
			delete mCellColor;
		}
	}

	MatrixData<T>* getData() { return mData; }
	void freeData() { delete mData; }

	void setFilter(const T& lower, const T& upper) {
		mFilter[0] = lower;
		mFilter[1] = upper;
	}

	void setNormalization(const float& domain_lower, const float& domain_upper,
		const float& range_lower, const float& range_upper)
	{
		mNormalizationDomain[0] = domain_lower;
		mNormalizationDomain[1] = domain_upper;
		mNormalizationRange[0] = range_lower;
		mNormalizationRange[1] = range_upper;
	}

	void set_range(NormScheme scheme) {
		if (scheme == MINMAX) {
			T minv, maxv;
			mData->updateMinMax();
			minv = mData->getMin();
			maxv = mData->getMax();

			//set filter
			mFilter[0] = minv;
			mFilter[1] = maxv;
			//set normalization 
			mNormalizationDomain[0] = minv;
			mNormalizationDomain[1] = maxv;
			mNormalizationRange[0] = 0;
			mNormalizationRange[1] = 1;


		}

		if (scheme == MAXMIN) {
			T minv, maxv;
			mData->updateMinMax();
			minv = mData->getMin();
			maxv = mData->getMax();

			// Set filter (if required)
			mFilter[0] = minv;
			mFilter[1] = maxv;

			// Set normalization domain (same as min-max normalization)
			mNormalizationDomain[0] = minv;
			mNormalizationDomain[1] = maxv;

			// Reverse normalization: Map large values to small ones and vice versa
			mNormalizationRange[0] = 1; // The reversed normalization will start at 1
			mNormalizationRange[1] = 0; // The reversed normalization will end at 0

		}
		
		if (scheme == UNIFORM) {
			mData->updateMinMax();
			T minv = mData->getMin();
			T maxv = mData->getMax();
			int width = mData->width();
			int height = mData->height();
			// Scale each value linearly to a new range [a, b]
			T a = 0.2;  // Lower bound of the new range
			T b = 0.7;  // Upper bound of the new range
			if (maxv != minv) { // Avoid division by zero
				for (int i = 0; i < height; ++i) {
					for (int j = 0; j < width; ++j) {
						T currentValue = mData->getValueQuick(j, i);
						T scaledValue = a + ((currentValue - minv) * (b - a) / (maxv - minv));
						mData->setValueQuick(j, i, scaledValue);
					}
				}
			}
			else {
				// If all values are the same, set them to a (or any constant value)
				for (int i = 0; i < height; ++i) {
					for (int j = 0; j < width; ++j) {
						mData->setValueQuick(j, i, a);
					}
				}
			}
			// Adjust normalization domain
			mNormalizationDomain[0] = a;
			mNormalizationDomain[1] = b;
			mNormalizationRange[0] = a;
			mNormalizationRange[1] = b;
		}

		if (scheme == ACC) {
			mData->updateMinMax();
			T minv = mData->getMin();
			T maxv = mData->getMax();
			int width = mData->width();
			int height = mData->height();
			// Scale each value linearly to a new range [a, b]
			T a = 0;  // Lower bound of the new range
			T b = 1.3;  // Upper bound of the new range
			if (maxv != minv) { // Avoid division by zero
				for (int i = 0; i < height; ++i) {
					for (int j = 0; j < width; ++j) {
						T currentValue = mData->getValueQuick(j, i);
						T scaledValue = a + ((currentValue - minv) * (b - a) / (maxv - minv));
						mData->setValueQuick(j, i, scaledValue);
					}
				}
			}
			else {
				// If all values are the same, set them to a (or any constant value)
				for (int i = 0; i < height; ++i) {
					for (int j = 0; j < width; ++j) {
						mData->setValueQuick(j, i, a);
					}
				}
			}
			// Adjust normalization domain
			mNormalizationDomain[0] = a;
			mNormalizationDomain[1] = b;
			mNormalizationRange[0] = a;
			mNormalizationRange[1] = b;
		}
	}

	void setColorScheme(const LinearColorMapType& color_map) {
		mColorSheme = color_map;
	}

	void setSelectFillColor(const vec4f& color) {
		mSelectFillColor = color;
	}

	void setHighlightColor(const vec4f& color) {
		mHighlightColor = color;
	}

	void setMargin(const float& margin) {
		mMargin = margin;
	}

	void clearSelection() {
		vec4f zero = makeVec4f(0.0f, 0.0f, 0.0f, 0.0f);
		mRowColor.assign(mRowOrder.size(), zero);
		mColColor.assign(mColOrder.size(), zero);
		if (mCellColor) {
			vec4f* color_data = mCellColor->getData();
			memset(color_data, 0, sizeof(vec4f)*mCellColor->MatrixSize());
		}
		mbHighlightCells = false;
		mMatrixBorderColor = makeVec4f(0.0f);
	}

	void selectMatrix() {
		mMatrixBorderColor = mHighlightColor;
	}

	void setData(MatrixData<T>* data) {
		
		mData = data;

		mRowCoordLookup.clear();
		mColCoordLookup.clear();
		mRowOrder.resize(data->height());
		mColOrder.resize(data->width());
		for (int i = 0; i < mRowOrder.size(); ++i) {
			mRowOrder[i] = i;
			mRowCoordLookup[i] = i;
		}
		for (int i = 0; i < mColOrder.size(); ++i) {
			mColOrder[i] = i;
			mColCoordLookup[i] = i;
		}
		mRowSize.assign(data->height(), 1.0f);
		mColSize.assign(data->width(), 1.0f);

		allocateCellColor();

		clearSelection();
	}

	void setOrigin(const vec2f& org) {
		mArea.origin = org;
	}

	void setAxes(const vec2f& row_axis, const vec2f& col_axis) {
		mArea.row_axis = row_axis;
		mArea.col_axis = col_axis;
	}

	void setRowAxis(const vec2f& row_axis) {
		mArea.row_axis = row_axis;
	}

	void setColumnAxis(const vec2f& col_axis) {
		mArea.col_axis = col_axis;
	}

	void setArea(const RectDisplayArea& area) {
		mArea.origin = area.origin;
		mArea.row_axis = area.row_axis;
		mArea.col_axis = area.col_axis;
	}

	void setTranspose(const bool b_transpose) {
		mbTranspose = b_transpose;
	}

	void setHighlightSize(const bool& b_highlight_size) {
		mbHighlightSize = b_highlight_size;
	}

	void setHighlightSizeThresh(const float& thresh) {
		mHighlightSizeThresh = thresh;
	}

	void orderRows(const std::vector<int>& row_order) {
		mRowOrder.assign(row_order.begin(), row_order.end());
		mRowCoordLookup.clear();
		for (int i = 0; i < row_order.size(); ++i) {
			mRowCoordLookup[row_order[i]] = i;
		}
		mRowColor.resize(mRowOrder.size());
	}

	void orderColumns(const std::vector<int>& col_order) {
		mColOrder.assign(col_order.begin(), col_order.end());
		mColCoordLookup.clear();
		for (int i = 0; i < col_order.size(); ++i) {
			mColCoordLookup[col_order[i]] = i;
		}
		mColColor.resize(mColOrder.size());
	}

	void resetRowOrder() {
		int n = getNumRow();
		mRowOrder.resize(n);
		mRowCoordLookup.clear();
		for (int i = 0; i < n; ++i) {
			mRowOrder[i] = i;
			mRowCoordLookup[i] = i;
		}
	}

	void resetColumnOrder() {
		int n = getNumColumn();
		mColOrder.resize(n);
		mColCoordLookup.clear();
		for (int i = 0; i < n; ++i) {
			mColOrder[i] = i;
			mColCoordLookup[i] = i;
		}
	}

	void setRowSize(const std::vector<float>& row_size) {
		mRowSize.assign(row_size.begin(), row_size.end());
	}

	void setColumnSize(const std::vector<float>& col_size) {
		mColSize.assign(col_size.begin(), col_size.end());
	}

	void allocateCellColor() {
		if (mCellColor != NULL) {
			delete mCellColor;
		}
		if (mData != NULL) {
			if (mbTranspose) {
				mCellColor = new MatrixData<vec4f>(mData->height(), mData->width());
			} else {
				mCellColor = new MatrixData<vec4f>(mData->width(), mData->height());
			}
		}
	}

	vec2f getRowUnit() {
		vec2f row_unit = mArea.row_axis / mColOrder.size();
		return row_unit;
	}

	vec2f getColumnUnit() {
		vec2f col_unit = mArea.col_axis / mRowOrder.size();
		return col_unit;
	}

	vec2f getRowAxis() {
		return mArea.row_axis;
	}

	vec2f getColumnAxis() {
		return mArea.col_axis;
	}

	vec2f getOrigin() {
		return mArea.origin;
	}

	RectDisplayArea getArea() {
		RectDisplayArea ret;
		ret.origin = mArea.origin;
		ret.row_axis = mArea.row_axis;
		ret.col_axis = mArea.col_axis;
		return ret;
	}

	int getRowCoord(const int& idx) {
		std::map<int, int>::iterator it = mRowCoordLookup.find(idx);
		if (it != mRowCoordLookup.end()) {
			return it->second;
		}
		return -1;
	}

	int getColumnCoord(const int& idx) {
		std::map<int, int>::iterator it = mColCoordLookup.find(idx);
		if (it != mColCoordLookup.end()) {
			return it->second;
		}
		return -1;
	}

	vec2i getCellCoord(const vec2f& p) {
		vec2f rp = p - mArea.origin;
		vec2f row_unit = getRowUnit();
		vec2f col_unit = getColumnUnit();
		int r = (int)((rp*col_unit) / (col_unit*col_unit));
		int c = (int)((rp*row_unit) / (row_unit*row_unit));
		return makeVec2i(c, r);
	}

	vec2i getCellCoord(const vec2i& idx) {
		vec2i ret;
		ret.x = getColumnCoord(idx.x);
		ret.y = getRowCoord(idx.y);
		return ret;
	}

	vec2i getCellIdx(const vec2i& coord) {
		if (coord.x < 0 || coord.x >= mColOrder.size()
			|| coord.y < 0 || coord.y >= mRowOrder.size())
		{
			return makeVec2i(-1, -1);
		}
		return makeVec2i(mColOrder[coord.x], mRowOrder[coord.y]);
	}

	vec2i getCellIdx(const vec2f& p) {
		return getCellIdx(getCellCoord(p));
	}

	int getClickElement(const vec2f& p) {
		vec2i cell_idx = getCellIdx(getCellCoord(p));
		return (mSelectType == Column) ? cell_idx.x : cell_idx.y;
	}

	vec2i get_click_element(const vec2f& p) {
		vec2i cell_idx = getCellIdx(getCellCoord(p));
		return makeVec2i(cell_idx.x,cell_idx.y);
	}

	void setSelectElement(const int& eid) {
		if (mSelectType == Row) {
			selectRowWithId(eid, mHighlightColor);
		}
		selectColumnWithId(eid, mHighlightColor);
	}

	void setSelectType(const SelectType& select_type) {
		mSelectType = select_type;
	}

	const T& getELementWithId(const vec2i& idx) {
		if (mbTranspose) {
			return mData->getValueQuick(idx.y, idx.x);
		}
		return mData->getValueQuick(idx.x, idx.y);
	}

	const T& getElementWithCoord(const vec2i& coord) {
		return getELementWithId(getCellIdx(coord));
	}

	const T& getElement(const vec2f& p) {
		return getELementWithId(getCellIdx(p));
	}

	vec2f getPosWithId(const vec2i &idx) {
		vec2i coord;
		coord.x = getColumnCoord(idx.x);
		if (idx.x >= 0) {
			coord.y = getRowCoord(idx.y);
			if (coord.y >= 0) {
				return getPosWithCoord(coord);
			}
		}
		return makeVec2f(-1, -1);
	}
	vec2f getPosWithCoord(const vec2i &coord) {
		vec2f row_unit = getRowUnit();
		vec2f col_unit = getColumnUnit();
		return (mArea.origin + (coord.x + 0.5f)*row_unit + (coord.y + 0.5f)*col_unit);
	}
	void getRowIds(const vec2f& a, const vec2f& b, std::vector<int>& ret) {
		vec2i coord_a = getCellCoord(a);
		vec2i coord_b = getCellCoord(b);
		if (coord_a.x > coord_b.x) std::swap(coord_a.x, coord_b.x);
		if (coord_a.y > coord_b.y) std::swap(coord_a.y, coord_b.y);

		ret.clear();
		if (coord_a.x <= 0 && coord_b.x >= (mColOrder.size() - 1)) {
			int upper = clamp(coord_b.y, 0, mRowOrder.size() - 1);
			int lower = clamp(coord_a.y, 0, mRowOrder.size() - 1);
			for (int i = lower; i <= upper; ++i) {
				ret.push_back(mRowOrder[i]);
			}
		}
	}
	void getColumnIds(const vec2f& a, const vec2f& b, std::vector<int>& ret) {
		vec2i coord_a = getCellCoord(a);
		vec2i coord_b = getCellCoord(b);
		if (coord_a.x > coord_b.x) std::swap(coord_a.x, coord_b.x);
		if (coord_a.y > coord_b.y) std::swap(coord_a.y, coord_b.y);

		ret.clear();
		if (coord_a.y <= 0 && coord_b.y >= (mRowOrder.size() - 1)) {
			int upper = clamp(coord_b.x, 0, mColOrder.size() - 1);
			int lower = clamp(coord_a.x, 0, mColOrder.size() - 1);
			for (int i = lower; i <= upper; ++i) {
				ret.push_back(mColOrder[i]);
			}
		}
	}

	int getDisplayNumRow() {
		return mRowOrder.size();
	}
	int getDisplayNumColumn() {
		return mColOrder.size();
	}
	int getNumRow() {
		if (mbTranspose) {
			return mData->width();
		}
		return mData->height();
	}
	int getNumColumn() {
		if (mbTranspose) {
			return mData->height();
		}
		return mData->width();
	}

	void selectRowWithId(const int& idx, const vec4f& highlight_color) {
		int coord = getRowCoord(idx);
		selectRowWithCoord(coord, highlight_color);
	}
	void selectColumnWithId(const int& idx, const vec4f& highlight_color) {
		int coord = getColumnCoord(idx);
		selectColumnWithCoord(coord, highlight_color);
	}
	void selectRowWithCoord(const int& coord, const vec4f& highlight_color) {
		if (coord >= 0 && coord < mRowColor.size()) {
			mRowColor[coord] = highlight_color;
			mbHighlightCells = false;
		}
	}
	void selectColumnWithCoord(const int& coord, const vec4f& highlight_color) {
		if (coord >= 0 && coord < mColColor.size()) {
			mColColor[coord] = highlight_color;
			mbHighlightCells = false;
		}
	}
	void selectCellWithCoord(const vec2i& coord, const vec4f& highlight_color) {
		vec2i cell_id = getCellIdx(coord);
		mCellColor->setValueQuick(cell_id.x, cell_id.y, highlight_color);
		mbHighlightCells = true;
	}
	void selectCellsWithCoord(const std::vector<vec2i>& coords, const vec4f& highlight_color) {
		for (int i = 0; i < coords.size(); ++i) {
			vec2i cell_id = getCellIdx(coords[i]);
			mCellColor->setValueQuick(cell_id.x, cell_id.y, highlight_color);
		}
		mbHighlightCells = true;
	}
	void selectCellWithId(const vec2i& id, const vec4f& highlight_color) {
		mCellColor->setValueQuick(id.x, id.y, highlight_color);
		mbHighlightCells = true;
	}
	void selectCellsWithIds(const std::vector<vec2i>& ids, const vec4f& highlight_color) {
		for (int i = 0; i < ids.size(); ++i) {
			mCellColor->setValue(ids[i].x, ids[i].y, highlight_color);
		}
		mbHighlightCells = true;
	}
	void drawCell(const T& val, const RectDisplayArea& area);
	void drawSelectedCell(const vec2i& coord, const vec2f& origin,
		const vec2f& row_unit, const vec2f& col_unit)
	{
		vec2i cell_id = getCellIdx(coord);
		vec4f cell_color = mCellColor->getValueQuick(cell_id.x, cell_id.y);
		if (cell_color.a == 0.0f) return;

		if (mSelectFillColor.a > 1e-6) {
			glColor4f(mSelectFillColor.r, mSelectFillColor.g, mSelectFillColor.b, mSelectFillColor.a);
			glBegin(GL_QUADS);
			glVertex(origin);
			glVertex(origin + row_unit);
			glVertex(origin + row_unit + col_unit);
			glVertex(origin + col_unit);
			glEnd();
		}

		glLineWidth(4.0f);
		glColor4f(cell_color.r, cell_color.g, cell_color.b, cell_color.a);
		glBegin(GL_LINE_LOOP);
		glVertex(origin);
		glVertex(origin + row_unit);
		glVertex(origin + row_unit + col_unit);
		glVertex(origin + col_unit);
		glEnd();
	}
	void drawSelectedCell(const vec2i& coord, const vec2f& origin,
		const vec2f& row_unit, const vec2f& col_unit, const vec4f& color)
	{
		if (mSelectFillColor.a > 1e-6) {
			glColor4f(mSelectFillColor.r, mSelectFillColor.g, mSelectFillColor.b, mSelectFillColor.a);
			glBegin(GL_QUADS);
			glVertex(origin);
			glVertex(origin + row_unit);
			glVertex(origin + row_unit + col_unit);
			glVertex(origin + col_unit);
			glEnd();
		}

		glLineWidth(4.0f);
		glColor4f(color.r, color.g, color.b, color.a);
		glBegin(GL_LINE_LOOP);
		glVertex(origin);
		glVertex(origin + row_unit);
		glVertex(origin + row_unit + col_unit);
		glVertex(origin + col_unit);
		glEnd();
	}
	void drawSelectionFrame(vec2i coord1, vec2i coord2, const vec4f& color) {
		if (color.a == 0.0f) return;
		if (coord1.x > coord2.x) std::swap(coord1.x, coord2.x);
		if (coord1.y > coord2.y) std::swap(coord1.y, coord2.y);
		vec2f row_unit = getRowUnit();
		vec2f col_unit = getColumnUnit();
		vec2f p = mArea.origin + coord1.x*row_unit + coord1.y*col_unit;
		vec2f row = (coord2.x - coord1.x + 1)*row_unit;
		vec2f col = (coord2.y - coord1.y + 1)*col_unit;
		glLineWidth(2.0f);
		glColor4f(color.r, color.g, color.b, color.a);
		glBegin(GL_LINE_LOOP);
		glVertex(p);
		glVertex(p + row);
		glVertex(p + row + col);
		glVertex(p + col);
		glEnd();
	}
	void updateElementDisplayArea(const T& val, const RectDisplayArea& area);
	void updateAllELementDisplayArea() {
		vec2f row_unit = getRowUnit();
		vec2f col_unit = getColumnUnit();
		float row_margin = (length(row_unit)*mMargin > 0.1f) ? mMargin : 0.0f;
		float col_margin = (length(col_unit)*mMargin > 0.1f) ? mMargin : 0.0f;

		vec2f frame_org_offset = 0.5f*(row_margin*row_unit + col_margin*col_unit);
		vec2f frame_row_offset = -row_margin*row_unit;
		vec2f frame_col_offset = -col_margin*col_unit;

		RectDisplayArea area;
		vec2f origin_center = mArea.origin + 0.5f*(row_unit + col_unit);
		vec2f lb_offset = (row_margin - 0.5f)*row_unit + (col_margin - 0.5f)*col_unit;
		vec2f cell_row_axis = (1.0f - 2.0f*row_margin)*row_unit;
		vec2f cell_col_axis = (1.0f - 2.0f*col_margin)*col_unit;
		for (int i = 0; i < mRowOrder.size(); ++i) {
			for (int j = 0; j < mColOrder.size(); ++j) {
				int r = mRowOrder[i];
				int c = mColOrder[j];

				T val = mbTranspose ? (mData->getValueQuick(r, c)) : (mData->getValueQuick(c, r));

				float size = (mbHighlightSize) ? (1.0f) : (clamp(min(mRowSize[i], mColSize[j]), 0.0f, 1.0f));
				float row_offset = row_margin + (0.5f - row_margin)*(1.0f - size);
				float col_offset = col_margin + (0.5f - col_margin)*(1.0f - size);
				vec2f origin = mArea.origin + i*col_unit + j*row_unit;
				area.origin = origin + col_offset*col_unit + row_offset*row_unit;
				area.row_axis = size*cell_row_axis;
				area.col_axis = size*cell_col_axis;

				updateElementDisplayArea(val, area);
			}
		}
	}
	void display() {
		vec2f row_unit = getRowUnit();
		vec2f col_unit = getColumnUnit();
		float row_margin = (length(row_unit)*mMargin > 0.1f) ? mMargin : 0.0f;
		float col_margin = (length(col_unit)*mMargin > 0.1f) ? mMargin : 0.0f;
		vec4f border_color = makeVec4f(0.1f, 0.1f, 0.1f, 1.0f);

		vec2f frame_org_offset = 0.5f*(row_margin*row_unit + col_margin*col_unit);
		vec2f frame_row_offset = -row_margin*row_unit;
		vec2f frame_col_offset = -col_margin*col_unit;

		RectDisplayArea area;
		vec2f origin_center = mArea.origin + 0.5f*(row_unit + col_unit);
		vec2f lb_offset = (row_margin-0.5f)*row_unit + (col_margin-0.5f)*col_unit;
		vec2f cell_row_axis = (1.0f - 2.0f*row_margin)*row_unit;
		vec2f cell_col_axis = (1.0f - 2.0f*col_margin)*col_unit;
		for (int i = 0; i < mRowOrder.size(); ++i) {
			for (int j = 0; j < mColOrder.size(); ++j) {
				int r = mRowOrder[i];
				int c = mColOrder[j];
				int ccc = mData->getValueQuick(c, r);
				T val = mbTranspose ? (mData->getValueQuick(r, c)) : (mData->getValueQuick(c, r));

				float size = (mbHighlightSize) ? (1.0f) : (clamp(std::min(mRowSize[i], mColSize[j]), 0.0f, 1.0f));
				float row_offset = row_margin + (0.5f - row_margin)*(1.0f - size);
				float col_offset = col_margin + (0.5f - col_margin)*(1.0f - size);
				vec2f origin = mArea.origin + i*col_unit + j*row_unit;
				area.origin = origin + col_offset*col_unit + row_offset*row_unit;
				area.row_axis = size*cell_row_axis;
				area.col_axis = size*cell_col_axis;

				drawCell(val, area);
				if (mbHighlightCells) {
					drawSelectedCell(makeVec2i(j, i), origin + frame_org_offset, row_unit + frame_row_offset, col_unit + frame_col_offset);
				} else if (mbHighlightSize &&
					(mRowSize[i] > mHighlightSizeThresh && mColSize[j] > mHighlightSizeThresh)) {
					drawSelectedCell(makeVec2i(j, i), origin + frame_org_offset, row_unit + frame_row_offset, col_unit + frame_col_offset, mHighlightColor);
				}
			}
		}
		if (!mbHighlightCells) {
			int row_last = mColColor.size() - 1;
			for (int i = 0; i < mRowColor.size(); ++i) {
				drawSelectionFrame(makeVec2i(0, i), makeVec2i(row_last, i), mRowColor[i]);
			}
			int col_last = mRowColor.size() - 1;
			for (int i = 0; i < mColColor.size(); ++i) {
				drawSelectionFrame(makeVec2i(i, 0), makeVec2i(i, col_last), mColColor[i]);
			}
		}
		if (mMatrixBorderColor.a != 0.0f) {
			int row_last = mColColor.size() - 1;
			int col_last = mRowColor.size() - 1;
			drawSelectionFrame(makeVec2i(0, 0), makeVec2i(row_last, col_last), mMatrixBorderColor);
		}
		
	}

	inline bool inDisplayArea(const vec2f& p) {
		return inRectDisplayArea(p, mArea);
	}


	void mouse_left_click(const vec2f& p) {
		/*clearSelection();
		vec2i click_coord = get_click_element(p);
		addOrRemove(selected_coords, click_coord);
		selectCellsWithCoord(selected_coords, mHighlightColor);*/

		clearSelection();
		vec2i click_coord = get_click_element(p);
		addOrRemove(selected_coords_left, click_coord);
		selectCellsWithCoord(selected_coords_left, ColorMap::getColorByName(ColorMap::Purple_Heart));
		selectCellsWithCoord(selected_coords_right, ColorMap::getColorByName(ColorMap::Yellow_Orange));
		//mCellColor->setValueQuick(click_coord.x, click_coord.y, ColorMap::getColorByName(ColorMap::Purple_Heart));
		mbHighlightCells = true;
	}

	void mouse_right_click(const vec2f& p) {
		clearSelection();
		vec2i click_coord = get_click_element(p);
		addOrRemove(selected_coords_right, click_coord);
		selectCellsWithCoord(selected_coords_left, ColorMap::getColorByName(ColorMap::Purple_Heart));
		selectCellsWithCoord(selected_coords_right, ColorMap::getColorByName(ColorMap::Yellow_Orange));
		mbHighlightCells = true;
		
	}

	void clear_selected() {
		selected_coords.clear();
		selected_coords_left.clear();
		selected_coords_right.clear();
	}

	int numRowElements() { return mData->height(); }
	int numColumnElements() { return mData->width(); }

	void addOrRemove(std::vector<vec2i>& selected_coords, const vec2i& coord) {
		// 查找coord是否已存在于selected_coords中
		auto it = std::find_if(selected_coords.begin(), selected_coords.end(),
			[&coord](const vec2i& item) { return item == coord; });

		if (it != selected_coords.end()) {
			// 如果找到，从vector中移除
			selected_coords.erase(it);
		}
		else {
			// 如果未找到，添加到vector中
			selected_coords.push_back(coord);
		}
	}

	
	float get_value_by_cord(const vec2i& coord) {
		float ret;
		int x = coord.x;
		int y = coord.y;
		float value=mData->getValueQuick(x, y);
		ret = value / (mNormalizationDomain[1]+0.05);
		return ret;
	}

	/*void set_border_color(vec4f color) {
		
		mMatrixBorderColor= color;
	}*/

//private:
	void glVertex(const vec2f& v){glVertex2f(v.x, v.y);}

	std::vector<vec2i> selected_coords;
	std::vector<vec2i> selected_coords_left;
	std::vector<vec2i> selected_coords_right;

	LinearColorMapType mColorSheme;
	SelectType mSelectType;

	MatrixData<T>* mData;
	MatrixData<vec4f>* mCellColor;

	RectDisplayArea mArea;

	std::vector<int> mRowOrder;
	std::vector<int> mColOrder;
	std::vector<vec4f> mRowColor;
	std::vector<vec4f> mColColor;
	std::vector<float> mRowSize;
	std::vector<float> mColSize;

	std::map<int,int> mRowCoordLookup;
	std::map<int,int> mColCoordLookup;

	bool mbTranspose;
	bool mbHighlightCells;
	bool mbHighlightSize;
	float mMargin;
	float mHighlightSizeThresh;

	vec4f mSelectFillColor;
	vec4f mHighlightColor;
	vec4f mMatrixBorderColor;

	T mFilter[2];
	T mNormalizationDomain[2];
	T mNormalizationRange[2];
};
#endif //_MATRIX_DISPLAY_H