#include "DisplayWidget.h"
#include "trackball.h"
#include "ViewportWidget.h"
#include "DataManager.h"
#include "ColorMap.h"
#include <QString>
#include <QFileDialog>

DisplayWidget::DisplayWidget(int x, int y, int w, int h, std::string name)
	:mName(name),
	mViewport(NULL),
	mWinX(x),
	mWinY(y),
	mWidth(w),
	mHeight(h),
	mTextRowMargin(0.1f),
	mbInRotate(false),
	mRotQuat(NULL),
	mCursorShape(CURSOR_ARROW),
	mViewIndex(0)
{
	mPrevMouseEvent.button = MOUSE_NO_BUTTON;
	mPrevMouseEvent.pos.x = 0.0f;
	mPrevMouseEvent.pos.y = 0.0f;
	mTranslate = new vec2f[1];
	mTranslate[0] = makeVec2f(0.0f, 0.0f);
	mRotQuat = new float[4];
	mCameraDist = new float[1];
	mCameraDist[0] = 0.0f;
	trackball(mRotQuat, 0.0f, 0.0f, 0.0f, 0.0f);
	build_rotmatrix(mRotMat, mRotQuat);

	DataManager::sharedManager()->createPointer(name + ".Pointer", reinterpret_cast<PointerType>(this));
}

void DisplayWidget::addMenuItem(std::string name, const bool& bEnabled, const vec4f& color) {
	MenuItem item = { name, bEnabled, color };
	mMenuItems.push_back(item);
}

void DisplayWidget::removeMenuItem(const std::string& name) {
	std::vector<int> remove_list;
	for (int i = 0; i < mMenuItems.size(); ++i) {
		if (mMenuItems[i].name == name) {
			remove_list.push_back(i);
		}
	}

	for (int i = remove_list.size() - 1; i >= 0; --i) {
		mMenuItems.erase(mMenuItems.begin() + i);
	}
}

void DisplayWidget::clearMenuItem() {
	mMenuItems.clear();
}

float* DisplayWidget::getRotationMatrix() {
	build_rotmatrix(mRotMat, mRotQuat);
	return (&mRotMat[0][0]);
}

float* DisplayWidget::getRotationQuat() {
	return mRotQuat;
}

void DisplayWidget::setRotationQuatRef(float* quat) {
	if (mRotQuat) delete[] mRotQuat;
	mRotQuat = quat;
}

void DisplayWidget::setCameraDistanceRef(float* cam_dist) {
	if (mCameraDist) delete[] mCameraDist;
	mCameraDist = cam_dist;
}

void DisplayWidget::setTranslationRef(vec2f* translate) {
	if (mTranslate) delete[] mTranslate;
	mTranslate = translate;
}


bool DisplayWidget::isCtrlPressed() {
	return mViewport->isCtrlPressed();
}

bool DisplayWidget::isInside(const vec2f& p) {
	return (p.x >= mWinX && p.y >= mWinY && p.x <= (mWinX + mWidth) && p.y <= (mWinY + mHeight));
}

bool DisplayWidget::isInside(const Area& a) {
	return (isInside(a.pos) && isInside(a.pos + a.size));
}

vec2f DisplayWidget::normalizeCoord(const vec2f& pos) {
	vec2f ret = makeVec2f(pos.x / mWidth * 2.0f - 1.0f, pos.y / mHeight * 2.0f - 1.0f);
	return ret;
}

void DisplayWidget::setViewport() {
	glViewport(mWinX, mWinY, mWidth, mHeight);
}

GLuint DisplayWidget::displayFBOId()
{
	return 0;
	//return mViewport->displayFBOId();
}

std::string DisplayWidget::openFile() {
	QString filePath("./");
	QString fileName =
		QFileDialog::getOpenFileName(mViewport, "Open File", filePath, "Any file(*.*)");
	if (fileName.length() == 0) return "";

	return fileName.toStdString();
}

std::string DisplayWidget::saveFile() {
	QString filePath("./");
	QString fileName =
		QFileDialog::getSaveFileName(mViewport, "Save File", filePath, "Any file(*.*)");
	if (fileName.length() == 0) return "";

	return fileName.toStdString();
}

void DisplayWidget::setViewport(float l, float r, float b, float t) {
	glViewport(mWinX + l * mWidth, mWinY + b * mHeight, (r - l) * mWidth, (t - b) * mHeight);
}

void DisplayWidget::mousePressEvent(const MouseEvent& e) {
	mFirstMouseEvent = e;
	mPrevMouseEvent = e;
	mbInRotate = false;
}

void DisplayWidget::mouseMoveEvent(const MouseEvent& e) {
	if (mPrevMouseEvent.button == MOUSE_RIGHT_BUTTON) {

		vec2f prev = normalizeCoord(mPrevMouseEvent.pos);
		vec2f curr = normalizeCoord(e.pos);

		if (abs(prev.x - curr.x) > 0.005f || abs(prev.y - curr.y) > 0.005f) {
			mbInRotate = true;
			float quat[4];
			trackball(quat, prev.x, prev.y, curr.x, curr.y);
			add_quats(quat, mRotQuat, mRotQuat);

			++mViewIndex;
		}
	}
	else if (mPrevMouseEvent.button == MOUSE_MIDDLE_BUTTON) {
		vec2f prev = normalizeCoord(mPrevMouseEvent.pos);
		vec2f curr = normalizeCoord(e.pos);
		*mTranslate = *mTranslate + (curr - prev);

		++mViewIndex;
	}

	mPrevMouseEvent.pos = e.pos;

	redraw();
}

void DisplayWidget::mouseReleaseEvent(const MouseEvent& e) {
	if (mPrevMouseEvent.button == MOUSE_RIGHT_BUTTON && !mbInRotate) {
	}
	mPrevMouseEvent.pos = e.pos;
	mPrevMouseEvent.button = MOUSE_NO_BUTTON;
	mFirstMouseEvent.button = MOUSE_NO_BUTTON;
}

void DisplayWidget::wheelEvent(const float& delta) {
	*mCameraDist += delta;
	++mViewIndex;
	redraw();
}

CursorShapeType DisplayWidget::getCursorType() {
	return mCursorShape;
}

void DisplayWidget::setCursorShape(const CursorShapeType& shape) {
	mCursorShape = shape;
}

void DisplayWidget::updateCursorShape() {
	mViewport->updateCursorShape();
}

void DisplayWidget::setTransferFunc(float* tf, const int& num) {
	mTransferFunc.clear();

	vec4f* vtf = (vec4f*)&tf[0];
	for (int i = 0; i < num; ++i) {
		mTransferFunc.push_back(vtf[i]);
	}
}

void DisplayWidget::drawBoundary(const char& lrbt_hint) {
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glDisable(GL_DEPTH_TEST);

	glEnable(GL_LINE_SMOOTH);
	glEnable(GL_POLYGON_SMOOTH);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, mWidth, 0, mHeight, -1000, 1000);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glDisable(GL_LIGHTING);


	glLineWidth(5.0f);
	glColor4f(0.8f, 0.8f, 0.8f, 1.0f);
	glBegin(GL_LINES);
	if (lrbt_hint & 8) {
		glVertex2f(0, 0);
		glVertex2f(0, mHeight);
	}
	if (lrbt_hint & 4) {
		glVertex2f(mWidth, 0);
		glVertex2f(mWidth, mHeight);
	}
	if (lrbt_hint & 2) {
		glVertex2f(0, 0);
		glVertex2f(mWidth, 0);
	}
	if (lrbt_hint & 1) {
		glVertex2f(0, mHeight);
		glVertex2f(mWidth, mHeight);
	}
	glEnd();

	glPopAttrib();
}

void DisplayWidget::drawLabel(const DisplayLabel& label, const float& font_size, const float& font_weight,
	const vec4f& font_color, const vec4f& label_color)
{
	float margin = 5.0f;
	if (label_color.a > 0.01f) {
		vec2f p1 = label.area.pos + makeVec2f(-margin);
		vec2f p2 = p1 + label.area.size + makeVec2f(2.0f * margin);
		drawRect(p1, p2, label_color, 1.0f);
	}

	drawText(label.area.pos, label.text, font_size, font_weight, font_color);
}

void DisplayWidget::drawText(const vec2f& pos, const std::string& str, const float& font_size, const float& font_weight, const vec4f& color) {
	if (str.empty()) return;
	int cur, prev = -1, cur_len;
	vec2f row_pos = pos;
	vec2f win_pos = makeVec2f(mWinX, mWinY);
	float y_offset = getTextSize("a", font_size).y;
	while ((cur = str.find('\n', prev + 1)) != std::string::npos) {
		cur_len = cur - prev - 1;
		mViewport->drawText(row_pos + win_pos, str.substr(prev + 1, cur_len), font_size, font_weight, color);
		prev = cur;
#ifdef DRAW_TEXT_TEXTURE
		row_pos.y += y_offset;
#else
		row_pos.y -= y_offset;
#endif
	}
	cur_len = str.length() - prev - 1;
	if (cur_len != 0) {
		mViewport->drawText(row_pos + win_pos, str.substr(prev + 1, cur_len), font_size, font_weight, color);
	}
}

void DisplayWidget::drawRotateText(const vec2f& pos, const std::string& str, const float& font_size, const float& font_weight, const vec4f& color,const float& angel) {
	if (str.empty()) return;
	int cur, prev = -1, cur_len;
	vec2f row_pos = pos;
	vec2f win_pos = makeVec2f(mWinX, mWinY);
	float y_offset = getTextSize("a", font_size).y;
	while ((cur = str.find('\n', prev + 1)) != std::string::npos) {
		cur_len = cur - prev - 1;
		mViewport->drawRotateText(row_pos + win_pos, str.substr(prev + 1, cur_len), font_size, font_weight, color, angel);
		prev = cur;
#ifdef DRAW_TEXT_TEXTURE
		row_pos.y += y_offset;
#else
		row_pos.y -= y_offset;
#endif
	}
	cur_len = str.length() - prev - 1;
	if (cur_len != 0) {
		mViewport->drawRotateText(row_pos + win_pos, str.substr(prev + 1, cur_len), font_size, font_weight, color, angel);
	}
}

void DisplayWidget::drawVerticalText(const vec2f& pos, const std::string& str, const float& font_size, const float& font_weight, const vec4f& color) {
	std::string t_str = getTransposeText(str);
	drawText(pos, t_str, font_size, font_weight, color);
}

void DisplayWidget::drawImage(const vec2f& pos, const std::string& path, const float& scale) {
	if (path.empty()) return;
	mViewport->drawImage(pos,path,scale);
}

void DisplayWidget::drawAllCachedTexts() {
	mViewport->drawTextBuffer();
}

void DisplayWidget::drawAllCachedImages() {
	mViewport->drawImageBuffer();
}

void DisplayWidget::drawPixmap(const vec2f& pos, const vec2f& size, const std::string& name) {
	mViewport->drawPixmap(pos + makeVec2f(mWinX, mWinY), size, name);
}

Area DisplayWidget::getLabelArea(const std::string& str, const float& font_size, const float& margin_ratio,
	const vec2f& anchor, const LabelAlignType& align)
{
	return alignArea(getTextSize(str, font_size, margin_ratio), anchor, align, 1.5f * font_size * margin_ratio);
}

Area DisplayWidget::alignArea(const vec2f& area_size, const vec2f& anchor, const LabelAlignType& align,
	const float& margin_size)
{
	Area ret;
	ret.size = area_size;
	if (align == LABEL_LEFT) {
		ret.pos.x = anchor.x + margin_size;
		ret.pos.y = anchor.y - 0.5f * area_size.y;
	}
	else if (align == LABEL_RIGHT) {
		ret.pos.x = anchor.x - area_size.x - margin_size;
		ret.pos.y = anchor.y - 0.5f * area_size.y;
	}
	else if (align == LABEL_BOTTOM) {
		ret.pos.x = anchor.x - 0.5f * area_size.x;
		ret.pos.y = anchor.y + margin_size;
	}
	else if (align == LABEL_TOP) {
		ret.pos.x = anchor.x - 0.5f * area_size.x;
		ret.pos.y = anchor.y - area_size.y - margin_size;
	}
	else if (align == LABEL_CENTER) {
		ret.pos = anchor - 0.5f * area_size;
	}

	return ret;
}

vec2f DisplayWidget::getTextSize(const std::string& str, const float& font_size, const float& margin_ratio) {
	int cur, prev = -1, len = 0, cur_len, num_row = 0;
	while ((cur = str.find('\n', prev + 1)) != std::string::npos) {
		cur_len = cur - prev - 1;
		if (cur_len > len) len = cur_len;
		prev = cur;
		++num_row;
	}
	cur_len = str.length() - prev - 1;
	if (cur_len > len) len = cur_len;
	if (cur_len != 0) ++num_row;
#ifdef DRAW_TEXT_TEXTURE
	return makeVec2f(len * font_size, ((2.0f + 3.0f * margin_ratio) * num_row - 3.0f * margin_ratio) * font_size);
#else
	//return makeVec2f(len*0.83f*font_size, ((1.0f + 2.0f*margin)*num_row - margin)*font_size);
	return makeVec2f(len * 1.245f * font_size, ((1.5f + 3.0f * margin_ratio) * num_row - 3.0f * margin_ratio) * font_size);
#endif
}

vec2f DisplayWidget::getTextSize(const std::string& str, const float& font_size) {
	return getTextSize(str, font_size, mTextRowMargin);
}

void DisplayWidget::clampLabelPos(DisplayLabel& label, const float& font_size) {
	if (label.area.pos.x < 5) {
		label.area.pos.x = 5;
	}
	else if (label.area.pos.x + label.area.size.x > mWidth - 5) {
		label.area.pos.x = mWidth - 5 - label.area.size.x;
	}
	if (label.area.pos.y < 5) {
		label.area.pos.y = 5;
	}
	else if (label.area.pos.y + (1 + mTextRowMargin) * font_size > mHeight - 5) {
		label.area.pos.y = mHeight - 5 - (1 + mTextRowMargin) * font_size;
	}
}

std::string DisplayWidget::getTransposeText(const std::string& str) {
	std::vector<std::string> t_str_arr;
	int r = 0, c = 0;
	for (int i = 0; i < str.size(); ++i) {
		if (str[i] == '\n') {
			r = 0;
			++c;
		}
		else {
			if (r >= t_str_arr.size()) {
				t_str_arr.resize(r + 1);
			}
			char ch = str[i];
			if (ch == '-' || ch == '_') {
				ch = '|';
			}
			else if (ch == '|') {
				ch = '-';
			}
			int gap = c - t_str_arr[r].size();
			if (gap > 0) {
				t_str_arr[r].insert(t_str_arr[r].end(), gap, ' ');
			}
			t_str_arr[r].push_back(ch);
			++r;
		}
	}
	std::string t_str;
	for (int i = 0; i < t_str_arr.size(); ++i) {
		t_str += t_str_arr[i] + '\n';
	}
	return t_str;
}

void DisplayWidget::updateLabelSize(DisplayLabel& label, const float& font_size) {
	label.area.size = getTextSize(label.text, font_size);
}

void DisplayWidget::updateLabelPos(DisplayLabel& label, const vec2f& pos, const float& font_size) {
	label.area.pos = pos + makeVec2f(-0.5f * label.area.size.x, 0.5f * label.area.size.y - (1 + mTextRowMargin) * font_size);
	clampLabelPos(label, font_size);
}

void DisplayWidget::computeNonOverlapLabelList(std::vector<int>& ret,
	const std::vector<int>& label_ids,
	const std::vector<DisplayLabel>& labels,
	const float& margin)
{
	ret.clear();
	for (int i = 0; i < label_ids.size(); ++i) {
		bool bOverlap = false;
		for (int j = 0; j < ret.size(); ++j) {
			if (isOverlapped(labels[label_ids[i]].area, labels[ret[j]].area, margin)) {
				bOverlap = true;
				break;
			}
		}
		if (!bOverlap) {
			ret.push_back(label_ids[i]);
		}
	}
}

#define NUM_QUATER_SPHERE_EDGES 3
void DisplayWidget::drawSphere(const vec3f& center, const float& radius, const vec4f& color) {
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);



	static std::vector<vec3f> ctrl_pts;
	if (ctrl_pts.empty()) {
		std::vector<float> sin_vals;
		std::vector<float> cos_vals;
		float step_size = 0.5f / NUM_QUATER_SPHERE_EDGES;
		for (int i = 0; i <= 4 * NUM_QUATER_SPHERE_EDGES; ++i) {
			float angle = M_PI * (i * step_size);
			sin_vals.push_back(sinf(angle));
			cos_vals.push_back(cosf(angle));
		}

		float lat = 3 * NUM_QUATER_SPHERE_EDGES, lng, z0, zr0, z1, zr1, x, y;
		z1 = sin_vals[lat];
		zr1 = cos_vals[lat];
		for (--lat; lat >= NUM_QUATER_SPHERE_EDGES; --lat) {
			z0 = z1; zr0 = zr1;
			z1 = sin_vals[lat];
			zr1 = cos_vals[lat];

			for (lng = 0; lng < sin_vals.size(); ++lng) {
				x = cos_vals[lng];
				y = sin_vals[lng];

				ctrl_pts.push_back(makeVec3f(x * zr0, y * zr0, z0));
				ctrl_pts.push_back(makeVec3f(x * zr1, y * zr1, z1));
			}
		}
	}

	glColor4f(color.x, color.y, color.z, color.w);
	float lat = 3 * NUM_QUATER_SPHERE_EDGES, lng;
	int i = 0;
	for (lat = 0; lat < 2 * NUM_QUATER_SPHERE_EDGES; ++lat) {
		glBegin(GL_QUAD_STRIP);
		for (lng = 0; lng <= 4 * NUM_QUATER_SPHERE_EDGES; ++lng, i += 2) {
			glNormal(ctrl_pts[i]);
			glVertex(center + radius * ctrl_pts[i]);
			glNormal(ctrl_pts[i + 1]);
			glVertex(center + radius * ctrl_pts[i + 1]);
		}
		glEnd();
	}
	glPopAttrib();
}

void DisplayWidget::drawCircle(const vec2f& center, const float& radius,
	const vec4f& color, const float& lineWidth, const vec4f& line_color)
{
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	static std::vector<vec2f> circle_points;
	if (circle_points.empty()) {
		for (int i = 0; i < 33; ++i) {
			float a = M_PI * (float)i / 16.0f;
			circle_points.push_back(makeVec2f(cosf(a), sinf(a)));
		}
	}

	if (color.w > 0.0f) {
		glColor(color);
		glBegin(GL_TRIANGLE_FAN);
		glVertex(center);
		for (int i = 0; i < circle_points.size(); ++i) {
			glVertex(center + radius * circle_points[i]);
		}
		glEnd();
	}

	if (lineWidth > 0.0f) {
		glLineWidth(lineWidth);
		glColor4f(line_color.x, line_color.y, line_color.z, line_color.w);
		glBegin(GL_LINE_LOOP);
		for (int i = 0; i < circle_points.size(); ++i) {
			glVertex(center + radius * circle_points[i]);
		}
		glEnd();
	}

	glPopAttrib();
}

void DisplayWidget::drawDashedCircle(const vec2f& center, const float& radius,
	const vec4f& color, const float& lineWidth, const vec4f& line_color, const int& num_edges)
{
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	int i;
	float x, y, a;

	if (color.w > 0.0f) {
		glColor4f(color.x, color.y, color.z, color.w);
		glBegin(GL_TRIANGLE_FAN);
		glVertex2f(center.x, center.y);
		for (i = 0; i < 33; ++i) {
			a = M_PI * (float)i / 16.0f;
			x = radius * cosf(a); y = radius * sinf(a);
			glVertex2f(center.x + x, center.y + y);
		}
		glEnd();
	}

	if (lineWidth > 0.0f) {
		glLineWidth(lineWidth);
		glColor4f(line_color.x, line_color.y, line_color.z, line_color.w);
		glBegin(GL_LINES);
		float fac = 2.0f * M_PI / num_edges;
		for (i = 0; i < num_edges; i += 2) {
			a = i * fac;
			x = radius * cosf(a); y = radius * sinf(a);
			glVertex2f(center.x + x, center.y + y);

			a = (i + 1) * fac;
			x = radius * cosf(a); y = radius * sinf(a);
			glVertex2f(center.x + x, center.y + y);
		}
		glEnd();
	}

	glPopAttrib();
}

void DisplayWidget::drawCircleHalo(const vec2f& center, const float& radius, const vec4f& color) {
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	int i;
	float x, y, a;

	glBegin(GL_TRIANGLE_FAN);
	glColor4f(color.x, color.y, color.z, 1.0f);
	glVertex2f(center.x, center.y);
	glColor4f(color.x, color.y, color.z, color.w);
	for (i = 0; i < 33; ++i) {
		a = M_PI * (float)i / 16.0f;
		x = radius * cosf(a); y = radius * sinf(a);
		glVertex2f(center.x + x, center.y + y);
	}
	glEnd();

	glPopAttrib();
}

void DisplayWidget::drawPie(const vec2f& center, const float& radius, const float& angle, const vec4f& color) {
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	int num_arc = angle / M_PI * 16.0f;
	num_arc %= 32;
	float arc_angle = angle / num_arc;

	int i;
	float x, y, a = 0.0f;

	glBegin(GL_TRIANGLE_FAN);
	glColor4f(color.x, color.y, color.z, color.w);
	glVertex2f(center.x, center.y);
	glVertex2f(center.x, center.y + radius);
	for (i = 1; i <= num_arc; ++i) {
		a += arc_angle;
		x = radius * sinf(a); y = radius * cosf(a);
		glVertex2f(center.x + x, center.y + y);
	}
	glEnd();

	glPopAttrib();
}

void DisplayWidget::drawLine(const vec2f& p1, const vec2f& p2, const float& lineWidth, const vec4f& color) {
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glLineWidth(lineWidth);
	glColor4f(color.x, color.y, color.z, color.w);

	glBegin(GL_LINES);
	glVertex2f(p1.x, p1.y);
	glVertex2f(p2.x, p2.y);
	glEnd();

	glPopAttrib();
}

void DisplayWidget::drawDashedLine(const vec2f& p1, const vec2f& p2, const float& lineWidth, const vec4f& color, const int& factor, const unsigned short& pattern) {
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glLineWidth(lineWidth);
	glColor4f(color.x, color.y, color.z, color.w);

	glEnable(GL_LINE_STIPPLE);
	glLineStipple(factor, pattern); // dashed

	glBegin(GL_LINES);
	glVertex2f(p1.x, p1.y);
	glVertex2f(p2.x, p2.y);
	glEnd();

	glDisable(GL_LINE_STIPPLE);
	glPopAttrib();
}

void DisplayWidget::drawRectDashed(const vec2f& p1, const vec2f& p2, const float& lineWidth, const vec4f& color, const int& factor, const unsigned short& pattern) {
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glLineWidth(lineWidth);
	glColor4f(color.x, color.y, color.z, color.w);

	glEnable(GL_LINE_STIPPLE);
	glLineStipple(factor, pattern); // dashed

	glBegin(GL_LINES);
	glVertex2f(p1.x, p1.y);
	glVertex2f(p1.x, p2.y);
	glVertex2f(p1.x, p2.y);
	glVertex2f(p2.x, p2.y);
	glVertex2f(p2.x, p2.y);
	glVertex2f(p2.x, p1.y);
	glVertex2f(p2.x, p1.y);
	glVertex2f(p1.x, p1.y);
	glEnd();

	glDisable(GL_LINE_STIPPLE);
	glPopAttrib();
}

void DisplayWidget::drawRectBorder(const vec2f& p1, const vec2f& p2, const float& lineWidth, const vec4f& color) {
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glLineWidth(lineWidth);
	glColor4f(color.x, color.y, color.z, color.w);

	glBegin(GL_LINES);
	glVertex2f(p1.x, p1.y);
	glVertex2f(p1.x, p2.y);
	glVertex2f(p1.x, p2.y);
	glVertex2f(p2.x, p2.y);
	glVertex2f(p2.x, p2.y);
	glVertex2f(p2.x, p1.y);
	glVertex2f(p2.x, p1.y);
	glVertex2f(p1.x, p1.y);
	glEnd();

	glPopAttrib();
}


void DisplayWidget::drawTriangle(const vec2f& p1, const vec2f& p2, const vec4f& color, const float& lineWidth, const vec4f& line_color) {
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	vec2f v = 0.5f * (p2 - p1);
	std::swap(v.x, v.y);

	vec2f p3 = p2 + v, p4 = p2 - v;
	if (color.a > 0.01f) {
		glColor(color);
		glBegin(GL_TRIANGLES);
		glVertex(p1);
		glVertex(p3);
		glVertex(p4);
		glEnd();
	}

	if (lineWidth > 0.01f && line_color.a > 0.01f) {
		glLineWidth(lineWidth);
		glColor(line_color);
		glBegin(GL_LINE_LOOP);
		glVertex(p1);
		glVertex(p3);
		glVertex(p4);
		glEnd();
	}

	glPopAttrib();
}

void DisplayWidget::drawArrow(const vec2f& p1, const vec2f& p2, const float& arrow_size, const float& offset, const float& lineWidth, const vec4f& color) {
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glLineWidth(lineWidth);
	glColor4f(color.x, color.y, color.z, color.w);

	vec2f v = p1 - p2;
	normalize(v);
	vec2f p4 = p1 - offset * v;
	vec2f p3 = p2 + offset * v;
	v *= arrow_size;
	float cos30 = 0.86602f;

	glBegin(GL_LINES);
	glVertex(p4);
	glVertex(p3);
	glEnd();

	glDisable(GL_LINE_STIPPLE);
	glBegin(GL_LINES);
	glVertex(p3);
	glVertex2f(p3.x + v.x * cos30 - v.y * 0.5f, p3.y + v.x * 0.5f + v.y * cos30);
	glVertex(p3);
	glVertex2f(p3.x + v.x * cos30 + v.y * 0.5f, p3.y - v.x * 0.5f + v.y * cos30);
	glEnd();

	glPopAttrib();
}

void DisplayWidget::drawArrow3D(const vec3f& p1, const vec3f& p2, const float& arrow_size, const float& offset, const vec4f& color) {
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);

	glColor(color);

	static vec2f cp2[8] = { {0.0f, 1.0f}, { 0.7071068f, 0.7071068f}, {1.0f, 0.0f}, { 0.7071068f, -0.7071068f},
	{0.0f, -1.0f}, { -0.7071068f, -0.7071068f}, {-1.0f, 0.0f}, {-0.7071068f,0.7071068f} };

	vec3f n = p1 - p2;
	vec3f p3 = p2 + clamp(offset, 0.0f, 1.0f) * n;
	vec3f p4 = p2 + clamp(1.5f * offset, 0.0f, 1.0f) * n;
	normalize(n);
	vec3f x = n + makeVec3f(1.0f, 0.0f, 0.0f);
	x -= (n * x) * n;
	normalize(x);
	vec3f y = cross(x, n);
	normalize(y);


	vec3f cp3[8];
	for (int i = 0; i < 8; ++i) {
		cp3[i] = (arrow_size * cp2[i].x) * x + (arrow_size * cp2[i].y) * y + p3;
	}

	vec3f cp4[8];
	float half_arrow_size = 0.5f * arrow_size;
	for (int i = 0; i < 8; ++i) {
		cp4[i] = (half_arrow_size * cp2[i].x) * x + (half_arrow_size * cp2[i].y) * y + p4;
	}

	glBegin(GL_TRIANGLE_FAN);
	glNormal(n);
	glVertex(p1);
	for (int i = 0; i < 8; ++i) {
		glNormal(cp4[i] - p4);
		glVertex(cp4[i]);
	}
	glNormal(cp4[0] - p4);
	glVertex(cp4[0]);
	glEnd();

	glBegin(GL_QUAD_STRIP);
	for (int i = 0; i < 8; ++i) {
		glNormal(cp4[i] - p4);
		glVertex(cp4[i]);
		glNormal(cp3[i] - p3);
		glVertex(cp3[i]);
	}
	glNormal(cp4[0] - p4);
	glVertex(cp4[0]);
	glNormal(cp3[0] - p3);
	glVertex(cp3[0]);
	glEnd();

	glBegin(GL_TRIANGLE_FAN);
	glNormal(-n);
	glVertex(p2);
	for (int i = 0; i < 8; ++i) {
		glNormal(cp3[i] - p3);
		glVertex(cp3[i]);
	}
	glNormal(cp3[0] - p3);
	glVertex(cp3[0]);
	glEnd();

	glPopAttrib();
}

vec2f DisplayWidget::findPointOnCurve(const float& ratio, const vec2f& p1, const vec2f& p2, const float& arc_angle, const float& offset) {
	float angle = arc_angle;
	vec2f ret;
	if (abs(angle) < 0.01f) {
		ret = (1.0f - ratio) * p1 + ratio * p2;
	}
	else {
		float ax = p1.x, bx = p2.x, ay = p1.y, by = p2.y;
		bool isCW = (angle < 0.0f);
		float cx, cy, px, py;
		float dist = 0.5f / tan(angle * 0.5f);
		if (isCW) {//clockwise
			cx = 0.5f * (ax + bx) + dist * (by - ay);
			cy = 0.5f * (ay + by) + dist * (ax - bx);
			angle = -angle;
		}
		else {
			cx = 0.5f * (ax + bx) + dist * (ay - by);
			cy = 0.5f * (ay + by) + dist * (bx - ax);
		}
		px = ax - cx;
		py = ay - cy;

		float offset_percentage = offset / (angle * sqrtf(px * px + py * py));
		angle *= (offset_percentage + ratio * (1.0f - 2.0f * offset_percentage));

		float sinVal = sin(angle);
		float cosVal = cos(angle);
		ret = makeVec2f(cx + px * cosVal - py * sinVal, cy + px * sinVal + py * cosVal);
	}
	return ret;
}

void DisplayWidget::drawCurve(const vec2f& p1, const vec2f& p2, const float& arc_angle, const float& arrow_size, const float& offset, const float& lineWidth, const vec4f& color) {
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glLineWidth(lineWidth);
	glColor(color);

	float angle = arc_angle;
	if (abs(angle) < 0.01f) {
		glBegin(GL_LINES);
		glVertex(p1);
		glVertex(p2);
		glEnd();
	}
	else {
		float ax = p1.x, bx = p2.x, ay = p1.y, by = p2.y;
		bool isCW = (angle < 0.0f);
		float cx, cy, px, py;
		float dist = 0.5f / tan(angle * 0.5f);
		if (isCW) {//clockwise
			cx = 0.5f * (ax + bx) + dist * (by - ay);
			cy = 0.5f * (ay + by) + dist * (ax - bx);
			angle = -angle;
		}
		else {
			cx = 0.5f * (ax + bx) + dist * (ay - by);
			cy = 0.5f * (ay + by) + dist * (bx - ax);
		}
		px = ax - cx;
		py = ay - cy;
		float sinVal, cosVal;

		glBegin(GL_LINE_STRIP);
		vec2f prev = p1, cur = p2;
		float offset_percentage = offset / (angle * sqrtf(px * px + py * py));
		for (float i = offset_percentage; i <= 1.0f - offset_percentage; i += 0.03f) {
			sinVal = sin(i * angle);
			cosVal = cos(i * angle);
			prev = cur;
			cur = makeVec2f(cx + px * cosVal - py * sinVal, cy + px * sinVal + py * cosVal);
			glVertex(cur);
		}
		glEnd();
		if (arrow_size > 0.1f) {
			drawArrow(prev, cur, arrow_size, 0.0f, lineWidth, color);
		}
	}

	glPopAttrib();
}

void DisplayWidget::drawHorizontalCurve(const vec2f& p1, const vec2f p2, const float& lineWidth, const vec4f& color) {
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glLineWidth(lineWidth);
	glColor4f(color.x, color.y, color.z, color.w);

	if ((std::abs(p1.y - p2.y)) < mWidth * 0.01f) {
		glBegin(GL_LINES);
		glVertex2f(p1.x, p1.y);
		glVertex2f(p2.x, p2.y);
		glEnd();
	}
	else {
		vec2f p3 = makeVec2f(0.5f * p1.x + 0.5f * p2.x, p1.y);
		vec2f p4 = makeVec2f(0.5f * p1.x + 0.5f * p2.x, p2.y);

		vec2f p;
		glBegin(GL_LINE_STRIP);
		for (float t = 0.0f; t <= 1.00001f; t += 0.05f) {
			p = (1 - t) * (1 - t) * (1 - t) * p1 + 3 * (1 - t) * (1 - t) * t * p3 + 3 * (1 - t) * t * t * p4 + t * t * t * p2;
			glVertex2f(p.x, p.y);
		}
		glEnd();
	}

	glPopAttrib();
}

void DisplayWidget::drawArc(const vec2f& center, const float& inner_radius, const float& outer_radius, const float& angle, const vec4f& color, const float& lineWidth, const vec4f& line_color) {
	drawArc(center, inner_radius, outer_radius, 0, angle, color, lineWidth, line_color);
}

void DisplayWidget::drawArc(const vec2f& center, const float& inner_radius, const float& outer_radius, const float& start_angle, const float& end_angle, const vec4f& color, const float& lineWidth, const vec4f& line_color) {
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	float angle = end_angle - start_angle;
	int num_arc = angle / M_PI * 160.0f;
	num_arc %= 321;
	if (num_arc <= 0) num_arc = 1;
	float arc_angle = angle / num_arc;

	int i;
	float x, y, a = start_angle;

	if (color.w > 1e-10) {
		glBegin(GL_QUAD_STRIP);
		glColor4f(color.x, color.y, color.z, color.w);
		for (i = 0; i <= num_arc; ++i, a += arc_angle) {
			x = sinf(a); y = cosf(a);
			glVertex2f(center.x + inner_radius * x, center.y + inner_radius * y);
			glVertex2f(center.x + outer_radius * x, center.y + outer_radius * y);
		}
		glEnd();
	}

	if (lineWidth > 1e-10) {
		glLineWidth(lineWidth);
		glBegin(GL_LINES);
		glColor(line_color);
		vec2f p1, p2, pp1, pp2;
		for (i = 0; i <= num_arc; ++i, a += arc_angle) {
			x = sinf(a); y = cosf(a);
			pp1 = p1; pp2 = p2;
			p1 = makeVec2f(center.x + inner_radius * x, center.y + inner_radius * y);
			p2 = makeVec2f(center.x + outer_radius * x, center.y + outer_radius * y);
			if (i == 0) {
				glVertex(p1);
				glVertex(p2);
			}
			else {
				glVertex(pp1);
				glVertex(p1);
				glVertex(pp2);
				glVertex(p2);
			}
			if (i == num_arc) {
				glVertex(p1);
				glVertex(p2);
			}
		}
		glEnd();
	}

	glPopAttrib();
}

void DisplayWidget::drawRect(const vec2f& p1, const vec2f p2, const vec4f& color, const float& lineWidth, const vec4f& line_color) {
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	int i;
	float x, y, a;

	vec2f ub, lb;
	if (p1.x > p2.x) {
		ub.x = p1.x; lb.x = p2.x;
	}
	else {
		ub.x = p2.x; lb.x = p1.x;
	}
	if (p1.y > p2.y) {
		ub.y = p1.y; lb.y = p2.y;
	}
	else {
		ub.y = p2.y; lb.y = p1.y;
	}


	if (color.w > 0.0f) {
		glColor4f(color.x, color.y, color.z, color.w);
		glBegin(GL_QUADS);
		glVertex2f(ub.x, ub.y);
		glVertex2f(lb.x, ub.y);
		glVertex2f(lb.x, lb.y);
		glVertex2f(ub.x, lb.y);
		glEnd();
	}

	if (lineWidth > 0.0f) {
		glLineWidth(lineWidth);
		glColor4f(line_color.x, line_color.y, line_color.z, line_color.w);
		glBegin(GL_LINE_LOOP);
		glVertex2f(ub.x, ub.y);
		glVertex2f(lb.x, ub.y);
		glVertex2f(lb.x, lb.y);
		glVertex2f(ub.x, lb.y);
		glEnd();
	}

	glPopAttrib();
}

void DisplayWidget::drawRect(const RectDisplayArea& rect, const vec4f& color, const float& lineWidth,
	const vec4f& line_color)
{
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	if (color.w > 0.0f) {
		glColor(color);
		glBegin(GL_QUADS);
		glVertex(rect.origin);
		glVertex(rect.origin + rect.row_axis);
		glVertex(rect.origin + rect.row_axis + rect.col_axis);
		glVertex(rect.origin + rect.col_axis);
		glEnd();
	}

	if (lineWidth > 0.0f) {
		glLineWidth(lineWidth);
		glColor(line_color);
		glBegin(GL_LINE_LOOP);
		glVertex(rect.origin);
		glVertex(rect.origin + rect.row_axis);
		glVertex(rect.origin + rect.row_axis + rect.col_axis);
		glVertex(rect.origin + rect.col_axis);
		glEnd();
	}

	glPopAttrib();
}

void DisplayWidget::drawRect(const RectDisplayArea& rect, const GLuint& texture,
	const RectDisplayArea& texture_rect, const vec4f& color)
{
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	glEnable(GL_TEXTURE_2D);

	glColor(color);
	glBindTexture(GL_TEXTURE_2D, texture);
	glBegin(GL_QUADS);
	glTexCoord(texture_rect.origin);
	glVertex(rect.origin);
	glTexCoord(texture_rect.origin + texture_rect.row_axis);
	glVertex(rect.origin + rect.row_axis);
	glTexCoord(texture_rect.origin + texture_rect.row_axis + texture_rect.col_axis);
	glVertex(rect.origin + rect.row_axis + rect.col_axis);
	glTexCoord(texture_rect.origin + texture_rect.col_axis);
	glVertex(rect.origin + rect.col_axis);
	glEnd();
	glBindTexture(GL_TEXTURE_1D, 0);

	glPopAttrib();
}

void DisplayWidget::drawRectHalo(const vec2f& p1, const vec2f p2, const vec4f& color) {
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	int i;
	float x, y, a;

	vec2f ub, lb, c;
	if (p1.x > p2.x) {
		ub.x = p1.x; lb.x = p2.x;
	}
	else {
		ub.x = p2.x; lb.x = p1.x;
	}
	if (p1.y > p2.y) {
		ub.y = p1.y; lb.y = p2.y;
	}
	else {
		ub.y = p2.y; lb.y = p1.y;
	}
	c = 0.5f * (ub + lb);

	glBegin(GL_TRIANGLE_FAN);
	glColor4f(color.x, color.y, color.z, 1.0f);
	glVertex2f(c.x, c.y);
	glColor4f(color.x, color.y, color.z, color.w);
	glVertex2f(ub.x, ub.y);
	glVertex2f(lb.x, ub.y);
	glVertex2f(lb.x, lb.y);
	glVertex2f(ub.x, lb.y);
	glVertex2f(ub.x, ub.y);
	glEnd();

	glPopAttrib();
}

void DisplayWidget::drawPentagon(const vec2f& center, const float& size, const vec4f& color, const float& lineWidth, const vec4f& line_color) {
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	vec2f p;
	if (color.w > 0.0f) {
		glColor4f(color.x, color.y, color.z, color.w);
		glBegin(GL_TRIANGLE_FAN);
		glVertex2f(center.x, center.y);
		glVertex2f(center.x, center.y - size);
		for (int i = 1; i < 5; ++i) {
			p.x = center.x + sinf(i * 0.4f * M_PI) * size;
			p.y = center.y - cosf(i * 0.4f * M_PI) * size;
			glVertex2f(p.x, p.y);
		}
		glVertex2f(center.x, center.y - size);
		glEnd();
	}

	if (lineWidth > 0.0f) {
		glLineWidth(lineWidth);
		glColor4f(line_color.x, line_color.y, line_color.z, line_color.w);
		glBegin(GL_LINE_LOOP);
		glVertex2f(center.x, center.y - size);
		for (int i = 1; i < 5; ++i) {
			p.x = center.x + sinf(i * 0.4f * M_PI) * size;
			p.y = center.y - cosf(i * 0.4f * M_PI) * size;
			glVertex2f(p.x, p.y);
		}
		glEnd();
	}

	glPopAttrib();
}

void DisplayWidget::drawPentagonHalo(const vec2f& center, const float& size, const vec4f& color) {
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	vec2f p;

	glBegin(GL_TRIANGLE_FAN);
	glColor4f(color.x, color.y, color.z, 1.0f);
	glVertex2f(center.x, center.y);
	glColor4f(color.x, color.y, color.z, color.w);
	glVertex2f(center.x, center.y - size);
	for (int i = 1; i < 5; ++i) {
		p.x = center.x + sinf(i * 0.4f * M_PI) * size;
		p.y = center.y - cosf(i * 0.4f * M_PI) * size;
		glVertex2f(p.x, p.y);
	}
	glVertex2f(center.x, center.y - size);
	glEnd();

	glPopAttrib();
}

void DisplayWidget::drawSquare(const vec2f& center, const float& size, const vec4f& color, const float& lineWidth, const vec4f& line_color) {
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	int i;
	float x, y, a;

	if (color.w > 0.0f) {
		glColor4f(color.x, color.y, color.z, color.w);
		glBegin(GL_QUADS);
		glVertex2f(center.x + 0.5f * size, center.y + 0.5f * size);
		glVertex2f(center.x - 0.5f * size, center.y + 0.5f * size);
		glVertex2f(center.x - 0.5f * size, center.y - 0.5f * size);
		glVertex2f(center.x + 0.5f * size, center.y - 0.5f * size);
		glEnd();
	}

	if (lineWidth > 0.0f) {
		glLineWidth(lineWidth);
		glColor4f(line_color.x, line_color.y, line_color.z, line_color.w);
		glBegin(GL_LINE_LOOP);
		glVertex2f(center.x + 0.5f * size, center.y + 0.5f * size);
		glVertex2f(center.x - 0.5f * size, center.y + 0.5f * size);
		glVertex2f(center.x - 0.5f * size, center.y - 0.5f * size);
		glVertex2f(center.x + 0.5f * size, center.y - 0.5f * size);
		glEnd();
	}

	glPopAttrib();
}

void DisplayWidget::drawSquareHalo(const vec2f& center, const float& size, const vec4f& color) {
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glBegin(GL_TRIANGLE_FAN);
	glColor4f(color.x, color.y, color.z, 1.0f);
	glVertex2f(center.x, center.y);
	glColor4f(color.x, color.y, color.z, color.w);
	glVertex2f(center.x + 0.5f * size, center.y + 0.5f * size);
	glVertex2f(center.x - 0.5f * size, center.y + 0.5f * size);
	glVertex2f(center.x - 0.5f * size, center.y - 0.5f * size);
	glVertex2f(center.x + 0.5f * size, center.y - 0.5f * size);
	glVertex2f(center.x + 0.5f * size, center.y + 0.5f * size);
	glEnd();

	glPopAttrib();
}

void DisplayWidget::drawBox(const vec3f& center, const vec3f& size, const vec4f& color) {
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glColor4f(color.x, color.y, color.z, color.w);
	vec3f len = 0.5f * size;
	glBegin(GL_QUAD_STRIP);
	glVertex3f(center.x - len.x, center.y - len.y, center.z - len.z);
	glVertex3f(center.x - len.x, center.y + len.y, center.z - len.z);
	glVertex3f(center.x + len.x, center.y - len.y, center.z - len.z);
	glVertex3f(center.x + len.x, center.y + len.y, center.z - len.z);
	glVertex3f(center.x + len.x, center.y - len.y, center.z + len.z);
	glVertex3f(center.x + len.x, center.y + len.y, center.z + len.z);
	glVertex3f(center.x - len.x, center.y - len.y, center.z + len.z);
	glVertex3f(center.x - len.x, center.y + len.y, center.z + len.z);
	glEnd();
	glBegin(GL_QUADS);
	glVertex3f(center.x - len.x, center.y - len.y, center.z - len.z);
	glVertex3f(center.x + len.x, center.y - len.y, center.z - len.z);
	glVertex3f(center.x + len.x, center.y - len.y, center.z + len.z);
	glVertex3f(center.x - len.x, center.y - len.y, center.z + len.z);
	glVertex3f(center.x - len.x, center.y + len.y, center.z - len.z);
	glVertex3f(center.x + len.x, center.y + len.y, center.z - len.z);
	glVertex3f(center.x + len.x, center.y + len.y, center.z + len.z);
	glVertex3f(center.x - len.x, center.y + len.y, center.z + len.z);
	glEnd();
}

void DisplayWidget::drawBoxFrame(const vec3f& center, const vec3f& size, const float& w, const vec4f& color) {
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	glColor4f(color.x, color.y, color.z, color.w);
	glLineWidth(w);
	vec3f len = 0.5f * size;
	glBegin(GL_LINE_STRIP);
	glVertex3f(center.x - len.x, center.y - len.y, center.z - len.z);
	glVertex3f(center.x + len.x, center.y - len.y, center.z - len.z);
	glVertex3f(center.x + len.x, center.y + len.y, center.z - len.z);
	glVertex3f(center.x - len.x, center.y + len.y, center.z - len.z);
	glVertex3f(center.x - len.x, center.y - len.y, center.z - len.z);
	glVertex3f(center.x - len.x, center.y - len.y, center.z + len.z);
	glVertex3f(center.x + len.x, center.y - len.y, center.z + len.z);
	glVertex3f(center.x + len.x, center.y - len.y, center.z - len.z);
	glVertex3f(center.x - len.x, center.y - len.y, center.z - len.z);
	glEnd();
	glBegin(GL_LINE_STRIP);
	glVertex3f(center.x + len.x, center.y + len.y, center.z + len.z);
	glVertex3f(center.x - len.x, center.y + len.y, center.z + len.z);
	glVertex3f(center.x - len.x, center.y - len.y, center.z + len.z);
	glVertex3f(center.x + len.x, center.y - len.y, center.z + len.z);
	glVertex3f(center.x + len.x, center.y + len.y, center.z + len.z);
	glVertex3f(center.x + len.x, center.y + len.y, center.z - len.z);
	glVertex3f(center.x - len.x, center.y + len.y, center.z - len.z);
	glVertex3f(center.x - len.x, center.y + len.y, center.z + len.z);
	glVertex3f(center.x + len.x, center.y + len.y, center.z + len.z);
	glEnd();
	glPopAttrib();
}

void DisplayWidget::drawCube(const vec3f& center, const float& len, const vec4f& color) {
	vec3f size = makeVec3f(len, len, len);
	drawBox(center, size, color);
}

void DisplayWidget::drawVolume(float* data, int numx, int numy, int numz, float gridx, float gridy, float gridz) {
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	vec4f color;
	float val;
	float xup, xlow, yup, ylow, zup, zlow;

	int numxy = numx * numy;

	zlow = 0.0f; zup = gridz;
	for (int i = 0; i < numz; i++, zlow = zup, zup += gridz) {
		ylow = 0.0f; yup = gridy;
		for (int j = 0; j < numy; j++, ylow = yup, yup += gridy) {
			xlow = 0.0f; xup = gridx;
			for (int k = 0; k < numx; k++, xlow = xup, xup += gridx) {
				val = data[i * numxy + j * numx + k];
				color = ColorMap::getPerceptualColor(val);
				glColor4f(color.x, color.y, color.z, val);
				glBegin(GL_LINE_STRIP);
				glVertex3f(xlow, ylow, zlow);
				glVertex3f(xlow, yup, zlow);
				glVertex3f(xlow, yup, zup);
				glVertex3f(xlow, ylow, zup);
				glVertex3f(xlow, ylow, zlow);
				glVertex3f(xup, ylow, zlow);
				glVertex3f(xup, ylow, zup);
				glVertex3f(xlow, ylow, zup);
				glEnd();
				glBegin(GL_LINE_STRIP);
				glVertex3f(xup, yup, zlow);
				glVertex3f(xlow, yup, zlow);
				glVertex3f(xlow, yup, zup);
				glVertex3f(xup, yup, zup);
				glVertex3f(xup, yup, zlow);
				glVertex3f(xup, ylow, zlow);
				glVertex3f(xup, ylow, zup);
				glVertex3f(xup, yup, zup);
				glEnd();
			}
		}
	}

	glPopAttrib();
}

void DisplayWidget::decomposedLookAt(float* ret_rotation, float* ret_translation, const vec3f& eye, const vec3f& target, const vec3f& up) {
	vec3f forward = eye - target;
	normalize(forward);
	vec3f left = cross(up, forward);
	normalize(left);
	vec3f real_up = cross(forward, left);

	memset(ret_rotation, 0, sizeof(float) * 16);
	ret_rotation[0] = left.x;
	ret_rotation[4] = left.y;
	ret_rotation[8] = left.z;
	ret_rotation[1] = real_up.x;
	ret_rotation[5] = real_up.y;
	ret_rotation[9] = real_up.z;
	ret_rotation[2] = forward.x;
	ret_rotation[6] = forward.y;
	ret_rotation[10] = forward.z;
	ret_rotation[15] = 1.0f;

	memset(ret_translation, 0, sizeof(float) * 16);
	ret_translation[0] = 1.0f;
	ret_translation[5] = 1.0f;
	ret_translation[10] = 1.0f;
	ret_translation[12] = -eye.x;
	ret_translation[13] = -eye.y;
	ret_translation[14] = -eye.z;
	ret_translation[15] = 1.0f;
}

GLuint DisplayWidget::bindTexture(const std::string& file_name) {
	//return mViewport->bindTexture(QString(file_name.c_str()));
	return 0;
}

//TODO: fix
void DisplayWidget::deleteTexture(const GLuint& texture) {
	//mViewport->deleteTexture(texture);
}

void DisplayWidget::redraw() {
	if (mViewport && mViewport->isReady())
		mViewport->update();
}

void DisplayWidget::makeCurrentGLContext() {
	if (mViewport && mViewport->isReady()) {
		mViewport->makeCurrent();
	}
}

GLuint DisplayWidget::createTextureR1D(const std::vector<float>& arr) {
	GLuint tex;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_1D, tex);
	glTexImage1D(GL_TEXTURE_1D, 0, GL_RED, arr.size(), 0, GL_RED, GL_FLOAT, arr.data());
	glBindTexture(GL_TEXTURE_1D, 0);
	return tex;
}

GLuint DisplayWidget::createTextureRGBA1D(const std::vector<vec4f>& arr) {
	GLuint tex;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_1D, tex);
	glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, arr.size(), 0, GL_RGBA, GL_FLOAT, arr.data());
	glBindTexture(GL_TEXTURE_1D, 0);
	return tex;
}

GLuint DisplayWidget::createTextureR2D(MatrixData<float>& mat) {
	GLuint tex;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, mat.width(), mat.height(), 0, GL_RED, GL_FLOAT, mat.getData());
	glBindTexture(GL_TEXTURE_2D, 0);
	return tex;
}

void DisplayWidget::updateOpenGLMVPMatrices() {
	glGetFloatv(GL_MODELVIEW_MATRIX, mModelViewMat);
	glGetFloatv(GL_PROJECTION_MATRIX, mProjectMat);
}

bool DisplayWidget::saveViewParameters() {
	std::string file_path = saveFile();
	if (file_path.empty()) {
		return false;
	}
	return saveViewParameters(file_path.c_str());
}

bool DisplayWidget::saveViewParameters(const char* file_path) {
	std::ofstream file;
	if (!open_file(file, file_path, true)) {
		return false;
	}
	file.write((char*)mCameraDist, sizeof(float));
	file.write((char*)mTranslate, sizeof(vec2f));
	file.write((char*)mRotQuat, sizeof(float) * 4);
	file.close();
	return true;
}

bool DisplayWidget::loadViewParameters() {
	std::string file_path = openFile();
	if (file_path.empty()) {
		return false;
	}
	return loadViewParameters(file_path.c_str());
}

bool DisplayWidget::loadViewParameters(const char* file_path) {
	std::ifstream file;
	if (!open_file(file, file_path, true)) {
		return false;
	}
	file.read((char*)mCameraDist, sizeof(float));
	file.read((char*)mTranslate, sizeof(vec2f));
	file.read((char*)mRotQuat, sizeof(float) * 4);
	file.close();
	updateViewIndex();
	redraw();
	return true;
}

void DisplayWidget::resetViewParameters() {
	*mCameraDist = 0.0f;
	*mTranslate = makeVec2f(0.0f, 0.0f);
	trackball(mRotQuat, 0.0f, 0.0f, 0.0f, 0.0f);
	build_rotmatrix(mRotMat, mRotQuat);
	updateViewIndex();
}
