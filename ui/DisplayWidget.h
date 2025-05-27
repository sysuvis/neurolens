#ifndef DISPLAY_WIDGET_H
#define DISPLAY_WIDGET_H

#include "typeOperation.h"
#ifdef MAC_OS
#include <QtOpenGL/QtOpenGL>
#else
#include <GL/glew.h>
#include <QOpenGLWidget>
#endif
#include <string>
#include <map>
#include "MatrixData.h"
#include "CursorShape.h"

inline void CheckOpenGLError(const char* fname, int line)
{
	GLenum err = glGetError();
	if (err != GL_NO_ERROR)
	{
		printf("OpenGL error %08x, at %s:%i\n", err, fname, line);
		abort();
	}
}

#ifdef _DEBUG
#define GL_DEBUG_CHECK() CheckOpenGLError(__FILE__, __LINE__)
#else
#define GL_DEBUG_CHECK()
#endif

#define GL_CHECK() CheckOpenGLError(__FILE__, __LINE__)

typedef enum {
	MOUSE_NO_BUTTON = 0,
	MOUSE_LEFT_BUTTON,
	MOUSE_MIDDLE_BUTTON,
	MOUSE_RIGHT_BUTTON
} MouseButtonType;

typedef struct {
	vec2f pos;
	MouseButtonType button;
} MouseEvent;

typedef struct {
	std::string name;
	bool enabled;
	vec4f color;
} MenuItem;
typedef std::vector<MenuItem> MenuItemSet;

typedef struct {
	Area area;
	std::string text;
	float font_size;
} DisplayLabel;

inline DisplayLabel makeDisplayLabel(const vec2f& pos, const vec2f& size, std::string text, const float& font_size) {
	DisplayLabel label = { pos, size, text, font_size };
	return label;
}

typedef enum {
	LABEL_LEFT = 0,
	LABEL_RIGHT,
	LABEL_BOTTOM,
	LABEL_TOP,
	LABEL_CENTER
} LabelAlignType;

class ViewportWidget;

class DisplayWidget {
public:
	DisplayWidget(int x, int y, int w, int h, std::string name);

	std::string& getName() { return mName; }

	void setViewportWidget(ViewportWidget* w) { mViewport = w; }

	//menu
	MenuItemSet* getMenuItems() { return &mMenuItems; }
	void addMenuItem(std::string name, const bool& bEnabled = true, const vec4f& color = makeVec4f(0.0f, 0.0f, 0.0f, 0.0f));
	void removeMenuItem(const std::string& name);
	void clearMenuItem();

	//mouse
	CursorShapeType getCursorType();
	void setCursorShape(const CursorShapeType& shape);
	void updateCursorShape();

	void setTransferFunc(float* tf, const int& num);

	void setAutoRedraw(const bool& auto_redraw) { mbAutoRedraw = auto_redraw; }

	void drawBoundary(const char& lrbt_hint = 0xf);

	void setViewport();
	float* getRotationMatrix();
	float* getRotationQuat();
	void setRotationQuatRef(float* quat);
	float getCameraDistance() { return *mCameraDist; }
	float* getCameraDistanceRef() { return mCameraDist; }
	void setCameraDistanceRef(float* cam_dist);
	vec2f* getTranslationRef() { return mTranslate; }
	void setTranslationRef(vec2f* translate);
	vec2f getTranslation() { return *mTranslate; }
	vec2i getPosition() { return makeVec2i(mWinX, mWinY); }
	MouseEvent& getPreviousMouseEvent() { return mPrevMouseEvent; }
	MouseEvent& getFirstMouseEvent() { return mFirstMouseEvent; }

	GLuint createTextureR1D(const std::vector<float>& arr);
	GLuint createTextureRGBA1D(const std::vector<vec4f>& arr);
	GLuint createTextureR2D(MatrixData<float>& mat);

	void updateOpenGLMVPMatrices();
	float* getModelViewMat() { return mModelViewMat; }
	float* getProjectMat() { return mProjectMat; }
	int getViewIndex() { return mViewIndex; }
	void updateViewIndex() { ++mViewIndex; }

	bool saveViewParameters();
	bool loadViewParameters();
	bool saveViewParameters(const char* file_path);
	bool loadViewParameters(const char* file_path);
	void resetViewParameters();

	virtual void init() = 0;
	virtual void display() = 0;
	virtual void menuCallback(const std::string& message) = 0;
	void makeCurrentGLContext();

	bool isCtrlPressed();
	bool isInRotation() { return mbInRotate; }
	bool isInside(const vec2f& p);
	bool isInside(const Area& a);
	void resetInRotationStatus() { mbInRotate = false; }
	virtual void mousePressEvent(const MouseEvent& e);
	virtual void mouseMoveEvent(const MouseEvent& e);
	virtual void mouseReleaseEvent(const MouseEvent& e);
	virtual void wheelEvent(const float& delta);


	//static drawing
	static void drawBox(const vec3f& center, const vec3f& size, const vec4f& color);
	static void drawBoxFrame(const vec3f& center, const vec3f& size, const float& w, const vec4f& color);
	static void drawLine(const vec2f& p1, const vec2f& p2, const float& lineWidth = 2.0f, const vec4f& color = makeVec4f(0.0f, 0.0f, 0.0f, 1.0f));
	static void drawDashedLine(const vec2f& p1, const vec2f& p2, const float& lineWidth, const vec4f& color, const int& factor=1, const unsigned short& pattern= 0x00FF);
	static void drawTriangle(const vec2f& p1, const vec2f& p2, const vec4f& color, const float& lineWidth = 2.0f, const vec4f& line_color = makeVec4f(0.0f, 0.0f, 0.0f, 1.0f));
	static void drawArrow(const vec2f& p1, const vec2f& p2, const float& arrow_size, const float& offset, const float& lineWidth, const vec4f& color = makeVec4f(0.0f, 0.0f, 0.0f, 1.0f));
	static void drawArrow3D(const vec3f& p1, const vec3f& p2, const float& arrow_size, const float& offset, const vec4f& color);
	static void drawCurve(const vec2f& p1, const vec2f& p2, const float& arc_angle = 1.0f, const float& arrow_size = 0.0f, const float& offset = 0.0f, const float& lineWidth = 1.0f, const vec4f& color = makeVec4f(0.0f, 0.0f, 0.0f, 1.0f));
	static void drawCube(const vec3f& center, const float& len, const vec4f& color);
	static void drawSphere(const vec3f& center, const float& radius, const vec4f& color);
	static void drawCircle(const vec2f& center, const float& radius, const vec4f& color, const float& lineWidth = 2.0f, const vec4f& line_color = makeVec4f(0.0f, 0.0f, 0.0f, 1.0f));
	static void drawDashedCircle(const vec2f& center, const float& radius, const vec4f& color, const float& lineWidth = 2.0f, const vec4f& line_color = makeVec4f(0.0f, 0.0f, 0.0f, 1.0f), const int& num_edges = 32);
	static void drawPie(const vec2f& center, const float& radius, const float& angle, const vec4f& color);
	static void drawArc(const vec2f& center, const float& inner_radius, const float& outer_radius, const float& angle, const vec4f& color, const float& lineWidth = 0.0f, const vec4f& line_color = makeVec4f(0.0f, 0.0f, 0.0f, 1.0f));
	static void drawArc(const vec2f& center, const float& inner_radius, const float& outer_radius, const float& start_angle, const float& end_angle, const vec4f& color, const float& lineWidth = 0.0f, const vec4f& line_color = makeVec4f(0.0f, 0.0f, 0.0f, 1.0f));
	static void drawRect(const vec2f& p1, const vec2f p2, const vec4f& color, const float& lineWidth = 2.0f, const vec4f& line_color = makeVec4f(0.0f, 0.0f, 0.0f, 1.0f));
	static void drawRect(const RectDisplayArea& rect, const vec4f& color, const float& lineWidth = 2.0f, const vec4f& line_color = makeVec4f(0.0f, 0.0f, 0.0f, 1.0f));
	static void drawRect(const RectDisplayArea& rect, const GLuint& texture,
		const RectDisplayArea& texture_rect = makeRectDisplayArea(0, 0, 1, 0, 0, 1), const vec4f& color = makeVec4f(1.0f));
	static void drawSquare(const vec2f& center, const float& size, const vec4f& color, const float& lineWidth = 2.0f, const vec4f& line_color = makeVec4f(0.0f, 0.0f, 0.0f, 1.0f));
	static void drawPentagon(const vec2f& center, const float& size, const vec4f& color, const float& lineWidth = 2.0f, const vec4f& line_color = makeVec4f(0.0f, 0.0f, 0.0f, 1.0f));
	static void drawPentagonHalo(const vec2f& center, const float& size, const vec4f& color);
	static void drawRectHalo(const vec2f& p1, const vec2f p2, const vec4f& color);
	static void drawCircleHalo(const vec2f& center, const float& radius, const vec4f& color);
	static void drawSquareHalo(const vec2f& center, const float& size, const vec4f& color);
	static void drawVolume(float* data, int numx, int numy, int numz, float gridx, float gridy, float gridz);
	static void drawRectDashed(const vec2f& p1, const vec2f& p2, const float& lineWidth, const vec4f& color, const int& factor = 1, const unsigned short& pattern = 0x00FF);
	static void drawRectBorder(const vec2f& p1, const vec2f& p2, const float& lineWidth, const vec4f& color);

	void decomposedLookAt(float* ret_rotation, float* ret_translation, const vec3f& eye, const vec3f& target, const vec3f& up);

	//non-static drawing
	void drawLabel(const DisplayLabel& label, const float& font_size, const float& font_weight,
		const vec4f& font_color = makeVec4f(0.0f, 0.0f, 0.0f, 1.0f), const vec4f& label_color = makeVec4f(1.0f, 1.0f, 0.878f, 1.0f));
	void drawText(const vec2f& pos, const std::string& str, const float& font_size, const float& font_weight, const vec4f& color = makeVec4f(0.0f, 0.0f, 0.0f, 1.0f));
	void drawRotateText(const vec2f& pos, const std::string& str, const float& font_size, const float& font_weight, const vec4f& color = makeVec4f(0.0f, 0.0f, 0.0f, 1.0f), const float& angel=0);
	void drawImage(const vec2f& pos, const std::string& path, const float& scale);
	void drawVerticalText(const vec2f& pos, const std::string& str, const float& font_size, const float& font_weight, const vec4f& color);
	void drawAllCachedTexts();
	void drawAllCachedImages();
	void drawPixmap(const vec2f& pos, const vec2f& size, const std::string& name);
	void drawHorizontalCurve(const vec2f& p1, const vec2f p2, const float& lineWidth, const vec4f& color = makeVec4f(0.0f, 0.0f, 0.0f, 1.0f));


	static Area getLabelArea(const std::string& str, const float& font_size, const float& margin_ratio, const vec2f& anchor, const LabelAlignType& align);
	static Area alignArea(const vec2f& area_size, const vec2f& anchor, const LabelAlignType& align, const float& margin_size);
	static vec2f getTextSize(const std::string& str, const float& font_size, const float& margin_ratio);
	static std::string getTransposeText(const std::string& str);

protected:
	//display
	void redraw();
	void setViewport(float l, float r, float b, float t);
	GLuint displayFBOId();

	//use member variable mTextRowMargin as default margin_ratio
	vec2f getTextSize(const std::string& str, const float& font_size);

	//map position
	vec2f normalizeCoord(const vec2f& pos);

	//texture
	GLuint bindTexture(const std::string& file_name);
	void deleteTexture(const GLuint& texture);

	std::string openFile();
	std::string saveFile();

	//cursor
	CursorShapeType	mCursorShape;

	void updateLabelSize(DisplayLabel& label, const float& font_size);
	void updateLabelPos(DisplayLabel& label, const vec2f& pos, const float& font_size = 15.0f);
	void clampLabelPos(DisplayLabel& label, const float& font_size);
	void computeNonOverlapLabelList(std::vector<int>& ret, const std::vector<int>& label_ids, const std::vector<DisplayLabel>& labels, const float& margin = 0.0f);
	vec2f findPointOnCurve(const float& ratio, const vec2f& p1, const vec2f& p2, const float& arc_angle, const float& offset);

	//geometry
	int mWinX, mWinY, mWidth, mHeight;

	//color map
	std::vector<vec4f>	mTransferFunc;

private:
	//name
	std::string		mName;

	//viewport
	ViewportWidget* mViewport;

	//rotation
	bool				mbInRotate;
	float* mRotQuat;
	float				mRotMat[4][4];

	//opengl matrices
	float mModelViewMat[16];
	float mProjectMat[16];
	int mViewIndex;

	//translation
	float* mCameraDist;
	vec2f* mTranslate;

	//mouse
	MouseEvent		mPrevMouseEvent;
	MouseEvent		mFirstMouseEvent;
	bool			mbAutoRedraw;

	//Pop-up menu
	MenuItemSet		mMenuItems;

	//drawing parameters
	float mTextRowMargin;
};

class DisplayBase {
public:
	enum AlignHint {
		FixCenter = 1,
		RoundRow = 2,
		FixCenterRoundRow = 3,
		RoundColumn = 4,
		FixCenterRoundColumn = 5,
		RoundRowColumn = 6,
		FixCenterRoundRowColumn = 7,
	};

	DisplayBase() {}
	~DisplayBase() {}
	virtual void display() = 0;
	//for selection
	virtual int getClickElement(const vec2f& p) = 0;
	virtual void setSelectElement(const int& eid) = 0;
	virtual void clearSelection() = 0;

	//for drawing area
	virtual void setArea(const RectDisplayArea& area) = 0;
	void alignAndRound(const float& pixel_per_unit, const unsigned int& hint) {
		roundElementSize(pixel_per_unit, hint);
		alignOriginToPixel(pixel_per_unit);
	}

	void alignOriginToPixel(const float& pixel_per_unit) {
		mArea.origin.x = alignCoordToPixel(pixel_per_unit, mArea.origin.x);
		mArea.origin.y = alignCoordToPixel(pixel_per_unit, mArea.origin.y);
	}

	float alignCoordToPixel(const float& pixel_per_unit, const float& coord) {
		return floorf(coord * pixel_per_unit) + 0.5f;
	}

	virtual int numRowElements()=0 ;
	virtual int numColumnElements()=0 ;

	void roundElementSize(const float& pixel_per_unit, const unsigned int& hint) {
		if (hint & RoundRow) roundElementSizeRow(pixel_per_unit, hint & FixCenter);
		if (hint & RoundColumn) roundElementSizeColumn(pixel_per_unit, hint & FixCenter);
	}

	void roundElementSizeRow(const float& pixel_per_unit, const bool& fix_center) {
		vec2f row_axis = roundElementSize(pixel_per_unit, mArea.row_axis, numRowElements());
		if (fix_center) mArea.origin += 0.5f * (mArea.row_axis - row_axis);
		mArea.row_axis = row_axis;
	}

	void roundElementSizeColumn(const float& pixel_per_unit, const bool& fix_center) {
		vec2f col_axis = roundElementSize(pixel_per_unit, mArea.col_axis, numColumnElements());
		if (fix_center) mArea.origin += 0.5f * (mArea.col_axis - col_axis);
		mArea.col_axis = col_axis;
	}

	vec2f roundElementSize(const float& pixel_per_unit, const vec2f& axis, const int& n) {
		float axis_len = length(axis);
		vec2f unit_vec = axis / axis_len;
		int pixel_per_element = (int)(axis_len * pixel_per_unit / n);
		axis_len = pixel_per_element * n;
		return unit_vec * axis_len;
	}

	inline bool inDisplayArea(const vec2f& p) {
		return inRectDisplayArea(p, mArea);
	}
	RectDisplayArea mArea;

	//to access parent widget
	void setParent(DisplayWidget* w) { mParent = w; }
	DisplayWidget* mParent;
};

#endif //DISPLAY_WIDGET_H