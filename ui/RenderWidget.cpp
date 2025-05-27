#include "RenderWidget.h"
#include "DataManager.h"
#include "MessageCenter.h"
#include <fstream>
#include "cudaStreamlineRenderer.h"
#include "definition.h"
#include "jitter.h"
#include <time.h>
#include "VolumeData.h"
#include "BrainFiberData.h"
#include "ColorMap.h"
#include "cudaSelection.h"
//#include "WindowsTimer.h"
RenderWidget::RenderWidget(int x, int y, int w, int h, 
	std::string name, BrainDataManager* data)
:DisplayWidget(x, y, w, h, name),
mData(data),
mStreamlineId(0)
{
	setData();
	
	
}

void RenderWidget::setData() {
	vec4f context_color = makeVec4f(ColorMap::getColorByName(ColorMap::Lavender_gray).xyz, 0.1f);
	mData->mRenderer->updateTubeColors(context_color);
	
	

}

void RenderWidget::setStreamline(const int& sid)
{
	mStreamlineId = sid;
	mStreamlineSegments.clear();

	if (sid < 0) {
		vec4f select_color = makeVec4f(ColorMap::getColorByName(ColorMap::Cambridge_Blue).xyz, 0.6f);
		mData->mRenderer->updateTubeColors(select_color);
	}
	else {
		vec4f context_color = makeVec4f(ColorMap::getColorByName(ColorMap::Dim_gray).xyz, 1.0f);
		vec4f select_color = makeVec4f(ColorMap::getColorByName(ColorMap::Red).xyz, 1.0f);
		mData->mRenderer->updateTubeColors(context_color);
		mData->mRenderer->updateStreamlineColor(sid, select_color);

		
	}
}

void RenderWidget::setStreamlines() //unused
{
	int streamline_id = 0;
	for (auto streamline : mData->mColorMask) {
		bool allZeros = all_of(streamline.begin(), streamline.end(), [](int num) {return num == 0; });
		if (!allZeros) {
			int color_id = 0;

			mData->mRenderer->updateStreamlineColor(streamline_id, ColorMap::getD3ColorNoGray(color_id));
		}
		streamline_id++;
	}

}

void RenderWidget::setStreamlines(float thresh) 
{
	vec4f context_color = makeVec4f(ColorMap::getColorByName(ColorMap::Dim_gray).xyz, 1.0f);
	vec4f select_color = makeVec4f(ColorMap::getColorByName(ColorMap::Yellow_Orange).xyz, 1.0f);
	
	static std::vector<float> lengths;
	if (lengths.empty()) {
		lengths.resize(mData->getNumStreamlines());
		for (int i = 0; i < mData->getNumStreamlines() - 1; i++) {
			Streamline stl = mData->mPool.streamlines[i];
			lengths[i] = mData->calculateLength(makeStreamlineSegment(stl.sid, stl.start, stl.start + stl.numPoint - 1));
		}
	}

	mData->mRenderer->updateTubeColors(context_color);
	for (int i = 0; i < mData->getNumStreamlines() - 1; i++) {
		if (lengths[i] < thresh) {
			mData->mRenderer->updateStreamlineColor(i, select_color);	
		}
	}
	redraw();
}

void RenderWidget::init(){
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	float ambient = 0.5f;
	float diffuse = 0.5f;
	float specular = 0.5f;

	float lightDist = 50000.0f;

	GLfloat light0_ambient[4] = {ambient, ambient, ambient, 1.0f};
	GLfloat light0_diffuse[4] = {diffuse, diffuse, diffuse, 1.0f};
	GLfloat light0_specular[4] = {specular, specular, specular, 1.0f};
	GLfloat light0_position[4] = {0.0f, 0.0f, lightDist, 1.0f};
	GLfloat light0_direction[3] = {-1.0f, -1.0f, -1.0f};
	glLightfv(GL_LIGHT0, GL_POSITION, light0_position);
	glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, light0_direction);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light0_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light0_specular);

	GLfloat mat_ambient[4] = {0.5f, 0.5f, 0.5f, 1.0f};
	GLfloat mat_diffuse[4] = {0.5f, 0.5f, 0.5f, 1.0f};
	GLfloat mat_specular[4] = {0.8f, 0.8f, 0.8f, 1.0f};
	glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialf(GL_FRONT, GL_SHININESS, 4.0f);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT);
}

void RenderWidget::display(){
	std::clock_t start = std::clock();
	//GL_CHECK();
	glPushAttrib(GL_ENABLE_BIT|GL_LINE_BIT|GL_CURRENT_BIT|GL_LIGHTING_BIT);

	glShadeModel(GL_SMOOTH);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	
	glClear(GL_ACCUM_BUFFER_BIT);
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
	glAccum(GL_ACCUM, 0.5);
	GLint viewport[4];
	int jitter;
	glGetIntegerv (GL_VIEWPORT, viewport);
	for (jitter = 0; jitter < ACSIZE; jitter++) {
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
		accPerspective (60.0,
			(GLdouble) viewport[2]/(GLdouble) viewport[3],
			1.0, 5000.0, j8[jitter].x, j8[jitter].y, 0.0, 0.0, 1.0);
		singleStepJitterDisplay();
		glAccum(GL_ACCUM, 0.5/ACSIZE);
	}
	glAccum(GL_ADD, -0.5);
	glAccum(GL_RETURN, 2.0);

	//draw boundry
	drawBoundary(8 | 4 | 2 | 1);

	glPopAttrib();
	std::clock_t end = std::clock();
	double duration = (double)(end - start) / CLOCKS_PER_SEC;
	std::cout << duration << endl;
}

void RenderWidget::singleStepJitterDisplay(){
	glPushAttrib(GL_ENABLE_BIT|GL_LINE_BIT|GL_CURRENT_BIT|GL_LIGHTING_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	vec3f vol_size = makeVec3f(mData->mDim);
	float cam_dist = 0.2f*(vol_size.x+ vol_size.y+ vol_size.z+getCameraDistance()/16.0f);
	gluLookAt(0.0f, 0.0f, cam_dist, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	vec2f translate = getTranslation();
	glTranslatef(translate.x*0.3f*cam_dist, translate.y*0.3f*cam_dist, 0.0f);
	float *rotMat = getRotationMatrix();
	glMultMatrixf(rotMat);
	glTranslatef(-0.5f*vol_size.x, -0.5f*vol_size.y, -0.5f*vol_size.z);

	static int prev_view_id = -1;
	//在半透明管道时，变换视角对所有四边形进行了重新排序
	/*if (prev_view_id != getViewIndex()) {
		updateOpenGLMVPMatrices();
		prev_view_id = getViewIndex();
		mData->mRenderer->sortQuadByDepth(getModelViewMat(), getProjectMat());
	}*/

	
	mData->mRenderer->enableRenderProgram();
	
	drawStreamlines();

	if (!mData->mFibers.empty()) { 
		
		//setStreamlines();
		setData();
		updateSegmentsColor(); 
	}
	
	
	

	drawBoxFrame(0.5f*vol_size, vol_size, 1.0f, makeVec4f(0.0f,0.0f,0.0f,1.0f));
	glEnable(GL_DEPTH_TEST);
	glPopAttrib();
}

void RenderWidget::updateSegmentsColor() {
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);

	//glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	//glEnable(GL_DEPTH_TEST);
	glDisable(GL_DEPTH_TEST);
	
	//IndexRange range = { 0,25 };
	
	// color regions
	//for (auto seg : mData->mSegMask) {
	//	int color_id = mData->mColorMask[seg.streamline_id][seg.segment.lower];
	//	//mData->mRenderer->updateSegmentColor(seg.streamline_id, ColorMap::getD3ColorNoGray(color_id), seg.segment);
	//	mData->mRenderer->drawStreamlineSegment(seg.streamline_id, seg.segment, ColorMap::getD3ColorNoGray(color_id));
	//	
	//}
	
	//color bundles
	
	mData->mRenderer->updateSegColor(mData->mFibers);


	glPopAttrib();
}

void RenderWidget::drawStreamlines(){
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);

	//turn on opacity
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_DEPTH_TEST);

	mData->mRenderer->drawAllStreamline();

	glPopAttrib();
}


void RenderWidget::drawStreamline(const int& sid) {
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);

	mData->mRenderer->drawSingleStreamline(sid, ColorMap::getD3Color(0));

	glPopAttrib();
}

void RenderWidget::mouseReleaseEvent(const MouseEvent& e){
	if (e.button==MOUSE_LEFT_BUTTON) {
		//static bool* streamline_marks = new bool[mData->getNumStreamlines()];
		//memset(streamline_marks, 1, sizeof(bool)*mData->getNumStreamlines());
		//vec2f normalized_click = normalizeCoord(e.pos);
		//StreamlineClickInfo info = cudaComputeClickStreamline(normalized_click,
		//	mData->mRenderer->getPoints_d(), mData->mRenderer->getStreamlines_d(), mData->getNumPoints(), mData->getNumStreamlines(),
		//	streamline_marks, getModelViewMat(), getProjectMat());
		//if (info.sid == -1)
		//	return;
		//DataManager::sharedManager()->setIntValue(SELECTED_LINE_ID_NAME, info.sid);
	} else {
		DisplayWidget::mouseReleaseEvent(e);
		if (e.button == MOUSE_RIGHT_BUTTON && !DisplayWidget::isInRotation()) {


			if (!DisplayWidget::loadViewParameters())
			{
				DisplayWidget::saveViewParameters();
			}
		}
	}
}

void RenderWidget::onDataItemChanged(const std::string& name) {
	DataManager* manager = DataManager::sharedManager();
	bool b_success;

	
}

