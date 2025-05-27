#include "DataManager.h"
#include "MessageCenter.h"
#include <fstream>
#include "cudaStreamlineRenderer.h"
#include "definition.h"
#include "jitter.h"
#include "WindowsTimer.h"
#include <time.h>
#include "VolumeData.h"
#include "VisibilityGraph.h"
#include "ColorMap.h"
#include "cudaSelection.h"
#include "FlowEncoder.h"
#include "GaussianSmoothing.h"


#define TIMING_RESULT
#ifdef TIMING_RESULT
#define PRINT_TIME printf
#define START_TIMER mTimer.start()
#endif
using std::vector;
using std::cout;
using std::endl;

FlowEncoder::FlowEncoder(int x, int y, int w, int h,
	std::string name,bool Reflag)
	:DisplayWidget(x, y, w, h, name)
{
	test = Reflag;
}

void FlowEncoder::init() {
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	//环境+漫反射+高光系数
	//bling-phone
	float ambient = 0.5f;
	float diffuse = 0.5f;
	float specular = 0.5f;

	//光照强度
	float lightDist = 50000.0f;

	GLfloat light0_ambient[4] = { ambient, ambient, ambient, 1.0f };
	GLfloat light0_diffuse[4] = { diffuse, diffuse, diffuse, 1.0f };
	GLfloat light0_specular[4] = { specular, specular, specular, 1.0f };
	GLfloat light0_position[4] = { 0.0f, 0.0f, lightDist, 1.0f };
	GLfloat light0_direction[3] = { -1.0f, -1.0f, -1.0f };
	glLightfv(GL_LIGHT0, GL_POSITION, light0_position);
	glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, light0_direction);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light0_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light0_specular);

	GLfloat mat_ambient[4] = { 0.5f, 0.5f, 0.5f, 1.0f };
	GLfloat mat_diffuse[4] = { 0.5f, 0.5f, 0.5f, 1.0f };
	GLfloat mat_specular[4] = { 0.8f, 0.8f, 0.8f, 1.0f };
	glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialf(GL_FRONT, GL_SHININESS, 4.0f);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT);


	dim = read_header(vec_hdr_path.c_str());
	readStreamlinePool(VelocityFromStreamlinePool, streamline_path.c_str());
	mData= new FlowEncoderDataManager(VelocityFromStreamlinePool);
	StreamlinePool tmpPool;
	mData->genStreamlineDataFromBrain();

	//vec3i size = makeVec3i(128, 128, 128);
	//mData->transform(size);
	
	mData->resample(RenderPool);
	mData->getReconstructPool(tmpPool, RenderPool);
	if (!test) {
		RenderPool = VelocityFromStreamlinePool;
	}

	if (test) {
		//RenderPool = rPool;
		RenderPool = tmpPool;
		mData->smooth(RenderPool,2);
	}

	mRender = new cudaStreamlineRenderer(RenderPool.streamlines.data(), RenderPool.points.data(), RenderPool.streamlines.size(), 8, 2);
	mRender->enableRenderProgram();
}

void FlowEncoder::mouseReleaseEvent(const MouseEvent& e)
{

	DisplayWidget::mouseReleaseEvent(e);
	if (e.button == MOUSE_RIGHT_BUTTON && !DisplayWidget::isInRotation()) {


		if (!DisplayWidget::loadViewParameters())
		{
			DisplayWidget::saveViewParameters();
		}
	}

}


void FlowEncoder::display() {
	//GL_CHECK();
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT | GL_LIGHTING_BIT);

	glShadeModel(GL_SMOOTH);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glClear(GL_ACCUM_BUFFER_BIT);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glAccum(GL_ACCUM, 0.5);
	GLint viewport[4];
	int jitter;
	glGetIntegerv(GL_VIEWPORT, viewport);
	for (jitter = 0; jitter < ACSIZE; jitter++) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		accPerspective(60.0,
			(GLdouble)viewport[2] / (GLdouble)viewport[3],
			1.0, 5000.0, j8[jitter].x, j8[jitter].y, 0.0, 0.0, 1.0);
		singleStepJitterDisplay();
		glAccum(GL_ACCUM, 0.5 / ACSIZE);
	}
	glAccum(GL_ADD, -0.5);
	glAccum(GL_RETURN, 2.0);

	glPopAttrib();
}

void FlowEncoder::singleStepJitterDisplay() {
	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT | GL_LIGHTING_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	vec3f vol_size = makeVec3f(dim);
	float cam_dist = vol_size.x + vol_size.y + vol_size.z + getCameraDistance() / 16.0f;
	gluLookAt(0.0f, 0.0f, cam_dist, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	vec2f translate = getTranslation();
	glTranslatef(translate.x * 0.3f * cam_dist, translate.y * 0.3f * cam_dist, 0.0f);
	float* rotMat = getRotationMatrix();
	glMultMatrixf(rotMat);
	glTranslatef(-0.5f * vol_size.x, -0.5f * vol_size.y, -0.5f * vol_size.z);

	static int prev_view_id = -1;
	if (prev_view_id != getViewIndex()) {
		updateOpenGLMVPMatrices();
		prev_view_id = getViewIndex();
	}

	//开始drawStreamline
	//给个颜色
	vec4f select_color1 = makeVec4f(ColorMap::getColorByName(ColorMap::Lavender_gray).xyz, 0.015f);
	vec4f select_color2 = makeVec4f(ColorMap::getColorByName(ColorMap::Brandeis_blue).xyz, 1.0f);
	vec4f select_color3 = makeVec4f(ColorMap::getColorByName(ColorMap::Cambridge_Blue).xyz, 0.6f);
	auto  select_color4 = makeVec4f(ColorMap::getColorByName(ColorMap::Black).xyz, 1.0f);
	vec4f context_color = makeVec4f(ColorMap::getColorByName(ColorMap::Brandeis_blue).xyz, 0.015f);
	//流线的粗细
	mRender->updateTubeRadius(0.2);
	mRender->updateTubeColors(context_color);
	mRender->drawAllStreamline();
	//结束drawStreamline

	glEnable(GL_DEPTH_TEST);

	//drawBoxFrame(0 * vol_size, vol_size, 1.0f, makeVec4f(0.0f, 0.0f, 0.0f, 1.0f));
	drawBoxFrame(0.5 * vol_size, vol_size, 1.0f, makeVec4f(0.0f, 0.0f, 0.0f, 1.0f));

	glPopAttrib();

}
