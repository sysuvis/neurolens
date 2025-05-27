#include "ViewportWidget.h"
#include <cuda_gl_interop.h>
#include <QString>
#include <QMenu>
#include <QAction>
#include <QFileDialog>
#include <QFont>
#include <QIcon>
#include <QPixmap>
#include "CursorShape.h"
#include "BasicFunctions.h"
#include "MessageCenter.h"

#define DRAW_TEXT_TEXTURE

#ifdef DRAW_TEXT_TEXTURE
#include "glText.h"
#endif

ViewportWidget::ViewportWidget(QWidget* parent)
	:QGLWidget(/*QGLFormat(QGL::SampleBuffers),*/parent),
	focusedWidget(NULL),
	menuWidget(NULL),
	mouseOverWidget(NULL),
	bReady(false),
	bCtrlPressed(false),
	mCurrentCursorShape(CURSOR_ARROW)
{
	setMouseTracking(true);
	setFocusPolicy(Qt::StrongFocus);
}

bool ViewportWidget::addChild(DisplayWidget* w, std::string name) {
	if (children.find(name) == children.end()) {
		children[name] = w;
		w->init();
		w->setViewportWidget(this);
	}
	else {
		printf("Err: adding an existing display widget: %s.\n", name.c_str());
		return false;
	}
	return true;
}

bool ViewportWidget::removeChild(const std::string& name) {
	DisplayMap::iterator iter;
	if ((iter = children.find(name)) != children.end()) {
		children.erase(iter);
	}
	else {
		printf("Err: adding an existing display widget: %s.\n", name.c_str());
		return false;
	}
	return true;
}

void ViewportWidget::initializeGL() {
	glViewport(0, 0, width(), height());
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClearAccum(.0f, .0f, .0f, .0f);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_MULTISAMPLE);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	//glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

	//for vbo and cuda
	cudaGLSetGLDevice(0);
	if (glewInit() != GLEW_OK) {
		qDebug("errer glew init");
	}
	if (!glewIsSupported("GL_VERSION_2_0")) {
		qDebug("ERROR: Support for necessary OpenGL extensions missing.");
	}

	DisplayWidget* dw;
	for (DisplayMap::iterator it = children.begin(); it != children.end(); ++it) {
		dw = it->second;
		dw->init();
	}

	emit finishInit();
}

void ViewportWidget::paintGL() {
	bReady = true;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	DisplayWidget* dw;
	for (DisplayMap::iterator it = children.begin(); it != children.end(); ++it) {
		dw = it->second;
		
		dw->setViewport();
		
		dw->display();
		dw->drawAllCachedImages();
		dw->drawAllCachedTexts();
	}
}

void ViewportWidget::updateCursorShape() {
	CursorShapeType cursor_shape;
	if (focusedWidget != NULL) {
		cursor_shape = focusedWidget->getCursorType();
	}
	else if (mouseOverWidget != NULL) {
		cursor_shape = mouseOverWidget->getCursorType();
	}
	else {
		return;
	}

	if (mCurrentCursorShape != cursor_shape) {
		mCurrentCursorShape = cursor_shape;
		this->setCursor(getCursorShape(mCurrentCursorShape));
	}
}

MouseEvent ViewportWidget::convertEvent(QMouseEvent* e) {
	vec2f pos = makeVec2f(e->x(), height() - e->y());
	MouseButtonType btn;
	switch (e->button()) {
	case Qt::NoButton:
		btn = MOUSE_NO_BUTTON;
		break;
	case Qt::RightButton:
		btn = MOUSE_RIGHT_BUTTON;
		break;
	case Qt::MiddleButton:
		btn = MOUSE_MIDDLE_BUTTON;
		break;
	case Qt::LeftButton:
		btn = MOUSE_LEFT_BUTTON;
		break;
	}
	MouseEvent ret = { pos, btn };
	return ret;
}

DisplayWidget* ViewportWidget::getWidget(const vec2f& pos) {
	DisplayWidget* dw;
	for (DisplayMap::iterator it = children.begin(); it != children.end(); ++it) {
		dw = it->second;
		if (dw->isInside(pos)) {
			return dw;
		}
	}
	return NULL;
}

void ViewportWidget::linkRotation(const std::string& name1, const std::string& name2) {
	DisplayWidget* dw;
	if (children.find(name1) != children.end()) {
		dw = children[name1];
	}
	else {
		return;
	}
	float* quat = dw->getRotationQuat();
	if (children.find(name2) != children.end()) {
		dw = children[name2];
	}
	else {
		return;
	}
	dw->setRotationQuatRef(quat);
}

void ViewportWidget::linkCameraDistance(const std::string& name1, const std::string& name2) {
	DisplayWidget* dw;
	if (children.find(name1) != children.end()) {
		dw = children[name1];
	}
	else {
		return;
	}
	float* cam_dist = dw->getCameraDistanceRef();
	if (children.find(name2) != children.end()) {
		dw = children[name2];
	}
	else {
		return;
	}
	dw->setCameraDistanceRef(cam_dist);
}

void ViewportWidget::linkTranslation(const std::string& name1, const std::string& name2) {
	DisplayWidget* dw;
	if (children.find(name1) != children.end()) {
		dw = children[name1];
	}
	else {
		return;
	}
	vec2f* translate = dw->getTranslationRef();
	if (children.find(name2) != children.end()) {
		dw = children[name2];
	}
	else {
		return;
	}
	dw->setTranslationRef(translate);
}

void ViewportWidget::createMenu(QMouseEvent* e) {
	if (focusedWidget == NULL)
		return;

	menuWidget = focusedWidget;

	QMenu menu;
	MenuItemSet* items = focusedWidget->getMenuItems();

	if (items->empty())
		return;

	for (int i = 0; i < items->size(); ++i) {
		if ((*items)[i].name == "\n") {
			menu.addSeparator();
		}
		else {
			QAction* action = menu.addAction(QString((*items)[i].name.c_str()), this, SLOT(menuCallback()));
			action->setEnabled((*items)[i].enabled);
			if ((*items)[i].color.w > 0.000001f) {
				vec3f color = 255.0f * (*items)[i].color.xyz;
				QPixmap pixmap(20, 20);
				pixmap.fill(QColor(color.x, color.y, color.z));
				action->setIcon(QIcon(pixmap));
			}
		}
	}
	menu.exec(mapToGlobal(e->pos()));
}

//TODO: fix implementation
void ViewportWidget::createPixmap(const std::string& path, const std::string& name) {
	//QPixmap pixmap(QString(path.c_str()));
	//GLuint tex_id = bindTexture(pixmap);
	//mPixmaps[name] = tex_id;
}

//TODO: fix implementation
void ViewportWidget::drawPixmap(const vec2f& pos, const vec2f& size, const std::string& name) {
	//glViewport(0,0,width(),height());
	//glPushAttrib(GL_ENABLE_BIT|GL_LINE_BIT|GL_CURRENT_BIT);
	//glDisable(GL_DEPTH_TEST);

	//glEnable(GL_TEXTURE_2D);
	//glEnable( GL_LINE_SMOOTH );
	//glEnable( GL_POLYGON_SMOOTH );
	//glHint( GL_LINE_SMOOTH_HINT, GL_NICEST );
	//glHint( GL_POLYGON_SMOOTH_HINT, GL_NICEST );

	//glMatrixMode(GL_PROJECTION);
	//glLoadIdentity();
	//glOrtho(0, width(), 0, height(), -1000, 1000);
	//glMatrixMode(GL_MODELVIEW);
	//glLoadIdentity();
	//glDisable(GL_LIGHTING);

	//glBindTexture(GL_TEXTURE_2D, mPixmaps[name]);

	//glBegin(GL_QUADS);
	//glTexCoord2d(0.0,0.0); glVertex2f(pos.x, pos.y);
	//glTexCoord2d(0.0,1.0); glVertex2f(pos.x, pos.y+size.y);
	//glTexCoord2d(1.0,1.0); glVertex2f(pos.x+size.x, pos.y+size.y);
	//glTexCoord2d(1.0,0.0); glVertex2f(pos.x+size.x, pos.y);
	//glEnd();

	//glBindTexture(GL_TEXTURE_2D, 0);

	//glPopAttrib();
	//drawTexture()
}


void ViewportWidget::drawText(const vec2f& pos, const std::string& str, const float& font_size, const float& font_weight, const vec4f& color) {
	vec2f mapped_pos = makeVec2f(pos.x, pos.y);
	mTextBuffer.push_back(display_text(mapped_pos, str, font_size, font_weight, color));
}

void ViewportWidget::drawRotateText(const vec2f& pos, const std::string& str, const float& font_size, const float& font_weight, const vec4f& color,const float& angel) {
	vec2f mapped_pos = makeVec2f(pos.x, pos.y);
	mTextRotateBuffer.push_back(display_rotate_text(mapped_pos, str, font_size, font_weight, color,angel));
}

void ViewportWidget::drawImage(const vec2f& pos, const std::string& path, const float& scale) {
	vec2f mapped_pos = makeVec2f(pos.x, pos.y);
	mImageBuffer.push_back(display_image(mapped_pos, path, scale));
}

void ViewportWidget::drawTextBuffer()
{
	if (mTextBuffer.empty()) return;
#ifdef DRAW_TEXT_TEXTURE
	static glText* tex = new glText();
	for (auto t : mTextBuffer) {
		glColor4f(t.color.r, t.color.g, t.color.b, t.color.a);
		tex->drawText(t.pos.x, t.pos.y, t.str, t.font_size);
	}
#else
	QPainter painter(this);
	for (auto t : mTextBuffer) {
		painter.setPen(QColor(t.color.r * 255, t.color.g * 255, t.color.b * 255, t.color.a * 255));
		QFont font("Lucida Console", t.font_size, t.font_weight);
		font.setStyleStrategy(QFont::PreferAntialias);
		painter.setFont(font);
		painter.drawText(t.pos.x, t.pos.y, QString(t.str.c_str()));
	}
	painter.end();
#endif // DRAW_TEXT_TEXTURE
	mTextBuffer.clear();
}

void ViewportWidget::drawRotateTextBuffer()
{
	if (mTextRotateBuffer.empty()) return;
#ifdef DRAW_TEXT_TEXTURE
	static glText* tex = new glText();
	for (auto t : mTextRotateBuffer) {
		glColor4f(t.color.r, t.color.g, t.color.b, t.color.a);
		tex->drawRotatedText(t.pos.x, t.pos.y, t.str, t.font_size,t.angel);
	}
#else
	QPainter painter(this);
	for (auto t : mTextBuffer) {
		painter.setPen(QColor(t.color.r * 255, t.color.g * 255, t.color.b * 255, t.color.a * 255));
		QFont font("Lucida Console", t.font_size, t.font_weight);
		font.setStyleStrategy(QFont::PreferAntialias);
		painter.setFont(font);
		painter.drawText(t.pos.x, t.pos.y, QString(t.str.c_str()));
	}
	painter.end();
#endif // DRAW_TEXT_TEXTURE
	mTextRotateBuffer.clear();
}

void ViewportWidget::drawImageBuffer()
{
	if (mImageBuffer.empty()) return;

	static glText* tex = new glText();
	for (auto i : mImageBuffer) {
		tex->drawFigure(i.pos.x, i.pos.y, i.path, i.scale);
	}

	mImageBuffer.clear();
}


bool ViewportWidget::saveImage() {
	QString filePath("./images/");
	QString fileName =
		QFileDialog::getSaveFileName(this, "Save Image", filePath, "Images (*.bmp)");
	if (fileName.length() == 0) return false;

	update();
	printf("%i %i\n", width(), height());
	bool ret = saveBMP(fileName.toStdString().c_str(), width(), height());

	return ret;
}

void ViewportWidget::menuCallback() {
	QAction* action = dynamic_cast<QAction*>(QObject::sender());
	menuWidget->menuCallback(action->text().toStdString());
}

void ViewportWidget::mousePressEvent(QMouseEvent* e) {
	MouseEvent me = convertEvent(e);

	focusedWidget = getWidget(me.pos);
	mouseOverWidget = focusedWidget;
	if (focusedWidget != NULL) {
		me.pos -= makeVec2f(focusedWidget->getPosition());
		focusedWidget->mousePressEvent(me);
		focusedWidget->getPreviousMouseEvent() = me;
	}
	updateCursorShape();
}

void ViewportWidget::mouseMoveEvent(QMouseEvent* e) {
	MouseEvent me = convertEvent(e);
	mouseOverWidget = getWidget(me.pos);

	if (focusedWidget != NULL) {
		me.pos -= makeVec2f(focusedWidget->getPosition());
		focusedWidget->mouseMoveEvent(me);
	}
	else if (mouseOverWidget != NULL) {
		me.pos -= makeVec2f(mouseOverWidget->getPosition());
		mouseOverWidget->mouseMoveEvent(me);
	}
	updateCursorShape();
}

void ViewportWidget::mouseReleaseEvent(QMouseEvent* e) {
	MouseEvent me = convertEvent(e);
	mouseOverWidget = getWidget(me.pos);

	if (focusedWidget != NULL) {
		me.pos -= makeVec2f(focusedWidget->getPosition());
		focusedWidget->mouseReleaseEvent(me);

		if (me.button == MOUSE_RIGHT_BUTTON && !focusedWidget->isInRotation()) {
			createMenu(e);
		}
		focusedWidget->resetInRotationStatus();
		focusedWidget->getPreviousMouseEvent().button = MOUSE_NO_BUTTON;
	}
	focusedWidget = NULL;
}

void ViewportWidget::wheelEvent(QWheelEvent* e) {
	vec2f pos = makeVec2f(e->pos().x(), height() - e->pos().y());

	focusedWidget = getWidget(pos);
	if (focusedWidget != NULL) {
		focusedWidget->wheelEvent(e->angleDelta().y());
	}
}

void ViewportWidget::keyPressEvent(QKeyEvent* e) {
	if (e->key() == Qt::Key_S) {
		saveImage();
	}
	else if (e->key() == Qt::Key_D) {
		MessageCenter::sharedCenter()->processMessage("debug", "Viewport");
	}
	else if (e->key() == Qt::Key_A) {
		MessageCenter::sharedCenter()->processMessage("Key A", "Viewport");
	}
	else if (e->key() == Qt::Key_Left) {
		MessageCenter::sharedCenter()->processMessage("Key Left", "Viewport");
	}
	else if (e->key() == Qt::Key_Right) {
		MessageCenter::sharedCenter()->processMessage("Key Right", "Viewport");
	}
	else if (e->key() == Qt::Key_Control) {
		bCtrlPressed = true;
	}
}

void ViewportWidget::keyReleaseEvent(QKeyEvent* e) {
	if (e->key() == Qt::Key_Control && bCtrlPressed) {
		bCtrlPressed = false;
	}
}