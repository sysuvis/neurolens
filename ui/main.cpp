//#include "FlowEncoder.h"

#include "VisibilityGraphMainWindow.h"
#include <QtWidgets/QApplication>
#include <QtOpenGL/QGLFormat>
#include <iostream>
#include <QWebSocket>
int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	
	QGLFormat glf = QGLFormat::defaultFormat();
	glf.setSampleBuffers(true);
	glf.setSamples(4);
	QGLFormat::setDefaultFormat(glf);

	MainWindow w;
	w.show();

	return a.exec();
}
