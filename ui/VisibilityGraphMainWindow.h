#ifndef EARTHVISMAINWINDOW_H
#define EARTHVISMAINWINDOW_H

#include <QtWidgets/QMainWindow>
#include "ui_VisibilityGraph.h"
#include <vector>

class ViewportWidget;
class RenderWidget;
class EmbeddingWidget;
class QHBoxLayout;
class QueryWidget;
class QWebSocket;

class MainWindow : public QMainWindow
{
	Q_OBJECT
public slots:
	void start();
	void connected();
	void send_message_to_server();


public:
	MainWindow(QWidget *parent = 0, Qt::WindowFlags flags = 0);
	~MainWindow();
	

//private:
	Ui::FlowEncoderClass ui;
	QHBoxLayout *mLayout;

	ViewportWidget *mViewport;
	QueryWidget* mQueryWidget;
	EmbeddingWidget *mEmbeddingWidget;
	RenderWidget *mRenderWidget;
	QWidget* mVisGraphPanel;
	//QWidget* mRenderPanel;

	QWidget* mPanel;
	QWidget* mPanel4embedding;
	QWidget* mLLM_panel;

	QWebSocket* mClient;
	

};

#endif // EARTHVISMAINWINDOW_H
