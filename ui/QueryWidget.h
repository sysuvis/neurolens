#pragma once
#include "DisplayWidget.h"
#include "typeOperation.h"
#include "DataUser.h"
#include <string>
#include "DataManager.h"
#include "definition.h"
#include "MessageCenter.h"
#include <iostream>
#include "glText.h"
#include <QWebSocket>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSlider>
#include <QPixmap>
#include <QPlainTextEdit>
#include <QGraphicsDropShadowEffect>
#include <QFrame>

class QueryWidget : public DataUser{
public:
	QueryWidget(std::string name);


	void init() {};
	//void display();

	void set_layout(int x, int y, int w, int h);
	QWidget* createPanel(QWidget* parent);

	void menuCallback(const std::string& message) { if (message == "do something") { printf("do something\n"); } }
	void onDataItemChanged(const std::string& name);

	QString get_query_content();

	void set_figure(QString id);
	void set_caption(QString content);
	void set_generation(QString content);
	void set_papers(QString content);
	void set_query_status(bool flag);


	void send_message_to_server(QString text_data);

	void analysis_message(QString text_data);
	QString readTextFile(const QString& filePath);

	QWidget* query_window;
	QWidget* panel_window;

	QLabel* text_icon;
	QLabel* text_display;
	QLineEdit* textEdit;
	QLabel* imageLabel;
	QLabel* caption_display;

	//control panel
	std::string mButton_submit;
	std::string tokens_set;
	int tokens_current=1028;
	std::string query_status;
	bool query_with_rag = true;

	std::string temperature;
	float temperature_current=0.5;
	std::string topk;
	int topk_current=2;

	//client
	QWebSocket* mClient;
	QString received_info;

	//message information
	QString figure_id;
	QString caption;
	QString generation;
	QString papers;


};

