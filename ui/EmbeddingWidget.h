#pragma once
#include "DisplayWidget.h"
#include "typeOperation.h"
#include "DataUser.h"
#include "VisibilityGraph.h"
#include "MatrixDisplay.h"
#include "GraphDisplay.h"
#include "LineChartDisplay.h"
#include "BarDisplay.h"
#include "BrainFiberData.h"
#include "GraphDisplay.h"
#include "BarDisplay.h"
#include "JointDisplay.h"
#include "Overview.h"
#include "EmbeddingDisplay.h"
#include "EmbeddingDisplay2.h"
#include <string>
class BrainDataManager;

class EmbeddingWidget : public DisplayWidget, public DataUser {
public:
	EmbeddingWidget(int x, int y, int w, int h, std::string name, BrainDataManager* data, GlobalDataManager* gdata);

	void init() {};
	void display();
	QWidget* createPanel(QWidget* parent);

	void menuCallback(const std::string& message) { if (message == "do something") { printf("do something\n"); } }
	void onDataItemChanged(const std::string& name);

	void setData();
	void set_selected_individuals(std::vector<int> left, std::vector<int> right);
	void set_selected_left(std::vector<int> left);
	void set_selected_right(std::vector<int> right);
	//bool readInfo();
	void read_subject_names(std::vector<std::string>& mSubject_names);

	long checkFileSize(string path);
	string makeBundlePath(string subject, int start, int end);
	string chooseBundlePath();
	void EmbeddingWidget::label_display();

	//for interaction
	void mousePressEvent(const MouseEvent& e);
	void mouseMoveEvent(const MouseEvent& e);
	void mouseReleaseEvent(const MouseEvent& e);



	//QWidget* createPanel(QWidget* parent);

	//void setStreamline(const int& streamline_id);
	
	BrainDataManager* mData;
	//embedding view
	//EmbeddingDisplay embedding_view;
	EmbeddingDisplay2 embedding_view;

	//bundle view
	JointDisplay mJointDisplay;
	//overview 
	Overview overview;
	
	//data
	vector<int> selected_left;
	vector<int> selected_right;
	vector<int> mSelectedROIs;

	vector<vector<int>> connection;
	vector<vector<int>> edges;
	vector<vector<int>> edgesSelected;

	//panel parameter
	string mButton_clear;
	string mButton_reset;
	string mButton_query;

	vector<string> mSubject_names;
	string mSelectionSubject;
	int mCurrentSubject=0;
	string mSelectionSubject2;
	int mCurrentSubject2 = 0;

	vector<string> mPoolingNames{"pooling","maxpooling","minpooling"};
	string mSelectionPooling;
	int mCurrentPooling=1;

	string mSelectionConv;
	int mCurrentConv = 3;
	

	vector<string> mFeature{ "Length","Curvature","Torsion","Torsuosity","Entropy" };
	string mSelection_X;
	string mSelection_Y;
	int mCurrent_X=0;
	int mCurrent_Y=1;

	vector<string> ROI_names;
	int current_ROI=0;
	string ROI_selection;
	
	

};