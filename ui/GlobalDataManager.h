#ifndef GLOBALDATAMANAGER_H
#define GLOBALDATAMANAGER_H

#include "DisplayWidget.h"
#include "typeOperation.h"
#include "DataUser.h"
//#include "VisibilityGraph.h"
#include "MatrixDisplay.h"
#include "GraphDisplay.h"
#include "LineChartDisplay.h"
#include "BarDisplay.h"
#include "BrainFiberData.h"
#include "MessageCenter.h"
#include "definition.h"
//class BrainDataManager;

class GlobalDataManager : public DataUser
{
public:
	GlobalDataManager(std::string name);

	void init() {}
	//void display();
	QWidget* createPanel(QWidget* parent);
	void menuCallback(const std::string& message) {}
	void onDataItemChanged(const std::string& name);

	//methods
	void read_subject_names(vector<string>& mSubject_names);
	std::string& getName() { return mName; }
	std::string getSubject() { return mSelectionSubject; }

	void read_roi_names(vector<string>& mROI_names);

	//void setStreamline(const int& streamline_id);

	//BrainDataManager* mData;
	std::string	mName;

	//global parameters
	//subjects
	vector<string> mSubject_names;
	string mSelectionSubject;
	int mCurrentSubject;
	//int mStreamlineId;

	vector<string> mROI_names;
	string mSelectionROI;
	int mCurrentROI;

	string mLinethresh;
	float line_thresh=1.0f;

	string mRadius;
	float radius = 0.2f;
};

#endif // GLOBALDATAMANAGER_H