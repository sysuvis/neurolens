#include "EmbeddingWidget.h"
#include "DataManager.h"
#include "definition.h"
#include "VRComplex.h"
#include "BrainFiberData.h"
#include "MessageCenter.h"
#include <iostream>
#include <fstream>
#include "glText.h"
#include "GlobalDataManager.h"
#define SINGLE_WIN_HEIGHT	1200

EmbeddingWidget::EmbeddingWidget(int x, int y, int w, int h, std::string name, BrainDataManager* data, GlobalDataManager* gdata):
	//成员变量赋值
	DisplayWidget(x, y, w, h, name),
	DataUser(),
	mData(data),
	mSelectionPooling(name + ". Pooling:"),
	mSelectionConv(name + ". Message Passing:"),
	mSelectionSubject(name + ". Subject"),
	mSelectionSubject2(name + ". Subject2"),
	mButton_clear(name+". Clear"),
	mButton_reset(name+". Reset"),
	mButton_query(name+". Query"),
	mSelection_X(name+". marginal_X"),
	mSelection_Y(name + ". marginal_Y"),
	ROI_selection(name+". ROI")
{
	
	embedding_view.set_data_pointer(gdata);
	ROI_names = embedding_view.gdata->mROI_names;
	// 
	//global data
	read_subject_names(mSubject_names);

	//Initialize
	DataManager* manager = DataManager::sharedManager();
	manager->createEnum(mSelectionSubject, mSubject_names, mCurrentSubject, DATA_ITEM_ENUM_COMBOBOX, this);
	manager->createEnum(mSelectionSubject2, mSubject_names, mCurrentSubject2, DATA_ITEM_ENUM_COMBOBOX, this);
	manager->createEnum(mSelectionPooling, mPoolingNames, mCurrentPooling, DATA_ITEM_ENUM_COMBOBOX, this);
	manager->createEnum(ROI_selection, ROI_names, current_ROI, DATA_ITEM_ENUM_COMBOBOX, this);
	manager->createInt(mSelectionConv, mCurrentConv, 5, 1, this, true);

	manager->createTrigger(mButton_clear, this); 
	manager->createTrigger(mButton_reset, this);
	manager->createTrigger(mButton_query, this);

	manager->createEnum(mSelection_X, mFeature, mCurrent_X, DATA_ITEM_ENUM_COMBOBOX, this);
	manager->createEnum(mSelection_Y, mFeature, mCurrent_Y, DATA_ITEM_ENUM_COMBOBOX, this);

	// layout settings
	mJointDisplay.setArea(makeRectDisplayArea(makeVec2f(100.0f, 70.0f), makeVec2f(300.0f, 0.0f), makeVec2f(0.0f, 300.0f)));

	overview.setArea(makeRectDisplayArea(makeVec2f(60.0f, 800.0f), makeVec2f(850.0f, 0.0f), makeVec2f(0.0f, 360.0f)));

	//embedding_view.setArea(makeRectDisplayArea(makeVec2f(60.0f, 500.0f), makeVec2f(1000.0f, 0.0f), makeVec2f(0.0f, 300.0f)));
	embedding_view.setArea(makeRectDisplayArea(makeVec2f(60.0f, 510.0f), makeVec2f(1000.0f, 0.0f), makeVec2f(0.0f, 300.0f)));

	setData();

	


	//addMenuItem("do something");
}
void EmbeddingWidget::setData() {
	//overview
	overview.setData();


	//embedding view
	//embedding_view.setData(selected_left, selected_right,mCurrentConv,mPoolingNames[mCurrentPooling]);
	embedding_view.setData(selected_left, selected_right, mCurrentConv, mPoolingNames[mCurrentPooling]);
	
	//JointDisplay management
	string BundlePath=chooseBundlePath();
	std::vector<int> default_comp_selected;
	mJointDisplay.setData(BundlePath, mCurrent_X, mCurrent_Y, default_comp_selected); //data_path and variable flags

	

}

void EmbeddingWidget::set_selected_individuals(std::vector<int> left, std::vector<int> right) {
	selected_left = left;
	selected_right = right;
}

void EmbeddingWidget::set_selected_left(std::vector<int> left) {
	selected_left = left;
}

void EmbeddingWidget::set_selected_right(std::vector<int> right) {
	selected_right = right;
}

void EmbeddingWidget::display()
{

	glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);

	glEnable(GL_LINE_SMOOTH);
	glEnable(GL_POLYGON_SMOOTH);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, mWidth, 0, mHeight, -1000, 1000);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	//draw brain image
	drawImage(makeVec2f(370, 490), "D:/DATA/brain/ICON/brain_dark.png", 260);
	
	mJointDisplay.display();

	overview.display();
	//embedding_view.display();
	embedding_view.display();
	
	label_display();

	drawText(makeVec2f(20, 360), mFeature[mCurrent_Y], 12, 5, ColorMap::getColorByName(ColorMap::Black));
	drawText(makeVec2f(390, 40), mFeature[mCurrent_X], 12, 5, ColorMap::getColorByName(ColorMap::Black));

	glPopAttrib();
}

void EmbeddingWidget::label_display() {
	//draw boundry
	drawBoundary(8 | 4 | 2 | 1);

	//view titles
	drawText(makeVec2f(120, 1020), "Age", 18, 1,ColorMap::getColorByName(ColorMap::Black));
	drawText(makeVec2f(90, 850 ), "Education", 18, 1, ColorMap::getColorByName(ColorMap::Black));
	drawText(makeVec2f(350, 1130 ), "Confusion Matrix", 18, 1, ColorMap::getColorByName(ColorMap::Black));
	drawText(makeVec2f(700, 1130 ), "Global Embedding", 18, 2, ColorMap::getColorByName(ColorMap::Black));
	drawText(makeVec2f(70, 10 ), "Geometric Features", 18, 2, ColorMap::getColorByName(ColorMap::Black));
	drawText(makeVec2f(630, 10 ), "DTI Features", 18, 2, ColorMap::getColorByName(ColorMap::Black));

	//matrix ticks
	drawText(makeVec2f(370, 1070 ), "AD", 13, 3, ColorMap::getColorByName(ColorMap::Black));
	drawText(makeVec2f(350,  1020 ), "lMCI", 13, 3, ColorMap::getColorByName(ColorMap::Black));
	drawText(makeVec2f(350,  970 ), "eMCI", 13, 2, ColorMap::getColorByName(ColorMap::Black));
	drawText(makeVec2f(370,  920 ), "HC", 13, 2, ColorMap::getColorByName(ColorMap::Black));
	drawText(makeVec2f(450,  860 ), "Prediction", 13, 2, ColorMap::getColorByName(ColorMap::Black));

	//parallel coordinate ticks
	drawText(makeVec2f(550, 380 ), "FA", 13, 5, ColorMap::getColorByName(ColorMap::Black));
	drawText(makeVec2f(650, 380 ), "MD", 13, 5, ColorMap::getColorByName(ColorMap::Black));
	drawText(makeVec2f(750, 380 ), "AXD", 13, 5, ColorMap::getColorByName(ColorMap::Black));
	drawText(makeVec2f(850, 380 ), "RD", 13, 5, ColorMap::getColorByName(ColorMap::Black));
	drawText(makeVec2f(950, 380 ), "Con", 13, 5, ColorMap::getColorByName(ColorMap::Black));

	//joint distribution axes
	drawArrow(makeVec2f(80, 52), makeVec2f(380, 52), 12, 5, 3, ColorMap::getColorByName(ColorMap::Black));
	drawArrow(makeVec2f(80, 50), makeVec2f(80, 350), 12, 5, 3, ColorMap::getColorByName(ColorMap::Black));

	//parting line
	drawDashedLine(makeVec2f(0, 840 ), makeVec2f(1000,840 ), 5, ColorMap::getColorByName(ColorMap::Gray), 1, 0x00FF);
	drawDashedLine(makeVec2f(0, 420 ), makeVec2f(1000, 420 ), 5, ColorMap::getColorByName(ColorMap::Gray), 1, 0x00FF);

	
	//brain region legend 
	float legend_size = 20;
	float gap = 160;
	drawRectBorder(makeVec2f(30, 440), makeVec2f(30+legend_size, 440+ legend_size), 3, ColorMap::getColorByName(ColorMap::Bright_lavender));
	drawRectBorder(makeVec2f(30 + gap, 440), makeVec2f(30 + gap + legend_size, 440 + legend_size), 3, ColorMap::getColorByName(ColorMap::Bright_turquoise));
	drawRectBorder(makeVec2f(30 + gap*2, 440), makeVec2f(30 + gap * 2 + legend_size, 440 + legend_size), 3, ColorMap::getColorByName(ColorMap::Black));
	drawRectBorder(makeVec2f(30 + gap * 3, 440), makeVec2f(30 + gap * 3 + legend_size, 440 + legend_size), 3, ColorMap::getColorByName(ColorMap::Stil_de_grain_yellow));

	drawText(makeVec2f(60, 435), "Frontal", 13, 5, ColorMap::getColorByName(ColorMap::Black));
	drawText(makeVec2f(60 + gap * 1, 435), "Parietal", 13, 5, ColorMap::getColorByName(ColorMap::Black));
	drawText(makeVec2f(60 + gap * 2, 435), "Temporal", 13, 5, ColorMap::getColorByName(ColorMap::Black));
	drawText(makeVec2f(60 + gap * 3, 435), "Occipital", 13, 5, ColorMap::getColorByName(ColorMap::Black));

}

string EmbeddingWidget::chooseBundlePath() {
	string subject = mSubject_names[mCurrentSubject];
	
	if (edgesSelected.empty()) { return makeBundlePath(subject,8, 23); }
	string cur_path;
	int maxFile_index = 0;
	long maxFile_length = 0;
	for (int idx = 0; idx < edgesSelected.size(); idx++) {
		int roi_id_1 = edgesSelected[idx][0] ;
		int roi_id_2 = edgesSelected[idx][1] ;
		cur_path = makeBundlePath(subject,roi_id_1, roi_id_2);
		long length = checkFileSize(cur_path);
		if (maxFile_length < length) {
			maxFile_index = idx;
			maxFile_length = length;
		}

	}

	if (maxFile_length == 0) {
		cout << "no avaliable bundle path" << endl;
		return cur_path;
	}
	else
	{
		return makeBundlePath(subject,edgesSelected[maxFile_index][0], edgesSelected[maxFile_index][1]);
	}

}

string EmbeddingWidget::makeBundlePath(string subject,int start, int end) {
	if (subject == "Cohort") {
		string geometric_feature_path = "D:/DATA/brain/geometric_features/" + subject + "/";
		return geometric_feature_path + "empty_default.txt";
	}
	else
	{
		string geometric_feature_path = "D:/DATA/brain/geometric_features/" + subject + "/";
		if(end<start){ return geometric_feature_path + to_string(end + 1) + "_" + to_string(start + 1) + ".dat"; }
		return geometric_feature_path + to_string(start + 1) + "_" + to_string(end + 1) + ".dat";
	}

}

long EmbeddingWidget::checkFileSize(string path) {
	std::ifstream file(path, std::ifstream::ate | std::ifstream::binary);

	if (!file) {
		return -1;
	}

	return file.tellg(); 
}


void EmbeddingWidget::read_subject_names(std::vector<std::string>& mSubject_names) {
	std::ifstream file("D:/DATA/brain/subject_name.txt");

	if (file.is_open()) {
		std::string line;
		while (getline(file, line)) {
			mSubject_names.push_back(line);
		}
		file.close();
	}
	else {
		std::cout << "Unable to open the file" << std::endl;
	}

	mSubject_names.push_back("Cohort");

}

//panel
QWidget* EmbeddingWidget::createPanel(QWidget* parent) {
	std::vector<std::string> pars;
	//pars.push_back(ROI_selection);
	pars.push_back(mButton_reset);
	pars.push_back(mSelectionSubject);
	pars.push_back(mSelectionSubject2);
	pars.push_back(mSelectionPooling);
	pars.push_back(mSelectionConv);
	pars.push_back(mButton_clear);
	pars.push_back(mSelection_X);
	pars.push_back(mSelection_Y);
	//pars.push_back(mButton_query);

	QWidget* ret = DataManager::sharedManager()->createInterface("    Exploration", pars, parent);
	return ret;
}


//listener
void EmbeddingWidget::onDataItemChanged(const std::string& name) {
	DataManager* manager = DataManager::sharedManager();
	bool b_success;
	if (name == mSelectionPooling) {
		mCurrentPooling = manager->getEnumValue(name, b_success);
		MessageCenter::sharedCenter()->processMessage("Scale Changed", getName());

	}
	else if (name == mSelectionConv)
	{
		mCurrentConv = manager->getIntValue(name, b_success);
		MessageCenter::sharedCenter()->processMessage("Scale Changed", getName());
	}
	else if (name == mSelectionSubject) {
		mCurrentSubject = manager->getEnumValue(name, b_success);
		MessageCenter::sharedCenter()->processMessage("Embedding Changed", getName());
	}
	else if (name == mSelectionSubject2) {
		mCurrentSubject2 = manager->getEnumValue(name, b_success);
		MessageCenter::sharedCenter()->processMessage("Embedding Changed", getName());
	}
	else if (name==mButton_clear)
	{
		MessageCenter::sharedCenter()->processMessage("Clear", getName());
	}
	else if (name == mButton_reset)
	{
		MessageCenter::sharedCenter()->processMessage("Reset", getName());
	}
	else if (name == mSelection_X )
	{
		mCurrent_X = manager->getEnumValue(name, b_success);
		MessageCenter::sharedCenter()->processMessage("Marginal Distribution Changed", getName());
	}
	else if ( name == mSelection_Y)
	{
		mCurrent_Y = manager->getEnumValue(name, b_success);
		MessageCenter::sharedCenter()->processMessage("Marginal Distribution Changed", getName());
	}
	else if (name == mButton_query)
	{
		MessageCenter::sharedCenter()->processMessage("Query", getName());
	}



}

//mouse interact for brushing
void EmbeddingWidget::mousePressEvent(const MouseEvent& e) {
	if (e.button == MOUSE_LEFT_BUTTON) {

		//embedding wiew
		if (embedding_view.inDisplayArea(e.pos)) {
			embedding_view.press(e.pos);
		}
	
		//overview
		if (overview.inDisplayArea(e.pos)) {
			overview.press(e.pos);
		}
	
	
	}

	if (e.button == MOUSE_RIGHT_BUTTON) {
		//overview
		if (overview.inDisplayArea(e.pos)) {
			overview.press_right(e.pos);
		}

	}

	if (e.button == MOUSE_MIDDLE_BUTTON) {
		//embedding brain
		if (embedding_view.inDisplayArea(e.pos)) {
			embedding_view.press_right(e.pos);
		}
	}

	//DisplayWidget::mousePressEvent(e);
	redraw();
}

void EmbeddingWidget::mouseMoveEvent(const MouseEvent& e) {
	MouseEvent pe = getPreviousMouseEvent();
	if (pe.button == MOUSE_LEFT_BUTTON) {

		//embedding wiew
		if (embedding_view.inDisplayArea(e.pos)) {
			embedding_view.move(e.pos);
		}
		if (overview.inDisplayArea(e.pos)) { overview.move(e.pos); }
	}	
		//overview
		if (overview.in_interaction()) {overview.drag(e.pos);}
		
	if (pe.button == MOUSE_RIGHT_BUTTON) {
		if (overview.inDisplayArea(e.pos)) { overview.move(e.pos); }
	}

	
	redraw();
	//DisplayWidget::mouseMoveEvent(e);
	
}

void EmbeddingWidget::mouseReleaseEvent(const MouseEvent& e) {
	MouseEvent pe = getPreviousMouseEvent();

	//embedding wiew
	if (embedding_view.inDisplayArea(e.pos)) {
		if (pe.button == MOUSE_LEFT_BUTTON) {
			
			embedding_view.left_mouse_release(e.pos);
			mSelectedROIs = embedding_view.mSelectedROIs;
			MessageCenter::sharedCenter()->processMessage(SELECT_POINT_CHANGE_MSG, getName());
		}
		if (e.button == MOUSE_RIGHT_BUTTON) {
			embedding_view.right_mouse_release(e.pos);
			edgesSelected = embedding_view.edgesSelected;
			//edgesSelected = { {16,62} }; // need -1 
			MessageCenter::sharedCenter()->processMessage("Select Edges", getName());

		}
	}
	
	//overview
	if (overview.inDisplayArea(e.pos)) { 
		if (pe.button == MOUSE_LEFT_BUTTON) {
			overview.release(e.pos); 
		}
		if (pe.button == MOUSE_RIGHT_BUTTON) {
			overview.release_right(e.pos);
		}
		
	}


	redraw();
	//DisplayWidget::mouseReleaseEvent(e);
}










