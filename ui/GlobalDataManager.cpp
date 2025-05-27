#include "GlobalDataManager.h"

//Initialize
GlobalDataManager::GlobalDataManager(std::string name):
DataUser(),
mSelectionSubject(name + ".Subject:"),
mCurrentSubject(0),
//mSelectionROI(name+".ROI:"),
mCurrentROI(0),
mLinethresh(name+".Filter:"),
mRadius(name+".Radius:")
{
    read_subject_names(mSubject_names);
    read_roi_names(mROI_names);
    DataManager* manager = DataManager::sharedManager();

    manager->createEnum(mSelectionSubject, mSubject_names, mCurrentSubject, DATA_ITEM_ENUM_COMBOBOX, this);
    //manager->createEnum(mSelectionROI, mROI_names, mCurrentROI, DATA_ITEM_ENUM_COMBOBOX, this);
    manager->createFloat(mLinethresh, line_thresh, 200, 0, this, true);
    manager->createFloat(mRadius, radius, 0.8, 0.1, this, true);

}


void GlobalDataManager::onDataItemChanged(const std::string& name) {
    DataManager* manager = DataManager::sharedManager();
    bool b_success;
    if (name == mSelectionSubject) {
        mCurrentSubject =manager->getEnumValue(name, b_success);
        MessageCenter::sharedCenter()->processMessage(SELECT_SUBJECT_CHANGE_MSG, getName());
        
    }
    else if (name == SELECTED_LINE_ID_NAME) {
        //manager->getIntValue(name, b_success);
        MessageCenter::sharedCenter()->processMessage("Selected Streamline Change", getName());
    }
    /*else if(name==mSelectionROI)
    {
        mCurrentROI = manager->getEnumValue(name, b_success);
        MessageCenter::sharedCenter()->processMessage(SELECT_ROI_CHANGE_MSG, getName());
    }*/
    else if (name == mLinethresh) {
        //line_thresh = manager->getFloatValue(name, b_success);
        MessageCenter::sharedCenter()->processMessage(FILTER_LINES_MSG, getName());
    }
    else if (name == mRadius) {
        //line_thresh = manager->getFloatValue(name, b_success);
        MessageCenter::sharedCenter()->processMessage("Radius Changed", getName());
    }


}


QWidget* GlobalDataManager::createPanel(QWidget* parent){
	std::vector<std::string> pars;

	pars.push_back(SELECTED_LINE_ID_NAME);
	pars.push_back(mSelectionSubject);
    //pars.push_back(mSelectionROI);
    pars.push_back(mLinethresh);
    pars.push_back(mRadius);


	QWidget* ret = DataManager::sharedManager()->createInterface(" Fiber Parameters", pars, parent);
	return ret;
}

void GlobalDataManager::read_subject_names(std::vector<std::string>& mSubject_names) {
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

void GlobalDataManager::read_roi_names(vector<string>& mROI_names) {
    std::ifstream file("D:/DATA/brain/roi_name.txt");

    if (file.is_open()) {
        std::string line;
        while (getline(file, line)) {
            mROI_names.push_back(line);
        }
        file.close();
    }
    else {
        std::cout << "Unable to open the file" << std::endl;
    }
}