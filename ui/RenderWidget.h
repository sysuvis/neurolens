#ifndef SIMPLE_STREAMLINE_RENDER_WIDGET_H
#define SIMPLE_STREAMLINE_RENDER_WIDGET_H

#include "DisplayWidget.h"
#include "StreamlinePool3d.h"
#include "typeOperation.h"
#include "DataUser.h"
#include <string>
#include <vector>
#include "VolumeData.h"

class cudaStreamlineRenderer;
class BrainDataManager;

class RenderWidget : public DisplayWidget, public DataUser{
public:
	RenderWidget(int x, int y, int w, int h, std::string name, BrainDataManager* data);

//protected:
	void init();
	void display();
	
	void menuCallback(const std::string& message){}
	void onDataItemChanged(const std::string& name);
	
	//void mousePressEvent(const MouseEvent& e);
	//void mouseMoveEvent(const MouseEvent& e);
	void mouseReleaseEvent(const MouseEvent& e);

	void setStreamline(const int& sid);

//private:
	void setData();
	void singleStepJitterDisplay();
	void drawStreamlines();
	void drawStreamline(const int& sid);
	void updateSegmentsColor();
	void setStreamlines();
	void setStreamlines(float thresh);

	int mStreamlineId;
	std::vector<StreamlineSegment> mStreamlineSegments;
	BrainDataManager* mData;


};


#endif //SIMPLE_STREAMLINE_RENDER_WIDGET_H