#ifndef GRAPH_DRAW_WIDGET
#define GRAPH_DRAW_WIDGET

#include "DisplayWidget.h"
#include "GplGraph.h"
#include "DataUser.h"

class GraphDisplayWidget : public DisplayWidget, public DataUser{
public:
	GraphDisplayWidget(int x, int y, int w, int h, std::string name, GplGraph* graph=NULL);
	~GraphDisplayWidget();

	void resetGraph();
	void setGraph(GplGraph* graph){mGraph=graph;}
	void setKeepAspectRatio(const bool& value){mbKeepRatio = value;}
	GplGraph* getGraph(){return mGraph;}
	void setDrawTriangle(const bool& value){mbDrawTriangle=value;}
	void setEqualSizeNodes(const bool& value){mbEqualSizeNodes=value;}
	void setNodeSize(const float& value){mNodeRadius=value;}
	
	void init();
	void display();

	const std::vector<int>& getSelectedNodes() const {return mSelectedNodes;}

protected:
	virtual void menuCallback(const std::string& message);
	virtual void onDataItemChanged(const std::string& name);
	virtual void mousePressEvent(const MouseEvent& e);
	virtual void mouseMoveEvent(const MouseEvent& e);
	virtual void mouseReleaseEvent(const MouseEvent& e);

	virtual void drawNode(const vec2f& pos, const float& size);
	virtual void drawEdge(const vec2f& pos1, const vec2f& pos2, const float& size);
	
	//drawing parameters
	float			mEdgeWidth;
	float			mEdgeThresh;
	float			mNodeRadius;
	//brushing
	vector<int>			mSelectedNodes;

private:
	void updateDrawingBound();
	vec2f mapToDrawingCoordinate(const vec2f& pos);
	void getSelectedNodes(const vec2f& p1, const vec2f& p2, vector<int>& ret);
	virtual void onNodesSelected();
	
	//keep original aspect ratio
	bool			mbKeepRatio;
	
	//drawing flags
	bool			mbDrawTriangle;
	bool			mbEqualSizeNodes;

	//drawing area
	vec2f			mUpperBound;
	vec2f			mLowerBound;

	//parameter name in data manager
	std::string		mEdgeWidthName;
	std::string		mEdgeThreshName;
	std::string		mNodeRadiusName;

	//graph data
	GplGraph*		mGraph;

	//brushing
	vec2f			mStartPos;
	vec2f			mEndPos;
};

#endif //GRAPH_DRAW_WIDGET