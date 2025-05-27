#include "GraphDisplayWidget.h"
#include "DataManager.h"
#include "MessageCenter.h"
#include "ColorMap.h"

GraphDisplayWidget::GraphDisplayWidget(int x, int y, int w, int h, std::string name, GplGraph* graph)
:DisplayWidget(x, y, w, h, name),
DataUser(),
mbKeepRatio(true),
mbDrawTriangle(false),
mbEqualSizeNodes(false),
mGraph(graph),
mEdgeWidth(1.0f),
mEdgeThresh(0.0f),
mNodeRadius(1.0f),
mEdgeWidthName(name+".edge width"),
mEdgeThreshName(name+".edge thresh"),
mNodeRadiusName(name+".node radius"),
mStartPos(makeVec2f(-GPL_INFINITE, -GPL_INFINITE))
{
	addMenuItem("Adjust Parameters");

	DataManager* manager = DataManager::sharedManager();
	manager->createFloat(mEdgeWidthName, mEdgeWidth, 0.0f, 20.0f, this, true);
	manager->createFloat(mEdgeThreshName, mEdgeThresh, 0.0f, 1.0f, this, true);
	manager->createFloat(mNodeRadiusName, mNodeRadius, 0.0f, 20.0f, this, true);
}

GraphDisplayWidget::~GraphDisplayWidget(){

}

void GraphDisplayWidget::resetGraph(){
	if (!mGraph) delete mGraph;
	mGraph = NULL;
}

void GraphDisplayWidget::init(){

}

void GraphDisplayWidget::display(){
	if (!mGraph){
		return;
	}

	updateDrawingBound();

	glPushAttrib(GL_ENABLE_BIT|GL_LINE_BIT|GL_CURRENT_BIT);
	glDisable(GL_DEPTH_TEST);

	glEnable( GL_LINE_SMOOTH );
	glEnable( GL_POLYGON_SMOOTH );
	glHint( GL_LINE_SMOOTH_HINT, GL_NICEST );
	glHint( GL_POLYGON_SMOOTH_HINT, GL_NICEST );

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(mLowerBound.x, mUpperBound.x, mLowerBound.y, mUpperBound.y);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glDisable(GL_LIGHTING);

	//draw edges
 	vector<GplNode>* nodes = mGraph->getNodes();
 	vec2f pos1, pos2;
 	float size;
	if(mbDrawTriangle){
		vector<GplTriEdges>* edges = mGraph->getTriangleEdges();
		//for each edge
		for (int i=0; i<edges->size(); ++i){
			pos1 = (*nodes)[(*edges)[i].i].pos;
			pos2 = (*nodes)[(*edges)[i].j].pos;
			drawEdge(pos1, pos2, 0.1f);
		}
	} else {
		vector<GplEdge>* edges = mGraph->getEdges();
		//for each edge
		for (int i=0; i<edges->size(); ++i){
			size = (*edges)[i].size;
			if (size<mEdgeThresh) continue;

			pos1 = (*nodes)[(*edges)[i].i].pos;
			pos2 = (*nodes)[(*edges)[i].j].pos;

			drawEdge(pos1, pos2, size);
		}
	}

	//draw nodes
	for (int i=0; i<nodes->size(); ++i){
		size = (*nodes)[i].size;
		pos1 = (*nodes)[i].pos;
		drawNode(pos1, size);
	}

	if(mStartPos.x>-GPL_INFINITE){
		drawRect(mStartPos, mEndPos, makeVec4f(-1.0f,-1.0f,-1.0f,-1.0f));
	}

	glPopAttrib();
}

void GraphDisplayWidget::drawNode(const vec2f& pos, const float& size){
	//vec4f color = getColor(size);
	//drawCircle(pos1, size*mNodeRadius, color);
	if(mbEqualSizeNodes){
		//drawSquare(pos, mNodeRadius*0.5f, color);
	} else {
		//drawSquare(pos, size*mNodeRadius*0.5f, color);
	}
}

void GraphDisplayWidget::drawEdge(const vec2f& pos1, const vec2f& pos2, const float& size){
	glLineWidth(size*mEdgeWidth);
	vec4f color = ColorMap::getRainbowColor(size);
	glColor4f(color.x, color.y, color.z, color.w);
	glBegin(GL_LINES);
	glVertex2f(pos1.x, pos1.y);
	glVertex2f(pos2.x, pos2.y);
	glEnd();
}

void GraphDisplayWidget::updateDrawingBound(){
	float l, r, b, t, w, h, x, y;
	mGraph->getRange(l, r, b, t);

	x = (r+l)*0.5f;
	y = (t+b)*0.5f;
	if (mbKeepRatio){
		if((r-l)*mHeight>(t-b)*mWidth){//fit width
			w = (r-l)*0.55f;
			h = mHeight/mWidth*w;
		} else {//fit height
			h = (t-b)*0.55f;
			w = mWidth/mHeight*h;
		}
	} else {
		w = (r-l)*0.55f;
		h = (t-b)*0.55f;
	}

	mUpperBound = makeVec2f(x+w, y+h);
	mLowerBound = makeVec2f(x-w, y-h);
}

vec2f GraphDisplayWidget::mapToDrawingCoordinate(const vec2f& pos){
	float fx, fy;
	fx = pos.x/mWidth;
	fy = pos.y/mHeight;
	
	vec2f ret;
	ret.x = fx*mUpperBound.x+(1.0f-fx)*mLowerBound.x;
	ret.y = fy*mUpperBound.y+(1.0f-fy)*mLowerBound.y;

	return ret;
}

void GraphDisplayWidget::getSelectedNodes(const vec2f& p1, const vec2f& p2, vector<int>& ret){
	if (!mGraph) return;
	
	vec2f ub, lb;
	if (p1.x>p2.x){
		ub.x = p1.x; lb.x = p2.x;
	} else {
		ub.x = p2.x; lb.x = p1.x;
	}
	if (p1.y>p2.y){
		ub.y = p1.y; lb.y = p2.y;
	} else {
		ub.y = p2.y; lb.y = p1.y;
	}
	//for all nodes
	vec2f pos;
	vector<GplNode>* nodes = mGraph->getNodes();
	for (int i=0; i<nodes->size(); ++i){
		pos = (*nodes)[i].pos;
		if (pos.x<=ub.x && pos.x>=lb.x && pos.y<=ub.y && pos.y>=lb.y) {
			ret.push_back(i);
		}
	}

}

void GraphDisplayWidget::menuCallback(const std::string& message){
	if (message=="Adjust Parameters"){
		DataManager* manager = DataManager::sharedManager();
		std::vector<std::string> pars;
		pars.push_back(mEdgeWidthName);
		pars.push_back(mNodeRadiusName);
		pars.push_back(mEdgeThreshName);
		manager->createInterface("Adjust Parameters", pars, NULL);
	} else {
		MessageCenter* center = MessageCenter::sharedCenter();
		center->processMessage(message, getName());
		redraw();
	}
}

void GraphDisplayWidget::onDataItemChanged(const std::string& name){
	std::string& widgetName = getName();
	DataManager* manager = DataManager::sharedManager();
	bool bSuccess;

	if (name==mEdgeWidthName){
		mEdgeWidth = manager->getFloatValue(name, bSuccess);
	} else if (name==mNodeRadiusName){
		mNodeRadius = manager->getFloatValue(name, bSuccess);
	} else if(name==mEdgeThreshName){
		mEdgeThresh = manager->getFloatValue(name, bSuccess);
	}
	redraw();
}

void GraphDisplayWidget::onNodesSelected(){
	MessageCenter* center = MessageCenter::sharedCenter();
	center->processMessage("Nodes Selected", getName());
}


void GraphDisplayWidget::mousePressEvent(const MouseEvent& e){
	if (e.button==MOUSE_LEFT_BUTTON) {
		mStartPos = mapToDrawingCoordinate(e.pos);
		mEndPos = mapToDrawingCoordinate(e.pos);
	} else {
		DisplayWidget::mousePressEvent(e);
	}
	redraw();
}

void GraphDisplayWidget::mouseMoveEvent(const MouseEvent& e){
	if (getPreviousMouseEvent().button==MOUSE_LEFT_BUTTON) {
		mEndPos = mapToDrawingCoordinate(e.pos);

		//mSelectedNodes.clear();
		//getSelectedNodes(mStartPos, mEndPos, mSelectedNodes);
		//onNodesSelected();
	} else {
		DisplayWidget::mousePressEvent(e);
	}
	redraw();
}

void GraphDisplayWidget::mouseReleaseEvent(const MouseEvent& e){
	if (getPreviousMouseEvent().button==MOUSE_LEFT_BUTTON) {
		mSelectedNodes.clear();
		getSelectedNodes(mStartPos, mEndPos, mSelectedNodes);
		onNodesSelected();
		
		mStartPos = makeVec2f(-GPL_INFINITE, -GPL_INFINITE);
	} else {
		DisplayWidget::mouseReleaseEvent(e);
	}
	redraw();
}