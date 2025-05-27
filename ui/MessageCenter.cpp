#include "MessageCenter.h"
#include "DataManager.h"
#include "VolumeData.h"
#include "GplGraph.h"
#include "ViewportWidget.h"
#include "GraphDisplayWidget.h"
#include "definition.h"
#include <vector>
#include <queue>
#include <algorithm>
#include "EmbeddingWidget.h"
#include "RenderWidget.h"
#include "ViewportWidget.h"
#include "BrainFiberData.h"
#include "GlobalDataManager.h"
#include "WindowsTimer.h"
#include "QueryWidget.h"

MessageCenter::MessageCenter(){}
MessageCenter::~MessageCenter(){}

MessageCenter* MessageCenter::sharedCenter(){
	static MessageCenter shared_center;
	return &shared_center;
}

void MessageCenter::processMessage(const std::string& message, const std::string& sender){
	DataManager* manager = DataManager::sharedManager();
	bool bSuccess;

	EmbeddingWidget*graph_display = reinterpret_cast<EmbeddingWidget*>(manager->getPointerValue("Graph Display.Pointer", bSuccess));
	RenderWidget *streamline_renderer = reinterpret_cast<RenderWidget*>(manager->getPointerValue("Render Widget.Pointer", bSuccess));
	ViewportWidget *viewport = reinterpret_cast<ViewportWidget*>(manager->getPointerValue("View Port.Pointer", bSuccess));
	BrainDataManager* data = reinterpret_cast<BrainDataManager*>(manager->getPointerValue("Brain Data.Pointer", bSuccess));
	GlobalDataManager* globaldata = reinterpret_cast<GlobalDataManager*>(manager->getPointerValue("Global Data.Pointer", bSuccess));
	QueryWidget* query_widget= reinterpret_cast<QueryWidget*>(manager->getPointerValue("Query Window.Pointer", bSuccess));
	

	if (message == SELECT_LINE_CHANGE_MSG) {
		int streamline_id = manager->getIntValue(SELECTED_LINE_ID_NAME, bSuccess);
		streamline_renderer->setStreamline(streamline_id);
		viewport->update();
	}
	else if (message == FILTER_LINES_MSG) {

		float thresh = manager->getFloatValue(globaldata->mLinethresh,bSuccess);
		streamline_renderer->setStreamlines(thresh);

		viewport->update();
	}
	else if (message == "Radius Changed") {
		//streamline_renderer->setData();
		float radius = manager->getFloatValue(globaldata->mRadius, bSuccess);
		data->mRenderer->updateTubeRadius(radius);

		viewport->update();
	}
	else if (message == SELECT_SUBJECT_CHANGE_MSG) {

		//data->setSubject(globaldata->mSelectionSubject);

		string subject = globaldata->mSubject_names[globaldata->mCurrentSubject];
		subject = subject == "Cohort" ? "003_S_4081" : subject;
		data->setSubject(subject);
		data->clearROIInvolved();
		data->setData();
		streamline_renderer->setData();
		//graph_display->setData();
		//manager->ChangeIntMAX(SELECTED_LINE_ID_NAME, data->getNumStreamlines() - 1);
	}
	else if (message == SELECT_POINT_CHANGE_MSG) {
		data->setROIs(graph_display->mSelectedROIs);
		//data->setEdges(graph_display->edges);

		cout << "Select ROI :" << endl;
		for (int idx=0; idx <graph_display->mSelectedROIs.size(); idx++) {
			cout<<globaldata->mROI_names[graph_display->mSelectedROIs[idx]]<<endl;
			
		}
		printf("------\n");
		//cout << graph_display->edges.size() << endl;
		//streamline_renderer->updateSegmentsColor();
		viewport->update();
	}
	else if (message == SELECT_ROI_CHANGE_MSG) {
		/*string roi = globaldata->mROI_names[globaldata->mCurrentROI];
		data->setROI(roi);
		graph_display->setData();

		viewport->update();*/
	}
	else if (message=="Embedding Changed")
	{
		vector<int> selected_left = graph_display->selected_left;
		vector<int> selected_right = graph_display->selected_right;
		
		if (graph_display->mCurrentSubject!= graph_display->mSubject_names.size()-1){ 
			selected_left.clear();
			selected_left.push_back(graph_display->mCurrentSubject);
		}
		if (graph_display->mCurrentSubject2 != graph_display->mSubject_names.size() - 1) {
			selected_right.clear();
			selected_right.push_back(graph_display->mCurrentSubject2);
		}
		graph_display->set_selected_individuals(selected_left, selected_right);

		//turn on the change render view automatically
		//globaldata->mCurrentSubject = graph_display->mCurrentSubject;
		//manager->setEnumValue(globaldata->mSelectionSubject, globaldata->mCurrentSubject);

		graph_display->embedding_view.setData(selected_left, selected_right, graph_display->mCurrentConv, graph_display->mPoolingNames[graph_display->mCurrentPooling]);
		viewport->update();
	}
	//else if (message == "Left Embedding Changed")
	//{
	//	vector<int> selected_left = graph_display->selected_left;

	//	if (graph_display->mCurrentSubject != graph_display->mSubject_names.size() - 1) {
	//		selected_left.clear();
	//		selected_left.push_back(graph_display->mCurrentSubject);
	//	}
	//	
	//	graph_display->set_selected_left(selected_left);
	//	//globaldata->mCurrentSubject = graph_display->mCurrentSubject;
	//	//manager->setEnumValue(globaldata->mSelectionSubject, globaldata->mCurrentSubject);
	//	graph_display->embedding_view.setData(graph_display->mCurrentConv, graph_display->mPoolingNames[graph_display->mCurrentPooling]);
	//	//graph_display->embedding_view.set_left_data(selected_left);
	//	
	//	viewport->update();
	//}
	//else if (message == "Right Embedding Changed")
	//{
	//	vector<int> selected_right = graph_display->selected_right;

	//	if (graph_display->mCurrentSubject2 != graph_display->mSubject_names.size() - 1) {
	//		selected_right.clear();
	//		selected_right.push_back(graph_display->mCurrentSubject2);
	//	}
	//	graph_display->set_selected_right(selected_right);
	//	//globaldata->mCurrentSubject = graph_display->mCurrentSubject;
	//	//manager->setEnumValue(globaldata->mSelectionSubject, globaldata->mCurrentSubject);
	//	
	//	graph_display->embedding_view.setData(graph_display->mCurrentConv, graph_display->mPoolingNames[graph_display->mCurrentPooling]);
	//	graph_display->embedding_view.set_right_data(selected_right);
	//	viewport->update();
	//}
	else if (message == "Scale Changed") {
		/*vector<int> selected_left = graph_display->selected_left;
		vector<int> selected_right = graph_display->selected_right;
		graph_display->embedding_view.setData(selected_left, selected_right,graph_display->mCurrentConv, graph_display->mPoolingNames[graph_display->mCurrentPooling]);
		
		graph_display->embedding_view.mLeft.setSelectedNodes(graph_display->mSelectedROIs);
		graph_display->embedding_view.mRight.setSelectedNodes(graph_display->mSelectedROIs);
		graph_display->embedding_view.mBrain.setSelectedNodes(graph_display->mSelectedROIs);

		graph_display->embedding_view.mLeft.setEdgeDatas(graph_display->embedding_view.edges);
		graph_display->embedding_view.mRight.setEdgeDatas(graph_display->embedding_view.edges2);

		graph_display->embedding_view.mLeft.clearSelectionBox();*/

		vector<int> selected_left = graph_display->selected_left;
		vector<int> selected_right = graph_display->selected_right;
		graph_display->embedding_view.setData(selected_left, selected_right, graph_display->mCurrentConv, graph_display->mPoolingNames[graph_display->mCurrentPooling]);

		graph_display->embedding_view.mLeft.mGraph.setSelectedNodes(graph_display->mSelectedROIs);
		graph_display->embedding_view.mRight.mGraph.setSelectedNodes(graph_display->mSelectedROIs);
		graph_display->embedding_view.mBrain.setSelectedNodes(graph_display->mSelectedROIs);

		graph_display->embedding_view.mLeft.mGraph.setEdgeDatas(graph_display->embedding_view.edges);
		graph_display->embedding_view.mRight.mGraph.setEdgeDatas(graph_display->embedding_view.edges2);

		graph_display->embedding_view.mLeft.mGraph.clearSelectionBox();

		graph_display->embedding_view.mRight.setData();
		graph_display->embedding_view.mLeft.setData();


		viewport->update();
	}
	else if (message=="Clear")
	{
		vector<int> selected_left = graph_display->selected_left;
		vector<int> selected_right = graph_display->selected_right;
		
		/*if (graph_display->mCurrentSubject != graph_display->mSubject_names.size() - 1) {
			selected_left.clear();
			selected_left.push_back(graph_display->mCurrentSubject);
		}
		if (graph_display->mCurrentSubject2 != graph_display->mSubject_names.size() - 1) {
			selected_right.clear();
			selected_right.push_back(graph_display->mCurrentSubject2);
		}*/
		//graph_display->set_selected_individuals(selected_left, selected_right);
		graph_display->embedding_view.setData(selected_left, selected_right,graph_display->mCurrentConv, graph_display->mPoolingNames[graph_display->mCurrentPooling]);
		//graph_display->embedding_view.set_left_data(selected_left);
		//graph_display->embedding_view.set_right_data(selected_right);

		
		//cout << graph_display->mSelectedROIs.size() << endl;
		viewport->update();
	}
	else if (message == "Reset")
	{
		graph_display->overview.setData();
		graph_display->overview.scatter_view.clearSelectionBox();

		viewport->update();
	}
	else if (message == "Select Individuals") {
		std::vector<int> selected_left = graph_display->overview.selected_left;
		std::vector<int> selected_right = graph_display->overview.selected_right;
		graph_display->set_selected_individuals(selected_left, selected_right);

		//graph_display->embedding_view.setData(selected_left, selected_right, graph_display->mCurrentConv, graph_display->mPoolingNames[graph_display->mCurrentPooling]);

		int left_current = selected_left.size() == 1 ? selected_left[0] : graph_display->mSubject_names.size()-1;
		int right_current = selected_right.size() == 1 ? selected_right[0] : graph_display->mSubject_names.size() - 1;

		if (selected_right.empty()) { right_current = graph_display->mCurrentSubject2; }

		//graph_display->mCurrentSubject = left_current;
		//graph_display->mCurrentSubject2 = right_current;
		manager->setEnumValue(graph_display->mSelectionSubject, left_current);
		manager->setEnumValue(graph_display->mSelectionSubject2, right_current);
		graph_display->embedding_view.setData(selected_left, selected_right, graph_display->mCurrentConv, graph_display->mPoolingNames[graph_display->mCurrentPooling]);

		std::cout << "Select Subject: "<< globaldata->mSubject_names[selected_left[0]] << std::endl;
		

		viewport->update();
	}
	else if (message == "Select Left Individuals") {
		std::vector<int> selected_left = graph_display->overview.selected_left;
		std::vector<int> selected_right = graph_display->overview.selected_right;
		graph_display->set_selected_individuals(selected_left, selected_right);
		
		viewport->update();
	}
	else if(message=="Select Edges") {

		cout << "Select bundles:" << endl;
		for (int idx = 0; idx < graph_display->edgesSelected.size(); idx++) {
			int roi_id_1 = graph_display->edgesSelected[idx][0] + 1;
			int roi_id_2 = graph_display->edgesSelected[idx][1] + 1;
			cout << roi_id_1<<"--->"<< roi_id_2<< endl;

		}
		printf("------\n");
		
		string BundlePath = graph_display->chooseBundlePath();
		std::vector<int> comp_seelcted = graph_display->selected_right;
		graph_display->mJointDisplay.setData(BundlePath, graph_display->mCurrent_X, graph_display->mCurrent_Y, comp_seelcted);

		int start=graph_display->mJointDisplay.extractDataFromPath(BundlePath).start;
		int end = graph_display->mJointDisplay.extractDataFromPath(BundlePath).end;
		graph_display->embedding_view.mBrain.set_brain_edge({ start-1,end-1 });

		data->setEdges(graph_display->edgesSelected);

		viewport->update();
	}
	else if (message == "Marginal Distribution Changed") {

		string BundlePath = graph_display->chooseBundlePath();
		std::vector<int> comp_seelcted = graph_display->selected_right;
		graph_display->mJointDisplay.setData(BundlePath, graph_display->mCurrent_X, graph_display->mCurrent_Y, comp_seelcted);
		viewport->update();

	}
	else if (message == "Query") {
		query_widget->set_layout(2410, 500, 1210,940);
	}
	else if (message == "Query Status Changed") {
		bool flag = manager->getBoolValue(query_widget->query_status, bSuccess);
		query_widget->set_query_status(flag);
	}
	else if (message == "Submit")
	{
		QString text_input = query_widget->get_query_content();
		query_widget->send_message_to_server(text_input);
		

	}
	else if (message == "Check ROI")
	{
		//graph_display->current_ROI = graph_display->embedding_view.check_ROI;
		manager->setEnumValue(graph_display->ROI_selection, graph_display->embedding_view.check_ROI);
		viewport->update();
	}
	else if (message == "NodeTrix Set")
	{
		graph_display->embedding_view.mLeft.setData();
		graph_display->embedding_view.mRight.setData();


		viewport->update();
	}
	else if (message == "SET SUBJECT BY EDGE")
	{
		string subject = graph_display->embedding_view.subject_left;
		data->setSubject(subject);
		data->clearROIInvolved();
		data->setData();
		streamline_renderer->setData();


		viewport->update();
	}
	
	

	
}
