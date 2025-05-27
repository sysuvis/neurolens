#pragma once

#include "VisibilityGraphDisplayWidget.h"
#include "DataManager.h"
#include "MessageCenter.h"
#include "definition.h"
#include "VRComplex.h"

VisibilityGraphDisplayWidget::VisibilityGraphDisplayWidget(int x, int y, int w, int h, std::string name,VisibilityGraphDataManager* data) :
//成员变量赋值
DisplayWidget(x, y, w, h, name),
DataUser(),
mData(data),
mFilter(makeRange(0.0f, 20.0f)),
mFilterLowName(name + ".Filter Low Bound"),
mFilterUpName(name + ".Filter Distance"),
mNormalization(makeRange(0.0f, 100.0f)),
mNormalizationLowName(name + ".Norm Low Bound"),
mNormalizationUpName(name + ".Degree Distance"),
mMatchDegreeDistThresh(0.2f),
mMatchDegreeDistThreshName(name + ".Similarity Thresh"),
mSelectedPointID(-1),
mSelectedFeatureLevel(-1),
mMatrixColorSchemeName(name + ".Matrix Color"),
mMatrixColorScheme(COLOR_MAP_GYRD),
mTsneColorModeName(name + ".TSNE Color"),
mTsneColorMode(VisibilityGraphDataManager::tsne_dbscan_color),
mDisMethod(DISTANCE_EUCLIDEAN),
mDbscanEpsName(name + ".DBScan Epsilon"),
mDbscanEps(mData->mDBScan.getEps()),
mDbscanMinNumNeighborsName(name + ".DBScan Min Num"),
mSelectionKunhuaDisName(name + ".Combine Dis"),
mSelectionKunhuaDis(0),
mDbscanMinNumNeighbors(mData->mDBScan.getMinNumSamples()),
mTsneSelectionRadius(10.0f),
mTsneSelectionRadiusName(name + ".Selection Radius"),
mTsneSelectionDisName(name + ".Dis Methods"),
mLineRadiusName(name + ".Line Radius"),
mLineRadius(0.2),
kunhua_dis_mat(data->mPool.streamlines.size(), data->mPool.streamlines.size(), &(data->mKunhuaDisMats[0][0]))
//构造函数开始
{
	DataManager* manager = DataManager::sharedManager();
	manager->createFloat(mFilterLowName, mFilter.lower, 100.0f, 0.0f, this, true);
	manager->createFloat(mFilterUpName, mFilter.upper, 500.0f, 0.0f, this, true);
	manager->createFloat(mNormalizationLowName, mNormalization.lower, 100.0f, 0.0f, this, true);
	//manager->createFloat(mNormalizationUpName, mNormalization.upper, 300.0f, 5.0f, this, true);
	//manager->createFloat(mMatchDegreeDistThreshName, mMatchDegreeDistThresh, 1.0f, 0.0f, this, true);
	manager->createFloat(mDbscanEpsName, mDbscanEps, 100.0f, 0.0f, this, true);
	manager->createInt(mDbscanMinNumNeighborsName, mDbscanMinNumNeighbors, 20, 0, this, true);
	manager->createFloat(mTsneSelectionRadiusName, mTsneSelectionRadius, 5.0f, 50.0f, this, true);
	
	manager->createFloat(mLineRadiusName, mLineRadius, 2.0f, 0.0f, this, true);

	auto tmp = ColorMap::getLinearColorSchemeNames();
	std::vector<std::string>& color_names = tmp;
	manager->createEnum(mMatrixColorSchemeName, color_names, mMatrixColorScheme, DATA_ITEM_ENUM_COMBOBOX, this);

	std::vector<std::string> tsne_color_name = { "Dataset", "DBScan" };
	manager->createEnum(mTsneColorModeName, tsne_color_name, mTsneColorMode, DATA_ITEM_ENUM_COMBOBOX, this);
	std::vector<std::string> tsne_dis_name;
	int num = data->mKunhuaDisMats.size();
	if (data->mKunhuaDisMats.size() == 7) {
		tsne_dis_name = { "Euclidean", "Geometric" , "MCP" , "Hausdorff" , "Endpoints", "Procrustes" , "Frechet" };
	}
	else if (data->mKunhuaDisMats.size() == 11) {
		tsne_dis_name = { "Euclidean", "Geometric" , "MCP" , "Hausdorff" , "Procrustes" , "Frechet", "0.5E+0.5F" , "0.5P+0.5G" , "0.8E+0.2F" , "0.8P+0.2G" , "E+P+F" };
	}
	else if (data->mKunhuaDisMats.size() == 15) {
		tsne_dis_name = { "Euclidean", "Geometric" , "MCP" , "Hausdorff" , "Procrustes" , "Frechet"};
		manager->createInt(mSelectionKunhuaDisName, mSelectionKunhuaDis, 10, 0, this, true);
	}
	manager->createEnum(mTsneSelectionDisName, tsne_dis_name, mDisMethod, DATA_ITEM_ENUM_COMBOBOX, this);
	//manager->createFloat(dE, dEtag, 1.0f, 0.0f, this, true);
	//manager->createFloat(dG, dGtag, 1.0f, 0.0f, this, true);
	//manager->createFloat(dM, dMtag, 1.0f, 0.0f, this, true);
	//manager->createFloat(dH, dHtag, 1.0f, 0.0f, this, true);
	//manager->createFloat(dP, dPtag, 1.0f, 0.0f, this, true);
	//manager->createFloat(dF, dFtag, 1.0f, 0.0f, this, true);

	//mData->updateTsneColorId(mTsneColorMode);

	mDisplayMask = DisplayMaskType(display_sample_graphs | display_latent_vector | display_degree_vector);
	//mVGDisplay = createVGDisplay(0);
	mFilterArray = { 10.0f, 20.0f, 40.0f, 80.0f, 100.0f };
	mVGMultiDisplay = createVGMultiDisplay(0, mFilterArray);

	//为每个distance创建一个散点图
	for (int i = 0; i < mData->mKunhuaDisMats.size(); ++i) {
		MatrixData<float> *tmp = new MatrixData<float>(data->mPool.streamlines.size(), data->mPool.streamlines.size(), &(data->mKunhuaDisMats[i][0]));
		kunhua_dis_mat = *tmp;
		createTsneGraphDisplay();
	}

	mTsne = mTsnes[0];
	mData->updateTsneDisplayLayout(mTsne.getNodeDisplayPosition());
	mData->initDBScan(mTsne.getNodeDisplayPosition(), mDbscanEps, mDbscanMinNumNeighbors);

	updateTsneColor();






 }

void VisibilityGraphDisplayWidget::display()
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

	drawVGDisplay(mVGDisplay);
	drawVGMultiDisplay(mVGMultiDisplay);
	// 画Tsne点的位置。
	if (mVGDisplay.streamline_id<0) drawTsne();

	glPopAttrib();
}

VGDisplay VisibilityGraphDisplayWidget::createVGDisplay(const int& streamline_id) {
	VGDisplay ret;

	if (streamline_id < 0) return ret;

	MatrixData<float>* org_dist_mat = &(mData->getVisGraphOfStreamline(streamline_id)->adj_mat);
	MatrixData<float>* dist_mat = new MatrixData<float>(org_dist_mat->width(), org_dist_mat->height());
	filterMatrix(*dist_mat, *org_dist_mat);

	float margin = 0.05f;
	float display_frac = 1.0f - 2.0f*margin;
	RectDisplayArea area;
	area = makeRectDisplayArea(margin*mWidth, margin*mHeight, 0.5f*mWidth*display_frac, 0.0f, 0.0f, mHeight*display_frac);

	GraphDisplay<float>* ret_graph = new GraphDisplay<float>();
	ret_graph->setArea(area); 
	ret_graph->setEdgeFilter(mFilter.lower, mFilter.upper);
	ret_graph->setEdgeNormalization(mNormalization.upper, mNormalization.lower, 0.15f, 0.5f);
	ret_graph->setData(dist_mat);
	ret_graph->updateNodePosition(true);
	ret_graph->setRadius(6.0f);

	area.origin.x += 0.5f*mWidth;
	MatrixDisplay<float>* ret_map = new MatrixDisplay<float>();
	ret_map->setArea(area);
	ret_map->setData(dist_mat);
	ret_map->setFilter(mFilter.lower, mFilter.upper);
	ret_map->setNormalization(mNormalization.upper, mNormalization.lower, 0.2f, 1.0f);
	ret_map->setColorScheme(mMatrixColorScheme);
	if(dist_mat->width()>300) ret_map->setMargin(0.0f);

	ret.data = dist_mat;
	ret.graph_display = ret_graph;
	ret.map_display = ret_map;
	ret.streamline_id = streamline_id;

	return ret;
}

VGMultiDisplay VisibilityGraphDisplayWidget::createVGMultiDisplay(const int& streamline_id,
	const std::vector<float>& filters)
{
	VGMultiDisplay ret;
	ret.streamline_id = streamline_id;
	if (streamline_id < 0) return ret;

	// 读数据
	VisibilityGraph* vg = mData->getVisGraphOfStreamline(streamline_id);
	int n = vg->n;
	int nn = n*n;
	MatrixData<float>* org_dist_mat = &(vg->adj_mat);

	int num_graph_per_line = mData->getNumFeatures()+1;
	if (num_graph_per_line < 8) num_graph_per_line = 8;
	float margin = 0.025f; 
	float width = mWidth / (float)(num_graph_per_line+(num_graph_per_line-1)*margin);
	float height = width;
	

	RectDisplayArea area = makeRectDisplayArea(width*margin, height*margin, 
		(1.0f-2.0f*margin)*width, 0, 0, (1.0f - 2.0f*margin)*height);

	MatrixData<int>* group_mat = new MatrixData<int>(n, n, -1);
	int *group_mat_data = group_mat->getData();

	bool b_graph_init = false;
	for (int i = 0; i < 0 /*filters.size()*/; ++i) {
		//create filter
		Range filter = makeRange(0.0f, filters[i]);
		//filter data
		MatrixData<float>* mat = new MatrixData<float>(n, n);
		filterMatrix(*mat, *org_dist_mat, filter);
		float* mat_data = mat->getData();
		//update the group mat data
		for (int j = 0; j < nn; ++j) {
			if (inRange(filter, mat_data[j]) && group_mat_data[j] < 0) {
				group_mat_data[j] = i;
			}
		}
		//create graph display
		GraphDisplay<float>* gd = new GraphDisplay<float>();
		gd->setArea(area);
		gd->setEdgeFilter(filter.lower, filter.upper);
		gd->setEdgeNormalization(mNormalization.upper, mNormalization.lower, 0.15f, 0.5f);
		gd->setData(mat);
		if (b_graph_init) {
			const std::vector<vec2f>& prev_nodes = ret.graph_displays[i - 1]->getNodePositions();
			gd->setNodePositions(prev_nodes);
			gd->updateNodePosition(false);
		} else {
			gd->updateNodePosition(true);
			b_graph_init = true;
		}
		gd->updateNodeColor(ColorMap::getD3Color(i));
		gd->setRadius(width*0.02f);
		ret.graph_displays.push_back(gd);
		//ret.all_displays.push_back(gd);
		//update area location
		area.origin.x += width;
	}

	ret.map_display = new MatrixDisplay<float>();
	ret.map_display->setColorScheme(mMatrixColorScheme);
	ret.map_display->setArea(area);
	ret.map_display->setData(org_dist_mat);
	ret.map_display->setFilter(-1e30, 1e30);//show all distance
	ret.map_display->setNormalization(mNormalization.upper, mNormalization.lower, 0.2f, 1.0f);
	//ret.all_displays.push_back(ret.map_display);

	//area.origin.x += width;
	//ret.group_map_display = new MatrixDisplay<int>();
	//ret.group_map_display->setColorScheme(mMatrixColorScheme);
	//ret.group_map_display->setArea(area);
	//ret.group_map_display->setData(group_mat);
	//ret.group_map_display->setFilter(0, filters.size() - 1);
	//ret.all_displays.push_back(ret.group_map_display);

	area.origin.x += width;
	int sample_size = mData->getSampleSize();
	//for (int i = 0; i < mData->getNumFeatures() && (mDisplayMask&display_sample_graphs) && 0; ++i) {
	for (int i = 0; i < 0; ++i) {
		MatrixData<float>* submat = new MatrixData<float>(sample_size, sample_size);
		mData->getSampleMatrix(*submat, streamline_id, 0, i);
		MatrixDisplay<float>* mat_display = new MatrixDisplay<float>();
		mat_display->setColorScheme(mMatrixColorScheme);
		mat_display->setArea(area);
		mat_display->setData(submat);
		mat_display->setFilter(-1e30, 1e30);
		mat_display->setNormalization(16.0f, 0.0f, 0.2f, 1.0f);
		area.origin.x += width + margin*width;
		ret.sample_map_displays.push_back(mat_display);
		//ret.all_displays.push_back(mat_display);
	}

	float cell_width = (n > 200) ? (1.0f / n) : 0.005f;
	area.origin.x = margin*width;
	area.origin.y += height+margin*height;
	area.row_axis = makeVec2f(0.0f, height);
	//area.col_axis = makeVec2f((1.0f - margin)*mWidth*n*cell_width, 0.0f);
	area.col_axis = makeVec2f((1.0f - margin)*mWidth*n*cell_width, 0.0f);

	if (mDisplayMask & display_degree_vector & 0) {
		ret.degree_display = new LineChartMarginDisplay();
		ret.degree_display->setArea(area);
		updateDegreeMap(ret);
		ret.all_displays.push_back(ret.degree_display);
		area.origin.y += height + margin*height;
	}
	
	/*if (mDisplayMask & display_degree_vector & 0) {
		//area.origin.x += 0.5f*mWidth;
		ret.latent_display = new MatrixDisplay<float>();
		ret.latent_display->setArea(area);
		mData->updateLatentDisplayData(streamline_id);
		ret.latent_display->setData(&vg->latent_mat);
		Range latent_range = mData->getLatentFeatureRange();
		ret.latent_display->setFilter(-1e30, 1e30);
		ret.latent_display->setSelectType(MatrixDisplay<float>::Row);
		if (mData->mLatentDisplayMode == VisibilityGraphDataManager::latent_tsne_color) {
			ret.latent_display->setColorScheme(COLOR_MAP_D3);
			ret.latent_display->setNormalization(0.0f, 20.0f, 0.0f, 20.0f);
		} else {
			ret.latent_display->setNormalization(latent_range.lower, latent_range.upper, 0.0f, 1.0f);
		}
		ret.all_displays.push_back(ret.latent_display);
	}

	//VRComplex vrc(2, 1e29, org_dist_mat->getData(), n);
	//vrc.incrementalVR();
	//vrc.updateCreation();
	//vrc.initSortedSimplexArray(2);
	//vrc.computePersistencePairs();
	//std::vector<Range> barcode;
	//vrc.genPersistenceBarcode(barcode, 1.0f, 1);

	//width = height;
	//area.origin.x = 0.0f;
	//area.origin.y += height;
	//area.row_axis = makeVec2f(width, 0.0f);
	//area.col_axis = makeVec2f(0.0f, height);

	//ret.barcode_display = new BarDisplay();
	//ret.barcode_display->setArea(area);
	//ret.barcode_display->setRange(mNormalization.lower, mNormalization.upper);
	//ret.barcode_display->setData(barcode);
*/
	ret.size = makeVec2f(mWidth, area.origin.y + area.row_axis.y + area.col_axis.y);

	return ret;
}

VGMultiDisplay VisibilityGraphDisplayWidget::createVGMultiDisplay(const std::vector<int>& tsne_ids)
{
	VGMultiDisplay ret;
	ret.streamline_id = mData->getNumStreamlines();

	int num_rows = 3;
	int num_graph_per_line = 9;
	float margin = 0.025f;
	float width = mWidth / (float)(num_graph_per_line + (num_graph_per_line - 1)*margin);
	float height = width;

	RectDisplayArea area = makeRectDisplayArea(width*margin, height*margin,
		(1.0f - 2.0f*margin)*width, 0, 0, (1.0f - 2.0f*margin)*height);

	int sample_size = mData->getSampleSize();
	for (int i = 0; i < tsne_ids.size() && i <num_graph_per_line*num_rows; ++i) {
		MatrixData<float>* submat = new MatrixData<float>(sample_size, sample_size);
		mData->getTsneSampleMatrix(*submat, tsne_ids[i]);
		MatrixDisplay<float>* mat_display = new MatrixDisplay<float>();
		mat_display->setColorScheme(mMatrixColorScheme);
		mat_display->setArea(area);
		mat_display->setData(submat);
		mat_display->setFilter(-1e30, 1e30);
		mat_display->setNormalization(16.0f, 0.0f, 0.2f, 1.0f);
		area.origin.x += width + margin*width;
		ret.sample_map_displays.push_back(mat_display);
		//ret.all_displays.push_back(mat_display);
		if ((i%num_graph_per_line) == (num_graph_per_line - 1)) {
			area.origin.x = width*margin;
			area.origin.y += height+height*margin;
		}
	}

	return ret;
}

void VisibilityGraphDisplayWidget::updateDegreeMap(VGMultiDisplay& vg_multidisplay) {
	if (vg_multidisplay.streamline_id < 0) return;

	VisibilityGraph* vg = mData->getVisGraphOfStreamline(vg_multidisplay.streamline_id);
	int n = vg->n;
	std::vector < std::vector<vec2f>> degrees(n);
	float** degree_data = vg->degree_mat.getMatrixPointer();
	int num_threshes = mData->mDistThreshes.size();
	for (int i = 0; i < n; ++i) {
		std::vector<vec2f>& degree = degrees[i];
		degree.reserve(num_threshes);
		for (int j = 0; j < num_threshes; ++j) {
			degree.push_back(makeVec2f(mData->mDistThreshes[j], degree_data[i][j]));
			if (degree_data[i][j] == n) break;
		}
	}
	vg_multidisplay.degree_display->setData(degrees, false);
	vg_multidisplay.degree_display->setRange(mNormalization.lower, mNormalization.upper, 0, 100);
}

bool VisibilityGraphDisplayWidget::matchPointDegree(const int & point_id, const float& dist_thresh) {
	if (mVGMultiDisplay.streamline_id>=0 && point_id>=0)
	{
		int global_point_id = mData->getGlobalPointId(mVGMultiDisplay.streamline_id, point_id);
		Range y;
		if (mVGMultiDisplay.degree_display->isYRangeSelected()){
			y = mVGMultiDisplay.degree_display->getSelectedYRange();
		} else {
			y = mVGMultiDisplay.degree_display->getYRange();
		}
		mData->matchDegreePattern(mSelectedSegments, global_point_id, y, dist_thresh);
		StreamlineSegment query_seg = mData->neighborhood(mData->getPointInfo(global_point_id), y.upper);
		mData->updateRendererWithMatchResults(mSelectedSegments, query_seg);
		redraw();
		//MessageCenter::sharedCenter()->processMessage(SELECT_SEGMENT_CHANGE_MSG, getName());
		return true;
	}
	return false;
}

bool VisibilityGraphDisplayWidget::matchPointLatent(const int& point_id, const int& feature_level, 
	const float& dist_thresh) 
{
	if (mVGMultiDisplay.streamline_id >= 0 && point_id >= 0)
	{
		int global_point_id = mData->getGlobalPointId(mVGMultiDisplay.streamline_id, point_id);
		mData->matchLatentPattern(mSelectedSegments, global_point_id, feature_level, dist_thresh);
		redraw();
		return true;
	}
	return false;
}

void VisibilityGraphDisplayWidget::releaseVGDisplay(VGDisplay& vg_display) {
	if (vg_display.streamline_id < 0) return;
	delete vg_display.data;
	delete vg_display.graph_display;
	delete vg_display.map_display;
}

void VisibilityGraphDisplayWidget::releaseVGMultiDisplay(VGMultiDisplay& vg_multi_display)
{
	if (vg_multi_display.streamline_id < 0) return;
	
	vg_multi_display.all_displays.clear();
	for (int i = 0; i < vg_multi_display.graph_displays.size(); ++i) {
		GraphDisplay<float>* gd = vg_multi_display.graph_displays[i];
		gd->freeData();
		delete gd;
	}

	if (vg_multi_display.group_map_display != NULL) {
		vg_multi_display.group_map_display->freeData();
		delete vg_multi_display.group_map_display;
	}
	if (vg_multi_display.map_display != NULL)
		delete vg_multi_display.map_display;
	if (vg_multi_display.degree_display != NULL)
		delete vg_multi_display.degree_display;

	for (auto& m : vg_multi_display.sample_map_displays) {
		m->freeData();
		delete m;
	}
	//delete vg_multi_display.barcode_display;
}

void VisibilityGraphDisplayWidget::drawVGDisplay(const VGDisplay& vg_display) {
	if (vg_display.streamline_id < 0) return;

	glPushAttrib(GL_CURRENT_BIT | GL_LINE_BIT | GL_ENABLE_BIT);
	vg_display.map_display->display();
	glLineWidth(1.0f);
	vg_display.graph_display->drawEdges();
	glLineWidth(2.0f);
	int n = vg_display.data->width();
	vec4f color = ColorMap::getPerceptualColor(1.0f);
	for (int i = 1; i < n; ++i) {
		vg_display.graph_display->drawLine(i - 1, i, color);
	}
	vg_display.graph_display->drawNodes();
	glPopAttrib();
}

void VisibilityGraphDisplayWidget::drawVGMultiDisplay(const VGMultiDisplay& vg_multi_display){
	if (vg_multi_display.streamline_id < 0) return;
	glPushAttrib(GL_CURRENT_BIT | GL_LINE_BIT | GL_ENABLE_BIT);

	int gd_count = 0;
	for (auto d : vg_multi_display.all_displays) {
		//GraphDisplay<float>* gd = dynamic_cast<GraphDisplay<float>*>(d);
		//if (gd!=NULL) {
			// 这些好像是flow graph里面的东西，用来可视化流线的graph图 
			//glLineWidth(1.0f);
			//gd->drawEdges();
			//glLineWidth(2.0f);
			//int n = gd->getData()->width();
			//vec4f color = ColorMap::getD3ColorNoGray(gd_count++);
			//for (int j = 1; j < n; ++j) {
			//	gd->drawLine(j - 1, j, color);
			//}
			//gd->drawNodes();
		//} else {
			//d->display();
		//}
	}

	glPopAttrib();
}

void VisibilityGraphDisplayWidget::createTsneGraphDisplay(){

	//if (mData->mTsneLayout.empty()) return;

	MatrixData<float>* org_dist_mat = &(kunhua_dis_mat);
	//MatrixData<float>* dist_mat = new MatrixData<float>(org_dist_mat->width(), org_dist_mat->height());
	MatrixData<float>* dist_mat = org_dist_mat;
	//filterMatrix(*dist_mat, *org_dist_mat);
	//dist_mat->printMat();
	float margin = 0.05f;
	RectDisplayArea area;
	area.row_axis = makeVec2f((1.0f - 2.0f * margin) * mWidth, 0.0f);
	area.col_axis = makeVec2f(0.0f, (1.0f - 2.0f * margin) * (mHeight - mVGMultiDisplay.size.y));
	area.origin = makeVec2f(margin * mWidth, margin * area.col_axis.y + mVGMultiDisplay.size.y);

	mTsne.setArea(area);
	//mTsne.setEdgeFilter(mFilter.lower, mFilter.upper);
	mTsne.setEdgeNormalization(mNormalization.upper, mNormalization.lower, 0.15f, 0.5f);
	mTsne.setData(dist_mat);
	mTsne.updateNodePosition(true);
	mTsne.setRadius(6.0f);

	//mTsne.setArea(area);
	mTsne.setBorderSize(0.0f);
	mTsne.setBrushThresh(5.0f);
	mTsne.setSelectionRadius(10.0f);
	mData->updateTsneDisplayLayout(mTsne.getNodeDisplayPosition());

	bool b_success;
	DataManager* manager = DataManager::sharedManager();    
	mDbscanEps = manager->getFloatValue(mDbscanEpsName, b_success);
	mDbscanMinNumNeighbors = manager->getIntValue(mDbscanMinNumNeighborsName, b_success);
	mData->initDBScan(mTsne.getNodeDisplayPosition(), mDbscanEps, mDbscanMinNumNeighbors);
	mTsnes.push_back(mTsne);
	


	/*if (mData->mTsneLayout.empty()) return;

	float margin = 0.05f;
	RectDisplayArea area;
	area.row_axis = makeVec2f((1.0f - 2.0f*margin)*mWidth, 0.0f);
	area.col_axis = makeVec2f(0.0f, (1.0f - 2.0f*margin)*(mHeight - mVGMultiDisplay.size.y));
	area.origin = makeVec2f(margin*mWidth, margin*area.col_axis.y+ mVGMultiDisplay.size.y);
	mTsne.initDisplay(mData->mTsneLayout); 
	mTsne.setArea(area);
	mTsne.setRadius(6.0f);
	mTsne.setBorderSize(0.0f);
	mTsne.setBrushThresh(5.0f);
	mTsne.setSelectionRadius(10.0f);
	mData->updateTsneDisplayLayout(mTsne.getNodeDisplayPosition());

	updateTsneColor();*/
}

void VisibilityGraphDisplayWidget::updateTsneColor()
{
	mData->updateTsneColorId(mTsneColorMode);
	if (mVGMultiDisplay.streamline_id >= 0 && mVGMultiDisplay.streamline_id!=mData->getNumStreamlines()) {
		//mData->updateLatentDisplayData(mVGMultiDisplay.streamline_id);
	}
	mTsne.updateNodeColor(mData->mTsneColorIds, COLOR_MAP_D3);
	mTsne.updateNodeColorOpacity(0.6f);
}

// 
void VisibilityGraphDisplayWidget::drawTsne(){
	mTsne.display();

	RectDisplayArea area;
	area.origin = makeVec2f(10.0f, mHeight-50.0f);
	area.row_axis = makeVec2f(30.0f, 0.0f);
	area.col_axis = makeVec2f(0.0f, 30.0f);
	vec2f text_pos = area.origin + 1.25f * area.row_axis + 0.25f * area.col_axis;
	
	if (mTsneColorMode == VisibilityGraphDataManager::tsne_dataset_color) {
		std::vector<int> color_ids = { 1, 2, 4, 5, 6, 8, 9 };
		std::vector<std::string> data_set = { "5cp", "Combustion", "Computer Room", "Crayfish", 
			"Supercurrent", "Plume", "Two Swirls" };
		for (int i = 0; i < data_set.size(); ++i) {
			//drawRect(area, ColorMap::getD3Color(color_ids[i]), 0.0f);
			//drawText(text_pos, data_set[i], 12.0f, 1.0f);
			//vec2f text_size = getTextSize(data_set[i], 12.0f);
			//area.origin.x += 50.0f + 1.5f*text_size.x;
			//text_pos.x += 50.0f + 1.5f*text_size.x;
		}
	}

	// 上面这个是点中的节点给颜色的
	mTsne.drawNodes(mData->mTsneMatchIds, 8.0f, makeVec4f(mData->mMatchColor.xyz,0.6f), 3.0f);
	// 下面这个我不懂是啥,好像是带颜色节点外面一圈黑色的
	mTsne.drawNodes(mData->mTsneQueryIds, 8.0f, makeVec4f(mData->mQueryColor.xyz,1.0f), 3.0f);
}

void VisibilityGraphDisplayWidget::updateFilter(VGDisplay& vg_display){
	if (vg_display.streamline_id < 0) return;
	MatrixData<float>* org_dist_mat = &(mData->getVisGraphOfStreamline(vg_display.streamline_id)->adj_mat);
	filterMatrix(*vg_display.data, *org_dist_mat);
	vg_display.graph_display->updateNodePosition(false);
	vg_display.graph_display->setEdgeFilter(mFilter.lower, mFilter.upper);
	vg_display.map_display->setFilter(mFilter.lower, mFilter.upper);
	vg_display.graph_display->setEdgeNormalization(mNormalization.upper, mNormalization.lower, 0.15f, 0.5f);
	vg_display.map_display->setNormalization(mNormalization.upper, mNormalization.lower, 0.2f, 1.0f);
}

void VisibilityGraphDisplayWidget::filterMatrix(MatrixData<float>& dst, MatrixData<float>& org, const Range& filter){
	//filter
	int n = org.width();
	for (int i = 0; i < n; ++i) {
		dst[i][i] = org[i][i];
		bool is_valid = true;
		for (int j = i + 1; j < n; ++j) {
			if (!is_valid) {
				dst[j][i] = dst[i][j] = 1e30;
			} else if (!inRange(filter, org[i][j])) {
				dst[j][i] = dst[i][j] = 1e30;
				is_valid = false;
			} else {
				dst[j][i] = dst[i][j] = org[i][j];
			}
		}
	}
	//floyd-warshall
	float dik, dikj;
	for (int k = 0; k < n; ++k) {
		for (int i = 0; i < n; ++i) {
			dik = dst[i][k];
			for (int j = 0; j < i; ++j) {
				dikj = dik + dst[k][j];
				if (dst[i][j] > dikj) {
					dst[j][i] = dst[i][j] = dikj;
				}
			}
		}
	}
}

void VisibilityGraphDisplayWidget::filterMatrix(MatrixData<float>& dst, MatrixData<float>& org) {
	filterMatrix(dst, org, mFilter);
}

void VisibilityGraphDisplayWidget::setStreamline(const int& streamline_id){
	if (mVGDisplay.streamline_id >= 0) {
		releaseVGDisplay(mVGDisplay);
		mVGDisplay = createVGDisplay(streamline_id);
	}
	if (mVGMultiDisplay.streamline_id >= 0) {
		releaseVGMultiDisplay(mVGMultiDisplay);
		mVGMultiDisplay = createVGMultiDisplay(streamline_id, mFilterArray);
	}
	mData->updateDisplayWithTsneIds(std::vector<int>(streamline_id));
	mTsneSampleMatrixIds.clear();
	mSelectedSegments.clear();
	mData->mTsneMatchIds.clear();
	mData->mTsneQueryIds.clear();
}

void VisibilityGraphDisplayWidget::onDataItemChanged(const std::string& name){
	DataManager* manager = DataManager::sharedManager();
	bool b_success;

	if (name == mFilterLowName) {
		mFilter.lower = manager->getFloatValue(name, b_success);
		updateFilter(mVGDisplay);
		redraw();
	} 
	else if (name == mFilterUpName) {
		mFilter.upper = manager->getFloatValue(name, b_success);
		updateFilter(mVGDisplay);
		redraw();
	} 
	else if (name == mNormalizationLowName) {
		mNormalization.lower = manager->getFloatValue(name, b_success);
		updateFilter(mVGDisplay);
		mData->updateDistThreshes(mNormalization);
		updateDegreeMap(mVGMultiDisplay);
		redraw();
	} 
	else if (name == mNormalizationUpName) {
		mNormalization.upper = manager->getFloatValue(name, b_success);
		updateFilter(mVGDisplay);
		mData->updateDistThreshes(mNormalization);
		updateDegreeMap(mVGMultiDisplay);
		redraw();
	} 
	else if (name == SELECTED_LINE_ID_NAME) {
		MessageCenter::sharedCenter()->processMessage("Selected Streamline Change", getName());
	} 
	else if (name == mMatchDegreeDistThreshName) {
		mMatchDegreeDistThresh = manager->getFloatValue(name, b_success);
		if (mSelectedPointID < 0) return;
		if (mClickedDisplayPointer == mVGMultiDisplay.latent_display) {
			matchPointLatent(mSelectedPointID, mSelectedFeatureLevel, mMatchDegreeDistThresh);
		} else {
			matchPointDegree(mSelectedPointID, mMatchDegreeDistThresh);
		}
	} 
	else if (name == mMatrixColorSchemeName) {
		mMatrixColorScheme = (LinearColorMapType)manager->getEnumValue(name, b_success);
		for (auto d : mVGMultiDisplay.sample_map_displays) {
			d->setColorScheme(mMatrixColorScheme);
		}
		if (mVGMultiDisplay.map_display != NULL)
			mVGMultiDisplay.map_display->setColorScheme(mMatrixColorScheme);

	} 
	else if (name == mDbscanEpsName) {
		mDbscanEps = manager->getFloatValue(name, b_success);
		mDbscanMinNumNeighbors = manager->getIntValue(mDbscanMinNumNeighborsName, b_success);
		mData->mDBScan.fit(mDbscanEps, mDbscanMinNumNeighbors);
		updateTsneColor();
		MessageCenter::sharedCenter()->processMessage("Selected Point Change", getName());
		redraw();
	} 
	else if (name == mDbscanMinNumNeighborsName) {
		mDbscanEps = manager->getFloatValue(mDbscanEpsName, b_success);
		mDbscanMinNumNeighbors = manager->getIntValue(name, b_success);
		mData->mDBScan.fit(mDbscanEps, mDbscanMinNumNeighbors);
		updateTsneColor();
		MessageCenter::sharedCenter()->processMessage("Selected Point Change", getName());
		redraw();
	} 
	else if (name == mTsneColorModeName) {
		mTsneColorMode = (VisibilityGraphDataManager::TsneColorMode)manager->getEnumValue(name, b_success);
		updateTsneColor();
		MessageCenter::sharedCenter()->processMessage("Selected Point Change", getName());
		redraw();
	} 
	else if (name == mTsneSelectionRadiusName) {
		mTsneSelectionRadius = manager->getFloatValue(name, b_success);
		mTsne.setSelectionRadius(mTsneSelectionRadius);
	} 
	else if (name == mTsneSelectionDisName) {
		mDisMethod = (DistanceMethodType)manager->getEnumValue(name, b_success);
		mDbscanEps = manager->getFloatValue(mDbscanEpsName, b_success);
		mDbscanMinNumNeighbors = manager->getIntValue(mDbscanMinNumNeighborsName, b_success);
		mTsne = mTsnes[mDisMethod];
		mData->initDBScan(mTsnes[mDisMethod].getNodeDisplayPosition(), mDbscanEps, mDbscanMinNumNeighbors);
		updateTsneColor();
		mData->updateTsneDisplayLayout(mTsne.getNodeDisplayPosition());
		MessageCenter::sharedCenter()->processMessage("Selected Point Change", getName());
		redraw();
	} 
	else if (name == mLineRadiusName) {
		mData->mRenderer->updateTubeRadius(manager->getFloatValue(name, b_success));
		MessageCenter::sharedCenter()->processMessage("Line Radius Change", getName());
		redraw();
	} 
	else if (name == mSelectionKunhuaDisName) {
		mDisMethod = (DistanceMethodType)manager->getIntValue(name, b_success);
		mDbscanEps = manager->getFloatValue(mDbscanEpsName, b_success);
		mDbscanMinNumNeighbors = manager->getIntValue(mDbscanMinNumNeighborsName, b_success);
		if (mDisMethod == 0) {
			mTsne = mTsnes[4];
			mData->initDBScan(mTsnes[4].getNodeDisplayPosition(), mDbscanEps, mDbscanMinNumNeighbors);
		}
		else if (mDisMethod == 10) {
			mTsne = mTsnes[0];
			mData->initDBScan(mTsnes[0].getNodeDisplayPosition(), mDbscanEps, mDbscanMinNumNeighbors);
		}
		else {
			mTsne = mTsnes[mDisMethod + 5];
			mData->initDBScan(mTsnes[mDisMethod + 5].getNodeDisplayPosition(), mDbscanEps, mDbscanMinNumNeighbors);
		}
		
		updateTsneColor();
		mData->updateTsneDisplayLayout(mTsne.getNodeDisplayPosition());
		MessageCenter::sharedCenter()->processMessage("Selected Point Change", getName());
		
		redraw();
	}
}

//QWidget* VisibilityGraphDisplayWidget::createPanel(QWidget* parent){
//	std::vector<std::string> pars;
//	//pars.push_back(mFilterLowName);
//	//pars.push_back(mFilterUpName);
//	//pars.push_back(mNormalizationLowName);
//	//pars.push_back(mNormalizationUpName);
//	pars.push_back(SELECTED_LINE_ID_NAME);
//	//pars.push_back(SELECTED_LINE_ID_NAME);
//	//pars.push_back(mMatchDegreeDistThreshName);
//	//pars.push_back(mMatrixColorSchemeName);
//	//pars.push_back(mTsneColorModeName);
//	pars.push_back(mTsneSelectionRadiusName); 
//	pars.push_back(mDbscanEpsName);
//	pars.push_back(mDbscanMinNumNeighborsName);
//	pars.push_back(mTsneSelectionDisName);
//	pars.push_back(mLineRadiusName);
//	if (mData->mKunhuaDisMats.size() == 15) {
//		pars.push_back(mSelectionKunhuaDisName);
//	}
//	//pars.push_back(dE);
//	//pars.push_back(dG);
//	//pars.push_back(dM);
//	//pars.push_back(dH);
//	//pars.push_back(dP);
//	//pars.push_back(dF);
//	QWidget* ret = DataManager::sharedManager()->createInterface("Parameters", pars, parent);
//	return ret;
//}

int VisibilityGraphDisplayWidget::getSelectedPoint(const vec2f& p){
	int selected_id;
	if (mTsne.inDisplayArea(p)){
		mTsne.setSelectionAnchor(p);
		mClickedDisplayPointer = &mTsne;
	} else if (mVGMultiDisplay.streamline_id >= 0) {
		for ( auto d : mVGMultiDisplay.all_displays) {
			selected_id = d->getClickElement(p);
			if (selected_id >= 0) {
				mClickedDisplayPointer = d;
				if (mClickedDisplayPointer == mVGMultiDisplay.latent_display) {
					mSelectedFeatureLevel = 
						mVGMultiDisplay.latent_display->getCellIdx(p).x / mData->getDisplayLatentFeatureDim();
					mSelectedFeatureLevel = clamp(mSelectedFeatureLevel, 0, mData->getNumFeatures()-1);
				} else {
					for (int i = 0; i < mVGMultiDisplay.sample_map_displays.size(); ++i) {
						if (mClickedDisplayPointer == mVGMultiDisplay.sample_map_displays[i]) {
							mClickedDisplayPointer = mVGMultiDisplay.latent_display;
							selected_id = (mSelectedPointID<0)?0:mSelectedPointID;
							mSelectedFeatureLevel = i;
						}
					}
				}
				return selected_id;
			}
		}
	} else if (mVGDisplay.streamline_id >= 0) {
		selected_id = mVGDisplay.graph_display->getNodeWithPos(p, 10.0f);
		if (selected_id >= 0) {
			mClickedDisplayPointer = mVGDisplay.graph_display;
			return selected_id;
		}

		selected_id = mVGDisplay.map_display->getCellIdx(p).x;
		if (selected_id >= 0) {
			mClickedDisplayPointer = mVGDisplay.map_display;
			return selected_id;
		}
	}
	
	return -1;
}

int VisibilityGraphDisplayWidget::getClickSampleTsneId(const vec2f &p){
	for (int i = 0; i < mVGMultiDisplay.sample_map_displays.size(); ++i) {
		auto& d = *mVGMultiDisplay.sample_map_displays[i];
		if (d.inDisplayArea(p)) {
			return mTsneSampleMatrixIds[i];
		}
	}
	return -1;
}

void VisibilityGraphDisplayWidget::setPoint(const int& pid) {
	vec4f highlight_color = ColorMap::getColorByName(ColorMap::Harvard_crimson);
	if (mVGMultiDisplay.streamline_id >= 0) {
		if (mVGMultiDisplay.degree_display != NULL) {
			mVGMultiDisplay.degree_display->selectLine(pid);
		}
		mVGMultiDisplay.map_display->selectColumnWithId(pid, highlight_color);

		for (auto& g : mVGMultiDisplay.graph_displays) {
			g->setSelectedNode(pid);
		}

		int sample_size = mData->getSampleSize();
		for (int i = 0; pid>=0 && i < mVGMultiDisplay.sample_map_displays.size(); ++i) {
			MatrixData<float>* submat = new MatrixData<float>(sample_size, sample_size);
			mData->getSampleMatrix(*submat, mVGMultiDisplay.streamline_id, pid, i);
			mVGMultiDisplay.sample_map_displays[i]->freeData();
			mVGMultiDisplay.sample_map_displays[i]->setData(submat);
			if (i == mSelectedFeatureLevel) 
				mVGMultiDisplay.sample_map_displays[i]->selectMatrix();
		}

		if (mVGMultiDisplay.latent_display != NULL) {
			if (mClickedDisplayPointer == mVGMultiDisplay.latent_display) {
				int d = mData->getDisplayLatentFeatureDim();
				int s = mSelectedFeatureLevel*d;
				for (int i = s; i < s + d; ++i) {
					mVGMultiDisplay.latent_display->selectCellWithId(makeVec2i(i, pid), highlight_color);
				}
			} else {
				mVGMultiDisplay.latent_display->selectRowWithId(pid, highlight_color);
			}
		}
	} else if (mVGDisplay.streamline_id >= 0) {
		mVGDisplay.map_display->selectColumnWithId(pid, makeVec4f(0.0f, 0.0f, 0.0f, 1.0f));
		mVGDisplay.graph_display->setSelectedNode(pid);
	}
}

void VisibilityGraphDisplayWidget::clearPointSelection() {
	if (mVGMultiDisplay.streamline_id >= 0) {
		for (auto& d : mVGMultiDisplay.all_displays) {
			d->clearSelection();
		}
	} else if (mVGDisplay.streamline_id >= 0) {
		mVGDisplay.map_display->clearSelection();
		mVGDisplay.graph_display->clearSelection();
	}
	mData->mTsneMatchIds.clear();
	mData->mTsneQueryIds.clear();
}

// 鼠标点击之后会得到点击的位置，从而得到点击的坐标点
void VisibilityGraphDisplayWidget::mousePressEvent(const MouseEvent& e){
	if (e.button == MOUSE_LEFT_BUTTON) {
		getSelectedPoint(e.pos);
	} 
	DisplayWidget::mousePressEvent(e);
	redraw();
}

// 鼠标滑动之后，会判断上一次的事件是否为点击，是的话就干活
void VisibilityGraphDisplayWidget::mouseMoveEvent(const MouseEvent& e){
	MouseEvent pe = getPreviousMouseEvent();
	if (pe.button==MOUSE_LEFT_BUTTON) {
		if (mTsne.inSelection()){
			mTsne.updateSelectionBox(e.pos);	// 选中一个框的时候就运行这个，然后会选中一大堆点。不过这只是一个判断条件，真正执行是下面的mouseReleaseEvent
		}else if (mClickedDisplayPointer == mVGMultiDisplay.degree_display && length(e.pos- pe.pos)>2.0f) {
			// 否则就是点中这个地方，然后半径内的区域被选中，画图。
			Range y;
			y.lower = mVGMultiDisplay.degree_display->getYValueWithPos(e.pos);
			y.upper = mVGMultiDisplay.degree_display->getYValueWithPos(getFirstMouseEvent().pos);
			mVGMultiDisplay.degree_display->selectYRange(y);
		}
		redraw();
	}
	DisplayWidget::mouseMoveEvent(e);
}

// 当鼠标是选中一个框的时候，就用这个。会选中框内的点。
void VisibilityGraphDisplayWidget::mouseReleaseEvent(const MouseEvent& e) {
	if (getPreviousMouseEvent().button == MOUSE_LEFT_BUTTON) {
		if (mTsne.inSelection()){
			mtsne_ids.clear();
			// 这里是用cuda计算出，点击的位置附近都有什么tsne点。
			if (mTsne.getSelectionBoxSize() < 5.0f) {
				mData->findTsneIdsInDisplay(mtsne_ids, e.pos, mTsneSelectionRadius);
				mTsne.endSelection();
			} else {
				mTsne.updateSelectionBox(e.pos);
				mTsne.finishSelection(mtsne_ids);
			}
			const std::vector<int>& dbscan_label = mData->mDBScan.getLabels();
			for (int i = 0; i < mtsne_ids.size(); ++i) {
				printf("%d ", dbscan_label[mtsne_ids[i]]);
			}
			printf("\n");
			MessageCenter::sharedCenter()->processMessage("Selected Point Change", getName());
			mData->updateDisplayWithTsneIds(mtsne_ids);
			if (!mtsne_ids.empty()) {
				//mTsneSampleMatrixIds.assign(mtsne_ids.begin(), mtsne_ids.end());
				//releaseVGMultiDisplay(mVGMultiDisplay);
				//mVGMultiDisplay = createVGMultiDisplay(mTsneSampleMatrixIds);
			}
		} else if (!mTsneSampleMatrixIds.empty()) {
			int click_tsne_id = getClickSampleTsneId(e.pos);
			if (click_tsne_id >= 0) {

				mData->updateDisplayWithTsneIds(std::vector<int>({ click_tsne_id }));
				for (int i = 0; i < mVGMultiDisplay.sample_map_displays.size(); ++i) {
					if (mTsneSampleMatrixIds[i] == click_tsne_id) {
						mVGMultiDisplay.sample_map_displays[i]->selectMatrix();
					} else {
						mVGMultiDisplay.sample_map_displays[i]->clearSelection();
					}
				}
				redraw();
			}
		} else if (mClickedDisplayPointer == mVGMultiDisplay.degree_display &&
			length(getFirstMouseEvent().pos - e.pos) > 5.0f) //drag
		{
			Range y;
			y.lower = mVGMultiDisplay.degree_display->getYValueWithPos(e.pos);
			y.upper = mVGMultiDisplay.degree_display->getYValueWithPos(getFirstMouseEvent().pos);
			mVGMultiDisplay.degree_display->selectYRange(y);
		} else {
			mSelectedPointID = getSelectedPoint(e.pos);
			clearPointSelection();
			setPoint(mSelectedPointID);
			if (mClickedDisplayPointer == mVGMultiDisplay.latent_display) {
				if (matchPointLatent(mSelectedPointID, mSelectedFeatureLevel, mMatchDegreeDistThresh)) {
					redraw();
				}
			} else {
				if (matchPointDegree(mSelectedPointID, mMatchDegreeDistThresh)) {
					redraw();
				}
			}
		}
	}
	DisplayWidget::mouseReleaseEvent(e);
}