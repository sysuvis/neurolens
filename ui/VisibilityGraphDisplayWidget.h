#pragma once

#include "DisplayWidget.h"
#include "typeOperation.h"
#include "DataUser.h"
#include "VisibilityGraph.h"
#include "MatrixDisplay.h"
#include "GraphDisplay.h"
#include "LineChartDisplay.h"
#include "BarDisplay.h"

typedef struct _VGDisplay {
	_VGDisplay() {
		streamline_id = -1;
		map_display = NULL;
		graph_display = NULL;
		data = NULL;
	}
	int streamline_id;
	MatrixDisplay<float>* map_display;
	GraphDisplay<float>* graph_display;
	MatrixData<float>* data;
} VGDisplay;

typedef struct _VGMultiDisplay {
	_VGMultiDisplay() {
		streamline_id = -1;
		num_groups = 0;
		map_display = NULL;
		group_map_display = NULL;
		degree_display = NULL;
		latent_display = NULL;
	}

	int streamline_id;
	int num_groups;
	vec2f size;
	MatrixDisplay<float>* latent_display;
	MatrixDisplay<float>* map_display;
	MatrixDisplay<int>* group_map_display;
	std::vector<MatrixDisplay<float>*> sample_map_displays;
	std::vector<GraphDisplay<float>*> graph_displays;
	LineChartMarginDisplay* degree_display;
	std::vector<DisplayBase*> all_displays;
	//BarDisplay* barcode_display;
} VGMultiDisplay;


typedef enum {
	DISTANCE_EUCLIDEAN = 0,
	DISTANCE_GEOMETRIC = 1,
	DISTANCE_MPC = 2,
	DISTANCE_HAUSDORFF =3 ,
	DISTANCE_PROCRUSTES =4 ,
	DISTANCE_FRECHET = 5
} DistanceMethodType;

class VisibilityGraphDisplayWidget : public DisplayWidget, public DataUser {
public:
	enum DisplayMaskType {
		display_sample_graphs = 1,
		display_degree_vector = 2,
		display_latent_vector = 3
	};

	

	VisibilityGraphDisplayWidget(int x, int y, int w, int h, std::string name,
		VisibilityGraphDataManager* data);

	~VisibilityGraphDisplayWidget() {}

	void init(){}
	void display();

	//QWidget* createPanel(QWidget* parent);

	void menuCallback(const std::string& message) {}
	void onDataItemChanged(const std::string& name);

	void setStreamline(const int& streamline_id);
	void setPoint(const int& point_id);
	void clearPointSelection();

	const std::vector<StreamlineSegment>& getSelectedSegments() { return mSelectedSegments; }

	void mousePressEvent(const MouseEvent& e);
	void mouseMoveEvent(const MouseEvent& e);
	void mouseReleaseEvent(const MouseEvent& e);

	std::vector<int> mtsne_ids;

private:
	std::vector<MatrixData<float>> kunhua_dis_mats;
	MatrixData<float> kunhua_dis_mat;
	VGDisplay createVGDisplay(const int& streamline_id);
	void releaseVGDisplay(VGDisplay& vg_display);
	void drawVGDisplay(const VGDisplay& vg_display);

	VGMultiDisplay createVGMultiDisplay(const int& streamline_id, const std::vector<float>& filters);
	VGMultiDisplay createVGMultiDisplay(const std::vector<int>& tsne_ids);
	void releaseVGMultiDisplay(VGMultiDisplay& vg_multi_display);
	void drawVGMultiDisplay(const VGMultiDisplay& vg_multi_display);

	void createTsneGraphDisplay();
	void updateTsneColor();
	void drawTsne();

	void updateFilter(VGDisplay& vg_display);
	void filterMatrix(MatrixData<float>& dst, MatrixData<float>& org, const Range& filter);
	void filterMatrix(MatrixData<float>& dst, MatrixData<float>& org);

	void updateDegreeMap(VGMultiDisplay& vg_multidisplay);

	bool matchPointDegree(const int& point_id, const float& dist_thresh);
	bool matchPointLatent(const int& point_id, const int& feature_level, const float& dist_thresh);

	int getSelectedPoint(const vec2f& p);
	int getClickSampleTsneId(const vec2f &p);

	Range mFilter;
	std::vector<float> mFilterArray;
	Range mNormalization;
	
	VisibilityGraphDataManager* mData;
	VGDisplay mVGDisplay;
	VGMultiDisplay mVGMultiDisplay;
	std::vector<GraphDisplay<float>> mTsnes;
	GraphDisplay<float> mTsne;

	//for interaction
	void* mClickedDisplayPointer;
	std::vector<StreamlineSegment> mSelectedSegments;
	int mSelectedPointID;
	int mSelectedFeatureLevel;
	std::vector<int> mTsneSampleMatrixIds;

	//parameters
	std::string mFilterLowName;
	std::string mFilterUpName;
	std::string mNormalizationLowName;
	std::string mNormalizationUpName;
	float mMatchDegreeDistThresh;
	std::string mMatchDegreeDistThreshName;
	std::string mMatrixColorSchemeName;
	LinearColorMapType mMatrixColorScheme;
	DistanceMethodType mDisMethod;
	std::string mTsneColorModeName;
	VisibilityGraphDataManager::TsneColorMode mTsneColorMode;
	std::string mDbscanEpsName;
	float mDbscanEps;
	std::string mDbscanMinNumNeighborsName;
	int mDbscanMinNumNeighbors;
	std::string mSelectionKunhuaDisName;
	int mSelectionKunhuaDis;
	DisplayMaskType mDisplayMask;
	float mTsneSelectionRadius;
	std::string mTsneSelectionRadiusName;
	std::string mTsneSelectionDisName;
	std::string mLineRadiusName;
	float mLineRadius;

	std::string dE;
	float dEtag;
	std::string dG;
	float dGtag;
	std::string dM;
	float dMtag;
	std::string dH;
	float dHtag;
	std::string dP;
	float dPtag;
	std::string dF;
	float dFtag;
};