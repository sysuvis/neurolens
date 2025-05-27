#pragma once
#ifndef RETRIEVEMODULE_H
#define RETRIEVEMODULE_H
//#include"Communication.h"
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

class RetriveModule : public DisplayWidget, public DataUser
{
public:
	RetriveModule(int x, int y, int w, int h, std::string name);

	

};

#endif // RETRIEVEMODULE_H