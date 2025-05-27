#include "RetriveModule.h"
#include "EmbeddingWidget.h"
#include "VisibilityGraphMainWindow.h"
#include "BrainFiberData.h"
#include "DataManager.h"
#include "ViewportWidget.h"
#include "definition.h"
#include "RenderWidget.h"
#include <QHBoxLayout>
#include <QFileDialog>
#include "GlobalDataManager.h"
#include "WindowsTimer.h"
#include <string>
#include <QWebSocket>
#include "QueryWidget.h"


#define SINGLE_WIN_WIDTH	1000
#define SINGLE_WIN_HEIGHT	1200
#define PANEL_WIDTH		120
#define NUM_WIN_ROW		1
#define NUM_WIN_COL		2
#define GROUP_BOX_MARGIN 20

MainWindow::MainWindow(QWidget *parent, Qt::WindowFlags flags)
:QMainWindow(parent, flags)

{	
	this->setMinimumSize(SINGLE_WIN_WIDTH*NUM_WIN_COL+ PANEL_WIDTH+200, SINGLE_WIN_HEIGHT*NUM_WIN_ROW + 20);
	this->setGeometry(50, 50, SINGLE_WIN_WIDTH*NUM_WIN_COL+ PANEL_WIDTH, SINGLE_WIN_HEIGHT*NUM_WIN_ROW+20);
	ui.setupUi(this);

	mViewport = new ViewportWidget;
	mViewport->setGeometry(PANEL_WIDTH, 0, NUM_WIN_COL*SINGLE_WIN_WIDTH, NUM_WIN_ROW*SINGLE_WIN_HEIGHT);
	DataManager::sharedManager()->createPointer("View Port.Pointer",reinterpret_cast<PointerType>(mViewport));

	mLayout = new QHBoxLayout;
	mLayout->addWidget(mViewport, mViewport->width());
	ui.centralWidget->setLayout(mLayout);

	//start produce_all_samples
	connect(mViewport, SIGNAL(finishInit()), this, SLOT(start()));
}

MainWindow::~MainWindow(){}
                           
void MainWindow::connected() {
	printf("socket connected\n");
}

void MainWindow::send_message_to_server()
{
	mClient->sendTextMessage("message");
}

void MainWindow::start() {

	//create data instance

	GlobalDataManager* globalData = new GlobalDataManager("Global Data");
	BrainDataManager * brainData = new BrainDataManager();
	QueryWidget* query_widget = new QueryWidget("Query Window");

	//create display widgets
	mEmbeddingWidget = new EmbeddingWidget(0, 0, SINGLE_WIN_WIDTH, SINGLE_WIN_HEIGHT, "Graph Display", brainData, globalData);
	mViewport->addChild(mEmbeddingWidget, mEmbeddingWidget->getName());

	mRenderWidget = new RenderWidget(SINGLE_WIN_WIDTH, 0, SINGLE_WIN_WIDTH, SINGLE_WIN_HEIGHT, "Render Widget", brainData);
	mViewport->addChild(mRenderWidget, mRenderWidget->getName());


	/*mSocket = new QWebSocket();
	mSocket->open(QUrl("localhost:5080"));
	connect(mSocket, &QWebSocket::connected, this, &MainWindow::connected);
	mSocket->sendTextMessage("anything");*/


	//create global variables
	DataManager* manager = DataManager::sharedManager();
	manager->createInt(SELECTED_LINE_ID_NAME, 0, brainData->getNumStreamlines() - 1, 0, globalData, true); //slider-> 全部流线都可以选择到：brainData->getNumStreamlines() - 1
	

	//create pointers for interaction
	manager->createPointer("Brain Data.Pointer", (PointerType)brainData);
	manager->createPointer("Global Data.Pointer", (PointerType)globalData);
	manager->createPointer("Query Window.Pointer", (PointerType)query_widget);

	//create control panels
	QWidget* panel_widget = new QWidget;

	/*mPanel = globalData->createPanel(panel_widget);
	mPanel->move(0, 0);
	mPanel->show();
	
	mPanel4embedding = mEmbeddingWidget->createPanel(panel_widget);
	mPanel4embedding->move(0, mPanel->height()+40);
	mPanel4embedding->show();

	mLLM_panel = query_widget->createPanel(panel_widget);
	mLLM_panel->move(0, mPanel->height()+mPanel4embedding->height() + 80);
	mLLM_panel->show();*/


	mPanel4embedding = mEmbeddingWidget->createPanel(panel_widget);
	mPanel4embedding->move(0, 0);
	mPanel4embedding->show();

	mLLM_panel = query_widget->createPanel(panel_widget);
	mLLM_panel->move(0,  mPanel4embedding->height() + 80);
	mLLM_panel->show();

	mLayout->insertWidget(0, panel_widget, mLLM_panel->width());

	/*QUrl url(QStringLiteral("ws://localhost:5080"));
	mClient = new QWebSocket();
	mClient->open(url);
	connect(mClient, &QWebSocket::connected, this, &MainWindow::connected);*/
}



