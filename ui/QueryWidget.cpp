#include "QueryWidget.h"

QueryWidget::QueryWidget(std::string name) :
	DataUser(),
    mButton_submit(name+". Query"),
    tokens_set(name+". Max_tokens"),
    query_status(name+". With RAG"),
    temperature(name+". Temperature"),
    topk(name+". Top_K")
{
	DataManager* manager = DataManager::sharedManager();

    manager->createTrigger(mButton_submit, this);
    manager->createInt(tokens_set, tokens_current, 2048, 100, this);
    manager->createBool(query_status,query_with_rag, this);
    manager->createFloat(temperature, temperature_current, 1, 0, this,true);
    manager->createInt(topk, topk_current, 5, 1, this,true);
    
    //set_layout(2500, 300, 1200, 800);

    QUrl url(QStringLiteral("ws://localhost:5088"));
    mClient = new QWebSocket();
    //QWebSocket::connect(mClient, &QWebSocket::connected, this, &QueryWidget::onConnected);
    QWebSocket::connect(mClient, &QWebSocket::textMessageReceived, [this](const QString& message) {
        received_info = message;
        analysis_message(received_info);
        set_figure(figure_id);
        set_caption(caption);
        set_generation(generation);
        set_papers(papers);

        });
    mClient->open(url);
    
    /*QWidget::connect(textEdit, &QLineEdit::returnPressed, [=]() { 
        QString text = textEdit->text();
        send_message_to_server(text);
    });*/

}

void QueryWidget::set_layout(int x, int y, int w, int h) {

    std::vector<std::string> pars;
    DataManager* manager = DataManager::sharedManager();
    query_window = manager->createInterface("Literature Query View", pars, NULL);
    query_window->setGeometry(x, y, w, h);
    query_window->setStyleSheet("background-color: white;");
    

    //set layout
    QVBoxLayout* MainLayout = new QVBoxLayout(query_window);
    QHBoxLayout* layout_up = new QHBoxLayout();
    QVBoxLayout* layout_down = new QVBoxLayout();
    QVBoxLayout* layout_left = new QVBoxLayout();
    QVBoxLayout* layout_right = new QVBoxLayout();

    QWidget* input_widget = new QWidget;
    input_widget->setLayout(layout_down);
    input_widget->setObjectName("inputwidget");
    input_widget->setStyleSheet("#inputwidget {"
        "border: 3px solid rgb(150,150,150);"
        "border-radius: 20px;"
        "padding: 0px;"
        "}");

    QWidget* chat_widget = new QWidget;
    chat_widget->setLayout(layout_left);
    chat_widget->setObjectName("chatwidget");
    chat_widget->setStyleSheet("#chatwidget {"
        "border: 3px solid rgb(189,189,189);"
        "border-radius: 10px;"
        "padding: 0px;"
        "}");

    QWidget* figure_widget = new QWidget;
    figure_widget->setLayout(layout_right);
    figure_widget->setObjectName("figureWidget");
    figure_widget->setStyleSheet("#figureWidget {"
        "border: 3px solid rgb(189,189,189);"
        "border-radius: 10px;"
        "padding: 0px;"
        "}");
    

    //set font
    QFont font_title;
    font_title.setFamily("Helvetica"); // 设置字体类型
    font_title.setPointSize(8);   // 设置字体大小
    font_title.setBold(true);

    QFont font;
    font.setFamily("Helvetica"); // 设置字体类型
    font.setPointSize(10);   // 设置字体大小

    QFont font_display;
    font_display.setFamily("Helvetica"); // 设置字体类型
    font_display.setPointSize(13);   // 设置字体大小
    //font.setBold(true);      // 设置加粗

    QFont font_input;
    font_input.setFamily("Helvetica"); // 设置字体类型
    font_input.setPointSize(14);   // 设置字体大小
    //font_input.setBold(true);      // 设置加粗

    text_icon = new QLabel("");

    //title 
    text_icon = new QLabel();
    text_icon->setFixedSize(700, 100);
    layout_left->addWidget(text_icon, 0, Qt::AlignCenter);
    text_icon->setStyleSheet("QLabel {"
        "border-image: url('D:/DATA/brain/ICON/title.png');"
        "color: rgb(82,82,82);"
        "}");
    text_icon->setText("Literature Query");
    text_icon->setAlignment(Qt::AlignRight);
    text_icon->setWordWrap(true);
    text_icon->setFont(font_title);

    //display text box
    text_display = new QLabel("");
    text_display->setFixedSize(700, 750);
    text_display->setWordWrap(true);
    text_display->setFont(font_display);
    
    /*text_display->setStyleSheet("QLabel {"
        "border-image: url('D:/DATA/brain/brain_dark.png') ;"
        "}");*/

    layout_left->addWidget(text_display);

    //input box
    textEdit = new QLineEdit();
    textEdit->setFixedSize(1300, 70);
    textEdit->setFont(font_input);
    textEdit->setStyleSheet("QLineEdit {"

        "border-image: url('D:/DATA/brain/ICON/enter.png');"
        "}");
    layout_down->addWidget(textEdit);


    //figure display
    imageLabel = new QLabel();
    imageLabel->setFixedSize(500, 380);
    layout_right->addWidget(imageLabel,0, Qt::AlignCenter);
    //set_figure("563.1");
    imageLabel->setScaledContents(true); // 设置图片自适应QLabel大小
    

    //caption display
    caption_display = new QLabel("");
    //caption_display->setStyleSheet("QLabel { border: 2px solid black; }");
    caption_display->setFixedSize(500,470);
    //caption_display->setGraphicsEffect(effect2);
    caption_display->setWordWrap(true);
    caption_display->setFont(font);
    caption_display->setStyleSheet("QLabel {"
        "padding: 2px;"
        "background-color : white;"
        "}");
    layout_right->addWidget(caption_display, 0, Qt::AlignCenter);

   
  
    layout_up->addWidget(chat_widget);
    layout_up->addWidget(figure_widget);
    
    MainLayout->addLayout(layout_up);
    MainLayout->addWidget(input_widget);
    //MainLayout->addLayout(layout_down);
}

void QueryWidget::set_figure(QString id) {
    QString path = "D:/DATA/brain/FIGURE/" + id + ".png";
    QPixmap pixmap(path);
    imageLabel->setPixmap(pixmap);
}

void QueryWidget::set_caption(QString content) {
    caption_display->setText(content);
}

void QueryWidget::set_generation(QString content) {
    text_display->setText(content);
}

void QueryWidget::set_papers(QString content) {
    text_icon->setText(content);
}

void QueryWidget::set_query_status(bool flag) { query_with_rag = flag; }

QString QueryWidget::get_query_content() { return textEdit->text(); }

//utils
void QueryWidget::analysis_message(QString text_data) {
    QStringList list = text_data.split("[SEP]");
    //generation = list[0].replace("\n","") + "\n" + "\n" + "From papers:\n" + list[1] + "\n" + list[2];
    generation = list[0].replace("\n", "");
    figure_id = list[3];
    QString caption_path = "D:/DATA/brain/CAPTION/captions/" + figure_id + ".txt";
    caption = readTextFile(caption_path);
    papers= QString("References:\n[1] ") + list[1] + QString("\n[2] ") + list[2];

    
}

QString QueryWidget::readTextFile(const QString& filePath) {
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qDebug() << "无法打开文件：" << filePath;
        return QString(); // 返回一个空的QString对象
    }

    QTextStream in(&file);
    QString content = in.readAll();
    file.close();

    return content;
}

//for Client
void QueryWidget::send_message_to_server(QString qtext_data)
{
    mClient->sendTextMessage(qtext_data);
}

//intercation

QWidget* QueryWidget::createPanel(QWidget* parent) {
	std::vector<std::string> pars;
    pars.push_back(query_status);
    pars.push_back(tokens_set);
    pars.push_back(temperature);
    pars.push_back(topk);
    pars.push_back(mButton_submit);
	QWidget* ret = DataManager::sharedManager()->createInterface("Literature Query", pars, parent);
	return ret;
}

void QueryWidget::onDataItemChanged(const std::string& name) {
    DataManager* manager = DataManager::sharedManager();
    if (name == mButton_submit) {
        MessageCenter::sharedCenter()->processMessage("Submit", "Literature Query View");
    }
    else if (name == query_status) {
        MessageCenter::sharedCenter()->processMessage("Query Status Changed", "Literature Query View");
    }
}
