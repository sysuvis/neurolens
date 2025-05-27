#include "DataItemEditWidget.h"
#include <QString>
#include <QFont>
#include <QStringList>

#if defined(__WINDOWS__) || defined (__WIN32) || defined (WIN32) || defined (__WIN32__) || defined (__WIN64)
QFont default_font("MS Shell Dlg 2", 8);
#elif defined (__APPLE__)
QFont default_font("Helvetica", 11);
#else
QFont default_font("Times", 8);
#endif

DataItemEditWidget::DataItemEditWidget(const std::string& name, const DATA_ITEM_TYPE& type,
	const float& minv, const float& maxv, const bool& bImmediateUpdate,
	DataManager* manager, QWidget* parent)
	:QWidget(parent),
	mName(name),
	mType(type),
	mMin(minv),
	mMax(maxv),
	mbImmediateUpdate(bImmediateUpdate),
	mManager(manager),
	mLabel(NULL),
	mLineEdit(NULL),
	mSlider(NULL),
	mCheckBox(NULL),
	mColorPicker(NULL),
	mButton(NULL),
	mComboBox(NULL)
{
	if (mMax < mMin) {
		float tmp = mMax;
		mMax = mMin;
		mMin = tmp;
	}

	QString trim_name = QString(mName.c_str()).split(".")[1];

	bool bSuccess;
	if (mType == DATA_ITEM_INT || mType == DATA_ITEM_FLOAT || mType == DATA_ITEM_SHORT) {
		//label
		mLabel = new QLabel(trim_name, this);
		mLabel->setGeometry(0, 3, 170, 40); //label 
		mLabel->setFont(default_font);
		//line edit
		mLineEdit = new QLineEdit(this);
		mLineEdit->setGeometry(180, 0, 60, 30); // edit 
		mLineEdit->setFont(default_font);
		//slider
		mSlider = new QSlider(this);
		mSlider->setOrientation(Qt::Horizontal);
		mSlider->setGeometry(0, 45, 240, 20);

		if (mType == DATA_ITEM_INT) {
			mIntValue = mManager->getIntValue(mName, bSuccess);
			mLineEdit->setText(QString::number(mIntValue));
			mSlider->setMinimum(mMin);
			mSlider->setMaximum(mMax);
			mSlider->setSingleStep((mMax - mMin) / 100);
			mSlider->setPageStep((mMax - mMin) / 10);
			mSlider->setValue(mIntValue);
		}
		else if (mType == DATA_ITEM_FLOAT) {
			mFloatValue = mManager->getFloatValue(mName, bSuccess);
			mLineEdit->setText(QString::number(mFloatValue));
			mSlider->setMinimum(0);
			mSlider->setMaximum(100);
			mSlider->setSingleStep(1);
			mSlider->setPageStep(10);
			mSlider->setValue((mFloatValue - mMin) / (mMax - mMin) * 100);
		}
		else if (mType == DATA_ITEM_SHORT) {
			mShortValue = mManager->getShortValue(mName, bSuccess);
			mLineEdit->setText(QString::number(mShortValue));
			mSlider->setMinimum(mMin);
			mSlider->setMaximum(mMax);
			mSlider->setSingleStep((mMax - mMin) / 100);
			mSlider->setPageStep((mMax - mMin) / 10);
			mSlider->setValue(mShortValue);
		}
		connect(mSlider, SIGNAL(valueChanged(int)), this, SLOT(onSliderValueChanged(int)));
		connect(mSlider, SIGNAL(sliderReleased()), this, SLOT(onSliderRelease()));
		connect(mLineEdit, SIGNAL(returnPressed()), this, SLOT(onLineEditReturn()));
	}
	else if (mType == DATA_ITEM_BOOL) {
		mBoolValue = mManager->getBoolValue(mName, bSuccess);
		mCheckBox = new QCheckBox(this);
		mCheckBox->setGeometry(0, 0, 240, 20);
		mCheckBox->setChecked(mBoolValue);
		mCheckBox->setText(trim_name);
		mCheckBox->setFont(default_font);
		connect(mCheckBox, SIGNAL(toggled(bool)), this, SLOT(onCheckBoxChanged(bool)));
	}
	else if (mType == DATA_ITEM_COLOR) {
		//label
		mLabel = new QLabel(trim_name, this);
		mLabel->setGeometry(0, 3, 175, 16);
		mLabel->setFont(default_font);
		//color picker
		mColorValue = manager->getColorValue(mName, bSuccess);
		mColorPicker = new QtColorPicker(this);
		mColorPicker->setGeometry(155, 0, 80, 20);

		mColorPicker->insertColor(QColor(0.122f * 255, 0.467f * 255, 0.706f * 255, 1.0f * 255));
		mColorPicker->insertColor(QColor(1.0f * 255, 0.498f * 255, 0.055f * 255, 1.0f * 255));
		mColorPicker->insertColor(QColor(0.173f * 255, 0.627f * 255, 0.173f * 255, 1.0f * 255));
		mColorPicker->insertColor(QColor(0.839f * 255, 0.153f * 255, 0.157f * 255, 1.0f * 255));
		mColorPicker->insertColor(QColor(0.58f * 255, 0.404f * 255, 0.741f * 255, 1.0f * 255));
		mColorPicker->insertColor(QColor(0.549f * 255, 0.337f * 255, 0.294f * 255, 1.0f * 255));
		mColorPicker->insertColor(QColor(0.89f * 255, 0.467f * 255, 0.761f * 255, 1.0f * 255));
		mColorPicker->insertColor(QColor(0.498f * 255, 0.498f * 255, 0.498f * 255, 1.0f * 255));
		mColorPicker->insertColor(QColor(0.737f * 255, 0.741f * 255, 0.133f * 255, 1.0f * 255));
		mColorPicker->insertColor(QColor(0.09f * 255, 0.745f * 255, 0.812f * 255, 1.0f * 255));
		mColorPicker->insertColor(QColor(0.682f * 255, 0.78f * 255, 0.91f * 255, 1.0f * 255));
		mColorPicker->insertColor(QColor(1.0f * 255, 0.733f * 255, 0.471f * 255, 1.0f * 255));
		mColorPicker->insertColor(QColor(0.596f * 255, 0.875f * 255, 0.541f * 255, 1.0f * 255));
		mColorPicker->insertColor(QColor(1.0f * 255, 0.596f * 255, 0.588f * 255, 1.0f * 255));
		mColorPicker->insertColor(QColor(0.773f * 255, 0.69f * 255, 0.835f * 255, 1.0f * 255));
		mColorPicker->insertColor(QColor(0.769f * 255, 0.612f * 255, 0.58f * 255, 1.0f * 255));
		mColorPicker->insertColor(QColor(0.969f * 255, 0.714f * 255, 0.824f * 255, 1.0f * 255));
		mColorPicker->insertColor(QColor(0.78f * 255, 0.78f * 255, 0.78f * 255, 1.0f * 255));
		mColorPicker->insertColor(QColor(0.859f * 255, 0.859f * 255, 0.553f * 255, 1.0f * 255));
		mColorPicker->insertColor(QColor(0.62f * 255, 0.855f * 255, 0.898f * 255, 1.0f * 255));
		mColorPicker->insertColor(QColor(0, 0, 0, 255));//black

		vec4f color = 255 * mColorValue;
		mColorPicker->setCurrentColor(QColor(color.x, color.y, color.z, color.w));

		connect(mColorPicker, SIGNAL(colorChanged(const QColor&)), this, SLOT(onColorChanged(const QColor&)));
	}
	else if (mType == DATA_ITEM_STRING) {
		//label
		mLabel = new QLabel(trim_name, this);
		mLabel->setGeometry(0, 3, 240, 16);
		mLabel->setFont(default_font);
		//line edit
		mTextEdit = new QTextEdit(this);
		mTextEdit->setGeometry(0, 25, 240, 20 * minv);
		mTextEdit->setFont(default_font);
		mTextEdit->setText(QString(manager->getStringValue(mName, bSuccess).c_str()));

		connect(mTextEdit, SIGNAL(textChanged()), this, SLOT(onTextEditChange()));
	}
}

DataItemEditWidget::DataItemEditWidget(const std::string& name, const DataItemEnum& enum_item,
	DataManager* manager, QWidget* parent)
	:QWidget(parent),
	mName(name),
	mType(DATA_ITEM_ENUM),
	mMin(0),
	mMax(0),
	mbImmediateUpdate(false),
	mManager(manager),
	mLabel(NULL),
	mLineEdit(NULL),
	mSlider(NULL),
	mCheckBox(NULL),
	mColorPicker(NULL),
	mButton(NULL),
	mComboBox(NULL)
{
	if (enum_item.widget_type == DATA_ITEM_ENUM_RADIOBUTTON) {
		for (int i = 0; i < enum_item.names.size(); ++i) {
			QRadioButton* button = new QRadioButton(this);
			button->setText(QString(enum_item.names[i].c_str()));
			button->setGeometry(0, i * 25, 240, 40);
			button->setFont(default_font);
			if (i == enum_item.val) {
				button->setChecked(true);
			}
			connect(button, SIGNAL(pressed()), this, SLOT(onEnumRadioButtonChanged()));
			mRadioButtons.push_back(button);
		}
	}
	else {
		QString trim_name = QString(mName.c_str()).split(".")[1];
		mLabel = new QLabel(trim_name, this);
		mLabel->setGeometry(0, 3, 120, 30);
		mLabel->setFont(default_font);

		mComboBox = new QComboBox(this);
		mComboBox->setFont(default_font);
		for (int i = 0; i < enum_item.names.size(); ++i) {
			mComboBox->addItem(QString(enum_item.names[i].c_str()));
		}
		mComboBox->setGeometry(120, 0, 145, 30); //combex for enum
		
		mComboBox->setCurrentIndex(enum_item.val);
		connect(mComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onEnumComboBoxChanged(int)));
	}
}

DataItemEditWidget::DataItemEditWidget(const std::string& name,
	DataManager* manager, QWidget* parent)
	:QWidget(parent),
	mName(name),
	mType(DATA_ITEM_TRIGGER),
	mMin(0),
	mMax(0),
	mbImmediateUpdate(false),
	mManager(manager),
	mLabel(NULL),
	mLineEdit(NULL),
	mSlider(NULL),
	mCheckBox(NULL),
	mColorPicker(NULL),
	mButton(NULL),
	mComboBox(NULL)
{
	QString trim_name = QString(mName.c_str()).split(".")[1];
	QFont font(QString("MS Shell Dlg 2"), 10);

	QPushButton* button = new QPushButton(this);
	button->setText(trim_name);
	button->setGeometry(0, 0, 240, 35);
	button->setFont(default_font);
	connect(button, SIGNAL(pressed()), this, SLOT(onEnumRadioButtonChanged()));
}

DataItemEditWidget::~DataItemEditWidget() {
	if (mLabel)
		delete mLabel;

	if (mLineEdit)
		delete mLineEdit;

	if (mSlider)
		delete mSlider;

	if (mCheckBox)
		delete mCheckBox;

	if (mColorPicker)
		delete mColorPicker;

	if (mButton)
		delete mButton;

	if (mComboBox)
		delete mComboBox;

	for (int i = 0; i < mRadioButtons.size(); ++i) {
		delete mRadioButtons[i];
	}
}

void DataItemEditWidget::onSliderValueChanged(int value) {
	if (mType == DATA_ITEM_INT || mType == DATA_ITEM_SHORT) {
		mLineEdit->setText(QString::number(value));
	}
	else if (mType == DATA_ITEM_FLOAT) {
		mLineEdit->setText(QString::number((float)value / 100.0f * (mMax - mMin) + mMin));
	}
	if (mbImmediateUpdate) {
		onSliderRelease();
	}
}

void DataItemEditWidget::onSliderRelease() {
	if (mType == DATA_ITEM_INT) {
		mIntValue = mSlider->value();
	}
	else if (mType == DATA_ITEM_SHORT) {
		mShortValue = mSlider->value();
	}
	else if (mType == DATA_ITEM_FLOAT) {
		bool bSuccess = false;
		mFloatValue = mLineEdit->text().toFloat(&bSuccess);
	}
	updateDataManager();
}

void DataItemEditWidget::onLineEditReturn() {
	if (mType == DATA_ITEM_INT || mType == DATA_ITEM_SHORT) {
		bool bSuccess = false;
		int value = mLineEdit->text().toInt(&bSuccess);
		if (bSuccess) {
			mIntValue = value;
			mShortValue = value;
			mSlider->setValue(value);
		}
	}
	else if (mType == DATA_ITEM_FLOAT) {
		bool bSuccess = false;
		float value = mLineEdit->text().toFloat(&bSuccess);
		if (bSuccess) {
			mFloatValue = value;
			mSlider->setValue((mFloatValue - mMin) / (mMax - mMin) * 100);
		}
	}
	updateDataManager();
}

void DataItemEditWidget::onTextEditChange() {
	std::string text = mTextEdit->toPlainText().toStdString();
	int ptr = text.size() - 1;
	while (ptr >= 0 && text[ptr] == '\n') --ptr;
	if (ptr != text.size() - 1) {
		text = text.substr(0, ptr + 1);
		mManager->setStringValue(mName, text);
		mTextEdit->setText(QString(text.c_str()));
		mTextEdit->moveCursor(QTextCursor::End);
	}
}

void DataItemEditWidget::onCheckBoxChanged(bool value) {
	mBoolValue = value;
	updateDataManager();
}

void DataItemEditWidget::onColorChanged(const QColor& value) {
	mColorValue = makeVec4f(value.redF(), value.greenF(), value.blueF(), value.alphaF());
	updateDataManager();
}

void DataItemEditWidget::onEnumRadioButtonChanged() {
	QObject* sender = QObject::sender();
	for (int i = 0; i < mRadioButtons.size(); ++i) {
		if (sender == mRadioButtons[i]) {
			mEnumValue = i;
			break;
		}
	}
	updateDataManager();
}



void DataItemEditWidget::onEnumComboBoxChanged(int idx) {
	mEnumValue = idx;
	updateDataManager();
}

void DataItemEditWidget::onButtonClicked() {
	updateDataManager();
}

void DataItemEditWidget::updateDataManager() {
	if (mType == DATA_ITEM_INT) {
		mManager->setIntValue(mName, mIntValue);
	}
	else if (mType == DATA_ITEM_SHORT) {
		mManager->setShortValue(mName, mShortValue);
	}
	else if (mType == DATA_ITEM_FLOAT) {
		mManager->setFloatValue(mName, mFloatValue);
	}
	else if (mType == DATA_ITEM_BOOL) {
		mManager->setBoolValue(mName, mBoolValue);
	}
	else if (mType == DATA_ITEM_COLOR) {
		mManager->setColorValue(mName, mColorValue);
	}
	else if (mType == DATA_ITEM_ENUM) {
		mManager->setEnumValue(mName, mEnumValue);
	}
	else if (mType == DATA_ITEM_TRIGGER) {
		mManager->trigger(mName);
	}
}