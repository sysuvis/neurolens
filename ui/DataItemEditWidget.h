#ifndef DATA_ITEM_EDIT_WIDGET_H
#define DATA_ITEM_EDIT_WIDGET_H

#include <QWidget>
#include <QLabel>
#include <QLineEdit>
#include <QSlider>
#include <QCheckBox>
#include <QRadioButton>
#include <QPushButton>
#include <QComboBox>
#include <QTextEdit>
#include <string>
#include "qtcolorpicker.h"
#include "typeOperation.h"
#include "DataManager.h"

typedef enum {
	DATA_ITEM_TYPE_NOT_SPECIFY = 0,
	DATA_ITEM_INT,
	DATA_ITEM_FLOAT,
	DATA_ITEM_SHORT,
	DATA_ITEM_BOOL,
	DATA_ITEM_COLOR,
	DATA_ITEM_ENUM,
	DATA_ITEM_STRING,
	DATA_ITEM_TRIGGER
} DATA_ITEM_TYPE;

class DataItemEditWidget : public QWidget {
	Q_OBJECT
public:
	DataItemEditWidget(const std::string& name, const DATA_ITEM_TYPE& type,
		const float& minv, const float& maxv, const bool& bImmediateUpdate,
		DataManager* manager, QWidget* parent = 0);

	DataItemEditWidget(const std::string& name, const DataItemEnum& enum_item,
		DataManager* manager, QWidget* parent = 0);

	DataItemEditWidget(const std::string& name, DataManager* manager, QWidget* parent = 0);

	~DataItemEditWidget();

public slots:
	void onSliderValueChanged(int value);
	void onSliderRelease();
	void onLineEditReturn();
	void onTextEditChange();
	void onCheckBoxChanged(bool value);
	void onColorChanged(const QColor& value);
	void onEnumRadioButtonChanged();
	void onEnumComboBoxChanged(int idx);
	void onButtonClicked();

private:
	void updateDataManager();

	DataManager* mManager;
	std::string mName;

	//type
	DATA_ITEM_TYPE	mType;

	//for slider
	float		mMin;
	float		mMax;
	bool		mbImmediateUpdate;

	//value
	int			mIntValue;
	short		mShortValue;
	float		mFloatValue;
	bool		mBoolValue;
	vec4f		mColorValue;
	int			mEnumValue;
	std::string mStringValue;

	//interface
	QLabel* mLabel;
	QLineEdit* mLineEdit;
	QTextEdit* mTextEdit;
	QSlider* mSlider;
	QCheckBox* mCheckBox;
	QPushButton* mButton;
	QComboBox* mComboBox;
	QtColorPicker* mColorPicker;
	std::vector<QRadioButton*> mRadioButtons;
};

#endif //DATA_ITEM_EDIT_WIDGET_H