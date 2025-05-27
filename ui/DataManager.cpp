#include "DataManager.h"
#include "DataItemEditWidget.h"
#include "DataUser.h"
#include <cstdarg>
#include <QGroupBox>

DataManager::DataManager() {
}

DataManager::~DataManager() {
}

DataManager* DataManager::sharedManager() {
	static DataManager shared_manager;
	return &shared_manager;
}

bool DataManager::createInt(std::string name, const int& value,
	const int& maxv, const int& minv, DataUser* user,
	const bool& bSliderUpdate)
{
	bool bNonExisting = true;
	if (integers.find(name) != integers.end()) {
		printf("Err: DataManager: Fail to create item. %s\n", name.c_str());
		bNonExisting = false;
	}

	integers[name].current = value;
	integers[name].maxv = maxv;
	integers[name].minv = minv;
	integers[name].bSliderUpdate = bSliderUpdate;
	if (user != NULL) dataUsers[name].insert(user);

	return bNonExisting;
}

bool DataManager::createFloat(std::string name, const float& value,
	const float& maxv, const float& minv, DataUser* user,
	const bool& bSliderUpdate)
{
	bool bNonExisting = true;
	if (floats.find(name) != floats.end()) {
		printf("Err: DataManager: Fail to create item. %s\n", name.c_str());
		bNonExisting = false;
	}

	floats[name].current = value;
	floats[name].maxv = maxv;
	floats[name].minv = minv;
	floats[name].bSliderUpdate = bSliderUpdate;
	if (user != NULL) dataUsers[name].insert(user);

	return bNonExisting;
}

bool DataManager::createShort(std::string name, const short& value,
	const short& maxv, const short& minv, DataUser* user,
	const bool& bSliderUpdate)
{
	bool bNonExisting = true;
	if (shorts.find(name) != shorts.end()) {
		printf("Err: DataManager: Fail to create item. %s\n", name.c_str());
		bNonExisting = false;
	}

	shorts[name].current = value;
	shorts[name].maxv = maxv;
	shorts[name].minv = minv;
	shorts[name].bSliderUpdate = bSliderUpdate;
	if (user != NULL) dataUsers[name].insert(user);

	return bNonExisting;
}

bool DataManager::createBool(std::string name, const bool& value, DataUser* user) {
	bool bNonExisting = true;
	if (bools.find(name) != bools.end()) {
		printf("Err: DataManager: Fail to create item. %s\n", name.c_str());
		bNonExisting = false;
	}

	bools[name] = value;
	if (user != NULL) dataUsers[name].insert(user);

	return bNonExisting;
}

bool DataManager::createColor(std::string name, const vec4f& value, DataUser* user) {
	bool bNonExisting = true;
	if (colors.find(name) != colors.end()) {
		printf("Err: DataManager: Fail to create item. %s\n", name.c_str());
		bNonExisting = false;
	}

	colors[name] = value;
	if (user != NULL) dataUsers[name].insert(user);

	return bNonExisting;
}

bool DataManager::createEnum(std::string name, const std::vector<std::string>& val_names,
	const int& value, const DataItemEnumWidgetType& widget_type, DataUser* user) {
	bool bNonExisting = true;
	if (enums.find(name) != enums.end()) {
		printf("Err: DataManager: Fail to create item. %s\n", name.c_str());
		bNonExisting = false;
	}

	DataItemEnum enum_item;
	enum_item.val = value;
	enum_item.names.assign(val_names.begin(), val_names.end());
	enum_item.widget_type = (DataItemEnumWidgetType)clamp((int)widget_type, 0, 1);
	enums[name] = enum_item;
	if (user != NULL) dataUsers[name].insert(user);

	return bNonExisting;
}

bool DataManager::createString(std::string name, const std::string& value, const int& num_lines, DataUser* user)
{
	bool bNonExisting = true;
	if (strings.find(name) != strings.end()) {
		printf("Err: DataManager: Fail to create item. %s\n", name.c_str());
		bNonExisting = false;
	}

	DataItemString string_item;
	string_item.str = value;
	string_item.num_lines = num_lines;
	strings[name] = string_item;
	if (user != NULL) dataUsers[name].insert(user);

	return bNonExisting;
}

bool DataManager::createPointer(std::string name, const PointerType& value, DataUser* user) {
	bool bNonExisting = true;
	if (pointers.find(name) != pointers.end()) {
		printf("Err: DataManager: Fail to create item. %s\n", name.c_str());
		bNonExisting = false;
	}

	pointers[name] = value;
	if (user != NULL) dataUsers[name].insert(user);

	return bNonExisting;
}

bool DataManager::createTrigger(std::string name, DataUser* user) {
	bool bNonExisting = true;
	if (dataUsers.find(name) != dataUsers.end()) {
		printf("Err: DataManager: Fail to create item. %s\n", name.c_str());
		bNonExisting = false;
	}

	if (user != NULL) dataUsers[name].insert(user);

	return bNonExisting;
}

bool DataManager::registerItem(const std::string& name, DataUser* user) {
	if (dataUsers.find(name) == dataUsers.end()) {
		printf("Err: DataManager: Fail to register item.\n");
		return false;
	}

	dataUsers[name].insert(user);

	return true;
}

int DataManager::getIntValue(const std::string& name, bool& bSuccess) {
	auto it = integers.find(name);
	if (it == integers.end()) {
		bSuccess = false;
		printf("Err: Fail to get value:%s.\n", name.c_str());
		return 0;
	}
	bSuccess = true;
	return (it->second.current);
}

float DataManager::getFloatValue(const std::string& name, bool& bSuccess) {
	auto it = floats.find(name);
	if (it == floats.end()) {
		bSuccess = false;
		printf("Err: Fail to get value:%s.\n", name.c_str());
		return 0;
	}
	bSuccess = true;
	return (it->second.current);
}

short DataManager::getShortValue(const std::string& name, bool& bSuccess) {
	auto it = shorts.find(name);
	if (it == shorts.end()) {
		bSuccess = false;
		printf("Err: Fail to get value:%s.\n", name.c_str());
		return 0;
	}
	bSuccess = true;
	return (it->second.current);
}

bool DataManager::getBoolValue(const std::string& name, bool& bSuccess) {
	auto it = bools.find(name);
	if (it == bools.end()) {
		bSuccess = false;
		printf("Err: Fail to get value:%s.\n", name.c_str());
		return 0;
	}
	bSuccess = true;
	return (it->second);
}

vec4f DataManager::getColorValue(const std::string& name, bool& bSuccess) {
	auto it = colors.find(name);
	if (it == colors.end()) {
		bSuccess = false;
		printf("Err: Fail to get value:%s.\n", name.c_str());
		return makeVec4f(0, 0, 0, 0);
	}
	bSuccess = true;
	return (it->second);
}

int DataManager::getEnumValue(const std::string& name, bool& bSuccess) {
	auto it = enums.find(name);
	if (it == enums.end()) {
		bSuccess = false;
		printf("Err: Fail to get value:%s.\n", name.c_str());
		return 0;
	}
	bSuccess = true;
	return (it->second.val);
}

std::string DataManager::getStringValue(const std::string& name, bool& bSuccess)
{
	auto it = strings.find(name);
	if (it == strings.end()) {
		bSuccess = false;
		printf("Err: Fail to get value:%s.\n", name.c_str());
		return 0;
	}
	bSuccess = true;
	return (it->second.str);
}


PointerType DataManager::getPointerValue(const std::string& name, bool& bSuccess) {
	auto it = pointers.find(name);
	if (it == pointers.end()) {
		bSuccess = false;
		printf("Err: Fail to get value:%s.\n", name.c_str());
		return 0;
	}
	bSuccess = true;
	return (it->second);
}

bool DataManager::setIntValue(const std::string& name, const int& value) {
	auto it = integers.find(name);
	if (it == integers.end()) {
		createInt(name, value);
		printf("Err: Fail to set value:%s.\n", name.c_str());
		return false;
	}
	it->second.current = value;
	notifyChange(name);
	return true;
}

bool DataManager::setFloatValue(const std::string& name, const float& value) {
	auto it = floats.find(name);
	if (it == floats.end()) {
		createFloat(name, value);
		printf("Err: Fail to set value:%s.\n", name.c_str());
		return false;
	}
	it->second.current = value;
	notifyChange(name);
	return true;
}

bool DataManager::setShortValue(const std::string& name, const short& value) {
	auto it = shorts.find(name);
	if (it == shorts.end()) {
		createShort(name, value);
		printf("Err: Fail to set value:%s.\n", name.c_str());
		return false;
	}
	it->second.current = value;
	notifyChange(name);
	return true;
}

bool DataManager::setBoolValue(const std::string& name, const bool& value) {
	auto it = bools.find(name);
	if (it == bools.end()) {
		createBool(name, value);
		printf("Err: Fail to set value:%s.\n", name.c_str());
		return false;
	}
	it->second = value;
	notifyChange(name);
	return true;
}

bool DataManager::setColorValue(const std::string& name, const vec4f& value) {
	auto it = colors.find(name);
	if (it == colors.end()) {
		createColor(name, value);
		printf("Err: Fail to set value:%s.\n", name.c_str());
		return false;
	}
	it->second = value;
	notifyChange(name);
	return true;
}

bool DataManager::setEnumValue(const std::string& name, const int& value) {
	auto it = enums.find(name);
	if (it == enums.end()) {
		//createColor(name, value);
		printf("Err: Fail to find enum item:%s.\n", name.c_str());
		return false;
	}
	if (value < 0 || value >= it->second.names.size()) {
		printf("Err: Fail to set value:%s.\n", name.c_str());
		return false;
	}
	it->second.val = value;
	notifyChange(name);
	return true;
}

bool DataManager::setStringValue(const std::string& name, const std::string& value)
{
	auto it = strings.find(name);
	if (it == strings.end()) {
		createString(name, value);
		printf("Err: Fail to set value:%s.\n", name.c_str());
		return false;
	}
	it->second.str = value;
	notifyChange(name);
	return true;
}

bool DataManager::setPointerValue(const std::string& name, const PointerType& value) {
	auto it = pointers.find(name);
	if (it == pointers.end()) {
		createBool(name, value);
		printf("Err: Fail to set value:%s.\n", name.c_str());
		return false;
	}
	it->second = value;
	notifyChange(name);
	return true;
}

void DataManager::trigger(const std::string& name) {
	notifyChange(name);
}

QWidget* DataManager::createInterface(std::string title, std::vector<std::string> names, QWidget* parent) {
#if defined(__WINDOWS__) || defined (__WIN32) || defined (WIN32) || defined (__WIN32__) || defined (__WIN64)
	QFont font("MS Shell Dlg 2", 8);
	int y = 15, yd = 70, yd2 = 50, yd3 = 100;
#elif defined (__APPLE__)
	QFont font("Helvetica", 11);
	int y = 10, yd = 40, yd2 = 20, yd3 = 25;
#else
	QFont font("Times", 8);
	int y = 5, yd = 45, yd2 = 25, yd3 = 30;
#endif

	QWidget* win;
	QGroupBox* gb;
	if (parent == NULL) {
		win = new QWidget();
		win->setWindowTitle(QString(title.c_str()));
	}
	else {
		win = new QWidget(parent);
		gb = new QGroupBox(QString(title.c_str()), win);
		gb->setFlat(false);
		gb->setFont(QFont("Helvetica", 12));
		y += 20;
	}

	int num = names.size();
	for (int i = 0; i < num; ++i) {
		std::string name = names[i];
		if (integers.find(name) != integers.end()) {
			DataItemInt& item = integers[name];
			DataItemEditWidget* widget = new DataItemEditWidget(name, DATA_ITEM_INT,
				item.minv, item.maxv, item.bSliderUpdate, this, win);
			widget->setGeometry(10, y, 240, yd);
			y += yd;
		}
		else if (floats.find(name) != floats.end()) {
			DataItemFloat& item = floats[name];
			DataItemEditWidget* widget = new DataItemEditWidget(name, DATA_ITEM_FLOAT,
				item.minv, item.maxv, item.bSliderUpdate, this, win);
			widget->setGeometry(10, y, 240, yd);
			y += yd;
		}
		else if (shorts.find(name) != shorts.end()) {
			DataItemShort& item = shorts[name];
			DataItemEditWidget* widget = new DataItemEditWidget(name, DATA_ITEM_SHORT,
				item.minv, item.maxv, item.bSliderUpdate, this, win);
			widget->setGeometry(10, y, 240, yd);
			y += yd;
		}
		else if (bools.find(name) != bools.end()) {
			DataItemEditWidget* widget = new DataItemEditWidget(name, DATA_ITEM_BOOL,
				0, 1, false, this, win);
			widget->setGeometry(10, y, 240, yd2);
			y += yd2;
		}
		else if (colors.find(name) != colors.end()) {
			DataItemEditWidget* widget = new DataItemEditWidget(name, DATA_ITEM_COLOR,
				0, 1, false, this, win);
			widget->setGeometry(10, y, 240, yd3);
			y += yd3;
		}
		else if (enums.find(name) != enums.end()) {
			DataItemEnum& enum_item = enums[name];
			DataItemEditWidget* widget = new DataItemEditWidget(name, enum_item, this, win);
			int height;
			if (enum_item.widget_type == DATA_ITEM_ENUM_RADIOBUTTON) {
				height = 20 * enum_item.names.size() + 10;
			}
			else {
				height = 40; //enum gap
			}
			widget->setGeometry(10, y, 240, height);
			y += height;
		}
		else if (strings.find(name) != strings.end()) {
			DataItemString& string_item = strings[name];
			DataItemEditWidget* widget = new DataItemEditWidget(name, DATA_ITEM_STRING,
				string_item.num_lines, string_item.num_lines, false, this, win);
			int height = yd3 + 20 * string_item.num_lines;
			widget->setGeometry(10, y, 240, height);
			y += height;
		}
		else if (dataUsers.find(name) != dataUsers.end()) {//trigger
			DataItemEditWidget* widget = new DataItemEditWidget(name, this, win);
			widget->setGeometry(10, y, 240, yd2);
			y += yd2;
		}
	}

	if (parent == NULL) {
		win->setGeometry(1265, 50, 260, y);
		win->show();
	}
	else {
		gb->setGeometry(5, 5, 250, y - 5);
		win->setGeometry(0, 0, 260, y);
	}

	return win;
}

bool DataManager::notifyChange(const std::string& name) {
	if (dataUsers.find(name) == dataUsers.end()) {
		printf("Err: DataManager: Fail to notify change of item: %s.\n", name.c_str());
		return false;
	}

	DataUserSet& users = dataUsers[name];
	for (DataUserSet::iterator it = users.begin(); it != users.end(); ++it) {
		(*it)->onDataItemChanged(name);
	}

	return true;
}