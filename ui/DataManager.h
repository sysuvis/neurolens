#ifndef DATA_MANAGER_H
#define DATA_MANAGER_H

#include <map>
#include <set>
#include <string>
#include <vector>
#include "typeOperation.h"

class DataUser;

typedef struct {
	int current, maxv, minv;
	bool bSliderUpdate;
} DataItemInt;

typedef struct {
	float current, maxv, minv;
	bool bSliderUpdate;
} DataItemFloat;

typedef struct {
	short current, maxv, minv;
	bool bSliderUpdate;
} DataItemShort;

typedef enum {
	DATA_ITEM_ENUM_RADIOBUTTON = 0,
	DATA_ITEM_ENUM_COMBOBOX
} DataItemEnumWidgetType;

typedef struct {
	std::vector<std::string> names;
	int val;
	DataItemEnumWidgetType widget_type;
} DataItemEnum;

typedef struct {
	std::string str;
	int num_lines;
} DataItemString;

typedef long long PointerType;
typedef std::map<std::string, DataItemInt> DMIntMap;
typedef std::map<std::string, DataItemFloat> DMFloatMap;
typedef std::map<std::string, DataItemShort> DMShortMap;
typedef std::map<std::string, bool> DMBoolMap;
typedef std::map<std::string, PointerType> DMPointerMap;
typedef std::map<std::string, vec4f> DMColorMap;
typedef std::map<std::string, DataItemEnum> DMEnumMap;
typedef std::map<std::string, DataItemString> DMStringMap;
typedef std::set<DataUser*> DataUserSet;
typedef std::map<std::string, DataUserSet> DataUserMap;

class QWidget;

class DataManager {
public:
	DataManager();
	~DataManager();
	static DataManager* sharedManager();

	bool createInt(std::string name, const int& value,
		const int& maxv = 1, const int& minv = 0, DataUser* user = NULL,
		const bool& bSliderUpdate = false);
	bool createFloat(std::string name, const float& value,
		const float& maxv = 1.0f, const float& minv = 0.0f, DataUser* user = NULL,
		const bool& bSliderUpdate = false);
	bool createShort(std::string name, const short& value,
		const short& maxv = 1, const short& minv = 0, DataUser* user = NULL,
		const bool& bSliderUpdate = false);
	bool createBool(std::string name, const bool& value, DataUser* user = NULL);
	bool createColor(std::string name, const vec4f& value, DataUser* user = NULL);
	bool createPointer(std::string name, const PointerType& value, DataUser* user = NULL);
	bool createEnum(std::string name, const std::vector<std::string>& val_names,
		const int& value, const DataItemEnumWidgetType& widget_type = DATA_ITEM_ENUM_RADIOBUTTON,
		DataUser* user = NULL);
	bool createString(std::string name, const std::string& value, const int& num_lines = 3, DataUser* user = NULL);
	bool createTrigger(std::string name, DataUser* user = NULL);

	bool registerItem(const std::string& name, DataUser* user);

	int getIntValue(const std::string& name, bool& bSuccess);
	float getFloatValue(const std::string& name, bool& bSuccess);
	short getShortValue(const std::string& name, bool& bSuccess);
	bool getBoolValue(const std::string& name, bool& bSuccess);
	vec4f getColorValue(const std::string& name, bool& bSuccess);
	PointerType getPointerValue(const std::string& name, bool& bSuccess);
	int getEnumValue(const std::string& name, bool& bSuccess);
	std::string getStringValue(const std::string& name, bool& bSuccess);

	bool setIntValue(const std::string& name, const int& value);
	bool setFloatValue(const std::string& name, const float& value);
	bool setShortValue(const std::string& name, const short& value);
	bool setBoolValue(const std::string& name, const bool& value);
	bool setColorValue(const std::string& name, const vec4f& value);
	bool setPointerValue(const std::string& name, const PointerType& value);
	bool setEnumValue(const std::string& name, const int& value);
	bool setStringValue(const std::string& name, const std::string& value);
	void trigger(const std::string& name);

	QWidget* createInterface(std::string title, std::vector<std::string> names, QWidget* parent = NULL);

private:
	bool notifyChange(const std::string& name);

	//data items
	DMIntMap		integers;
	DMFloatMap		floats;
	DMShortMap		shorts;
	DMBoolMap		bools;
	DMColorMap		colors;
	DMPointerMap	pointers;
	DMEnumMap		enums;
	DMStringMap		strings;

	//data user
	DataUserMap dataUsers;
};

#endif //DATA_MANAGER_H