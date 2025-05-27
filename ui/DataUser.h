#ifndef DATA_USER_H
#define DATA_USER_H

#include <string>

class DataUser{
public:
	DataUser(){};
	virtual void onDataItemChanged(const std::string& name)=0;
};

#endif //DATA_USER_H