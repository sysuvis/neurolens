#ifndef MESSAGE_CENTER_H
#define MESSAGE_CENTER_H

#include <string>

class MessageCenter{
public:
	MessageCenter();
	~MessageCenter();
	static MessageCenter* sharedCenter();
	void processMessage(const std::string& message, const std::string& sender);
};

#endif//MESSAGE_CENTER_H