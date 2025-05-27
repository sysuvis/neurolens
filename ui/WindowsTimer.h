#ifndef WINDOWS_TIMER_H
#define WINDOWS_TIMER_H

#include <Windows.h>

class WindowsTimer{
public:
	WindowsTimer();
	void start();
	double end();

private:
	LARGE_INTEGER mFreq, mStart;
};

#endif //WINDOWS_TIMER_H