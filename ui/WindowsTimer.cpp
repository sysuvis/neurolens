#include "WindowsTimer.h"

WindowsTimer::WindowsTimer(){
	QueryPerformanceFrequency(&mFreq);
	QueryPerformanceCounter(&mStart);
}

void WindowsTimer::start(){
	QueryPerformanceCounter(&mStart);
}

double WindowsTimer::end(){
	LARGE_INTEGER end;
	QueryPerformanceCounter(&end);

	return ((end.QuadPart-mStart.QuadPart)/(double)mFreq.QuadPart);
}