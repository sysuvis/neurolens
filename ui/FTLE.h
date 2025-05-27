#ifndef FTLE_H
#define FTLE_H

#include "typeOperation.h"
#include "VolumeData.h"

VolumeData<float>* FTLE(const char* directory, const char* filename_format, 
				 const int& start_time, const int& end_time,
				 const int& w, const int& h, const int& d);

#endif //FTLE_H