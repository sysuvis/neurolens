#ifndef VTK_READER_H
#define VTK_READER_H

#include "typeOperation.h"
#include "VolumeData.h"
#include <map>
#include <string>

class VTKReader {
public:
	VTKReader(char* file_path);
	~VTKReader();

	vec3f getOrigin(){return mOrigin;}
	vec3f getAspectRatios(){return mAspects;}
	vec3i getDimension(){return mDim;}

	float getScalarMax(std::string name){return mScalarMax[name];}
	float getScalarMin(std::string name){return mScalarMin[name];}
	VolumeData<float>* getScalarField(std::string name){return mScalarFields[name];}
	VolumeData<vec3f>* getVectorField(){return mVectorField;}

private:
	void switchEndian4(char* data, const int& num_item);
	bool readFile(char* file_path);
	void adjustVectorField();

	std::map<std::string, VolumeData<float>*> mScalarFields;	
	VolumeData<vec3f>* mVectorField;
	std::map<std::string, float> mScalarMin;
	std::map<std::string, float> mScalarMax;

	vec3f mOrigin;
	vec3f mAspects;
	vec3i mDim;
};

#endif