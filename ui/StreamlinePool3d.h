#ifndef STREAMLINEPOOL3D_H
#define STREAMLINEPOOL3D_H

#include "typeOperation.h"

class StreamlinePool3d{
private:
	vec3f *vecf;
	int vf_w, vf_h, vf_d;
	StreamlineTraceParameter para;

	bool isOutOfBound(const vec3f &pos);
	bool streamlineAtPos3d(vec3f &pos);

public:
	StreamlinePool3d(const char* vecFile, const char* hdrFile);
	StreamlinePool3d(vec3f *vf, int w, int h, int d);
	StreamlinePool3d(vec3f* points, vec3f* velos, Streamline* lines, const int& num_lines, const int& w, const int& h, const int& d);
	~StreamlinePool3d();

	void clear(){numStl=0;numPoint=0;}
	void setParameters(const StreamlineTraceParameter& p);
	void newRandomPool();
	void addRandomPool(const int &nums);
	void addPool(Streamline* stls, int nStl, vec3f* pnts, int nPnt);
	void traceSeeds(vec3f* seeds, int numSeeds);
	void recomputeVelos();
	bool vecAtPos3d(vec3f &ret, const vec3f &pos);
	bool nextPos3d(vec3f &ret, vec3f &pos, bool isFoward);//RK4
	bool loadFromFile(const char *filename);
	bool saveToFile(char *filename);

	int getVfWidth(){return vf_w;}
	int getVfHeight(){return vf_h;}
	int getVfDepth(){return vf_d;}
	vec3f* getVecField(){return vecf;}

	int numStl, numPoint;
	float radius;
	vec3f *pointArray;
	vec3f *veloArray;
	Streamline *stlArray;
};

#endif
