#include "definition.h"
#include "StreamlinePool3d.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <time.h>

#define abs(x)  ((x)>0?(x):-(x))

StreamlinePool3d::~StreamlinePool3d(){
    delete[] pointArray;
    delete[] veloArray;
    delete[] stlArray;
}

StreamlinePool3d::StreamlinePool3d(const char* vecFile, const char* hdrFile){
	int whd;
	std::ifstream ifVec, ifHeader;
	ifHeader.open(hdrFile);
	if(ifHeader.is_open()){
		ifHeader>>vf_w>>vf_h>>vf_d;
		ifHeader.close();
	} else {
		printf("Cannot read file: %s \n", hdrFile);
	}

	whd = vf_w*vf_h*vf_d;

	vecf = new vec3f[whd];

	ifVec.open(vecFile, std::ios::binary);
	if(ifVec.is_open()){
		if(strstr(vecFile, "electro3D")!=NULL) ifVec.seekg(sizeof(int)*3, std::ios::beg);
		ifVec.read((char*)vecf, whd*sizeof(vec3f));
		ifVec.close();

		//normalize vectors (scale so that the average length is 1)
		float maxlen, minlen, avglen = 0.0f, len;
		if(strstr(vecFile, "hurricane")!=NULL){//hurricane
			maxlen = 0.0f;
			minlen = vec3fLen(vecf[0]);
			int cnt = 0;
			for(int i=0; i<whd; i++){
				if((fabs(vecf[i].x)!=fabs(vecf[0].x))&&(fabs(vecf[i].y)!=fabs(vecf[0].y))&&(fabs(vecf[i].z)!=fabs(vecf[0].z))){
					len = vec3fLen(vecf[i]);
					if(len>maxlen) maxlen = len;
					if(len<minlen) minlen = len;
					avglen += len;
					cnt++;
				}
			}
			maxlen /= avglen;
			minlen /= avglen;
			len = cnt/avglen;
			for(int i=0; i<whd; i++) vecf[i] = len*vecf[i];
		} else if(strstr(vecFile, "vsfs9")!=NULL){//vsfs9
			maxlen = 0.0f;
			minlen = vec3fLen(vecf[0]);
			int cnt = 0;
			for(int i=0; i<whd; i++){
				len = vec3fLen(vecf[i]);
				if(len>0){
					if(len>maxlen) maxlen = len;
					if(len<minlen) minlen = len;
					avglen += len;
					cnt++;
				}
			}
			maxlen /= avglen;
			minlen /= avglen;
			len = cnt/avglen;
			for(int i=0; i<whd; i++) vecf[i] = len*vecf[i];
		} else {//other data sets
			avglen = vec3fLen(vecf[0]);
			maxlen = avglen;
			minlen = avglen;
			for(int i=1; i<whd; i++){
				len = vec3fLen(vecf[i]);
				if(len>maxlen) maxlen = len;
				if(len<minlen) minlen = len;
				avglen += len;
			}
			maxlen /= avglen;
			len = vf_w*vf_h*vf_d/avglen/10.0f;
			for(int i=0; i<whd; i++) vecf[i] = len*vecf[i];
		}
	} else {
		printf("Cannot read file: %s \n", vecFile);
	}

	numPoint = 0;
	numStl = 0;
	radius = TUBE_RADIUS;

	pointArray = new vec3f[MAX_STREAMLINE_NUM*MAX_POINT_NUM];
	veloArray = new vec3f[MAX_STREAMLINE_NUM*MAX_POINT_NUM];
	stlArray = new Streamline[MAX_STREAMLINE_NUM];

	para.max_streamline = MAX_STREAMLINE_NUM;
	para.max_point = MAX_POINT_NUM;
	para.min_point = MIN_POINT_NUM;
	para.segment_length = SEG_LEN;
	para.max_step = MAX_STEP_NUM;
	para.trace_interval = TRACE_INTERVAL;
}

StreamlinePool3d::StreamlinePool3d(vec3f *vf, int w, int h, int d){
    vecf = vf;
    vf_w = w;
    vf_h = h;
    vf_d = d;

    numPoint = 0;
    numStl = 0;
    radius = TUBE_RADIUS;

    pointArray = new vec3f[MAX_STREAMLINE_NUM*MAX_POINT_NUM];
    veloArray = new vec3f[MAX_STREAMLINE_NUM*MAX_POINT_NUM];
    stlArray = new Streamline[MAX_STREAMLINE_NUM];

	para.max_streamline = MAX_STREAMLINE_NUM;
	para.max_point = MAX_POINT_NUM;
	para.min_point = MIN_POINT_NUM;
	para.segment_length = SEG_LEN;
	para.max_step = MAX_STEP_NUM;
	para.trace_interval = TRACE_INTERVAL;
}

StreamlinePool3d::StreamlinePool3d( vec3f* points, vec3f* velos, Streamline* lines, const int& num_lines,
								   const int& w, const int& h, const int& d)
{
	vecf = NULL;
	vf_w = w;
	vf_h = h;
	vf_d = d;

	Streamline last_line = lines[num_lines-1];
	numPoint = last_line.start+last_line.numPoint;
	numStl = num_lines;
	radius = TUBE_RADIUS;

	pointArray = points;
	veloArray = velos;
	stlArray = lines;

	para.max_streamline = MAX_STREAMLINE_NUM;
	para.max_point = MAX_POINT_NUM;
	para.min_point = MIN_POINT_NUM;
	para.segment_length = SEG_LEN;
	para.max_step = MAX_STEP_NUM;
	para.trace_interval = TRACE_INTERVAL;
}

void StreamlinePool3d::setParameters(const StreamlineTraceParameter& p){
	para = p;
	if(para.max_streamline>MAX_STREAMLINE_NUM){
		delete[] stlArray;
		stlArray = new Streamline[para.max_streamline];
	}

	int num = para.max_streamline*para.max_point;
	if (num>MAX_STREAMLINE_NUM*MAX_POINT_NUM) {
		delete[] pointArray;
		delete[] veloArray;
		pointArray = new vec3f[num];
		veloArray = new vec3f[num];
	}
}

bool StreamlinePool3d::loadFromFile(const char *filename){
	std::ifstream infile;
    infile.open(filename, std::ios::binary);
    if(!infile.is_open()){
		printf("Unable to read file: %s.", filename);
        return false;
    }
    infile.read((char*)&numPoint, sizeof(int));
    infile.read((char*)&numStl, sizeof(int));
    infile.read((char*)&radius, sizeof(float));

	if(numStl>MAX_STREAMLINE_NUM){
		delete[] stlArray;
		stlArray = new Streamline[numStl];
	}
	if (numPoint>MAX_STREAMLINE_NUM*MAX_POINT_NUM) {
		delete[] pointArray;
		delete[] veloArray;
		pointArray = new vec3f[numPoint];
		veloArray = new vec3f[numPoint];
	}

    infile.read((char*)pointArray, sizeof(vec3f)*numPoint);
    infile.read((char*)veloArray, sizeof(vec3f)*numPoint);
    infile.read((char*)stlArray, sizeof(Streamline)*numStl);
    infile.close();

    return true;
}

bool StreamlinePool3d::saveToFile(char *filename){
    std::ofstream outfile;
    outfile.open(filename, std::ios::binary);
    if(!outfile.is_open()){
        printf("Unable to write file: %s.", filename);
        return false;
    }
    outfile.write((char*)&numPoint, sizeof(int));
    outfile.write((char*)&numStl, sizeof(int));
    outfile.write((char*)&radius, sizeof(float));
    outfile.write((char*)pointArray, sizeof(vec3f)*numPoint);
    outfile.write((char*)veloArray, sizeof(vec3f)*numPoint);
    outfile.write((char*)stlArray, sizeof(Streamline)*numStl);
    outfile.close();
    return true;
}

bool StreamlinePool3d::isOutOfBound(const vec3f &pos){
    if (pos.x<=0.0000001f || pos.x>=(vf_w-1.0000001f) || pos.y<=0.0000001f || pos.y>=(vf_h-1.0000001f) || pos.z<=0.0000001f || pos.z>=(vf_d-1.0000001f)) {
        return true;
    }
    return false;
}

bool StreamlinePool3d::vecAtPos3d(vec3f &ret, const vec3f &pos){
    if (pos.x<=0.0000001f || pos.x>=(vf_w-1.0000001f) || pos.y<=0.0000001f || pos.y>=(vf_h-1.0000001f) || pos.z<=0.0000001f || pos.z>=(vf_d-1.0000001f)) {
        return false;
    }

    int x = floorf(pos.x);
    int y = floorf(pos.y);
    int z = floorf(pos.z);
    int idx1 = x+y*vf_w+z*vf_w*vf_h;
    int idx2 = x+1+y*vf_w+z*vf_w*vf_h;
    int idx3 = x+(y+1)*vf_w+z*vf_w*vf_h;
    int idx4 = x+1+(y+1)*vf_w+z*vf_w*vf_h;
    int idx5 = x+y*vf_w+(z+1)*vf_w*vf_h;
    int idx6 = x+1+y*vf_w+(z+1)*vf_w*vf_h;
    int idx7 = x+(y+1)*vf_w+(z+1)*vf_w*vf_h;
    int idx8 = x+1+(y+1)*vf_w+(z+1)*vf_w*vf_h;

    float facx=pos.x-x, facy=pos.y-y, facz=pos.z-z;

    ret = (1-facx)*(1-facy)*(1-facz)*vecf[idx1]\
            +facx*(1-facy)*(1-facz)*vecf[idx2]\
            +(1-facx)*facy*(1-facz)*vecf[idx3]\
            +facx*facy*(1-facz)*vecf[idx4]\
            +(1-facx)*(1-facy)*facz*vecf[idx5]\
            +facx*(1-facy)*facz*vecf[idx6]\
            +(1-facx)*facy*facz*vecf[idx7]\
            +facx*facy*facz*vecf[idx8];
    return true;
}

bool StreamlinePool3d::nextPos3d(vec3f &ret, vec3f &pos, bool isFoward){//RK4
    vec3f k1, k2, k3, k4;
    float itv;
    if (isFoward) {
        itv = para.trace_interval;
    } else {
        itv = -para.trace_interval;
    }

    if (vecAtPos3d(k1, pos)) {
        if (vecAtPos3d(k2, pos+0.5f*itv*k1)) {
            if (vecAtPos3d(k3, pos+0.5f*itv*k2)) {
                if (vecAtPos3d(k4, pos+itv*k3)) {
                    ret = pos+(itv/6.0f)*(k1+2*k2+2*k3+k4);
                    return true;
                }
            }
        }
    }
    return false;
}

bool StreamlinePool3d::streamlineAtPos3d(vec3f &pos){
    vec3f* parray = new vec3f[para.max_point];
    vec3f* varray = new vec3f[para.max_point];
    vec3f lastPos = pos;
    parray[0] = pos;
	if(!vecAtPos3d(varray[0], pos)) {
		delete[] parray;
		delete[] varray;
		return false;
	}

    int count=1, stepCount=0, numBack=0;
    //backward tracing first
    while (nextPos3d(pos, pos, false) && count<para.max_point) {
        ++stepCount;
        if (dist3d(lastPos, pos)>=para.segment_length) {
            lastPos = pos;
            parray[count] = pos;
            vecAtPos3d(varray[count], pos);
            count++;
            stepCount = 0;
            continue;
        }
        if (stepCount>para.max_step) {
            break;
        }
    }
    numBack = count;
    if(isOutOfBound(parray[numBack-1])){
        --numBack;
        --count;
    }

    //forward tracing
    pos = parray[0];
    lastPos = pos;
    stepCount = 0;
    while (nextPos3d(pos, pos, true) && count<para.max_point) {
        stepCount++;
        if (dist3d(lastPos, pos)>=para.segment_length) {
            lastPos = pos;
            parray[count] = pos;
            vecAtPos3d(varray[count], pos);
            ++count;
            stepCount = 0;
            continue;
        }
        if (stepCount>para.max_step) {
            break;
        }
    }
    if(isOutOfBound(parray[count-1])){
        --count;
    }

    if (count<para.min_point) {
        //printf("Streamline Tracing at (%11.6f, %11.6f, %11.6f) fail: not enough points (%5d).\n", parray[0].x, parray[0].y, parray[0].z, count);
		delete[] parray;
		delete[] varray;
		return false;
    }
    //copy backward points to the global array
    for (int i=0; i<numBack; i++) {
        pointArray[numPoint+i] = parray[numBack-i-1];
        veloArray[numPoint+i] = varray[numBack-i-1];
    }
    //copy forward points
    memcpy(&pointArray[numPoint+numBack], &parray[numBack], sizeof(vec3f)*(count-numBack));
    memcpy(&veloArray[numPoint+numBack], &varray[numBack], sizeof(vec3f)*(count-numBack));

    Streamline stl = {numStl, numPoint, count};
    stlArray[numStl] = stl;
    ++numStl;
    numPoint += count;

//     printf("Streamline Tracing at (%5.2f, %5.2f, %5.2f) finished: backward (%4d); forward (%4d).\n",\
//            parray[0].x, parray[0].y, parray[0].z, numBack, count-numBack);
    
	delete[] parray;
	delete[] varray;
	return true;
}

void StreamlinePool3d::newRandomPool(){
    numPoint = 0;
    numStl = 0;
    addRandomPool(GENERATE_STREAMLINE_NUM);
}

void StreamlinePool3d::addRandomPool(const int &nums){
    int count = 0;
    vec3f seed;
    srand(time(NULL));
    while(count<nums){
        seed.x = rand()%(100*(vf_w-1))/100.0f;
        seed.y = rand()%(100*(vf_h-1))/100.0f;
        seed.z = rand()%(100*(vf_d-1))/100.0f;
        if(streamlineAtPos3d(seed)) ++count;

		printf("\rTracing Streamlines: #%4d finished.", count);
    }
	printf("\n");
}

void StreamlinePool3d::addPool(Streamline* stls, int nStl, vec3f* pnts, int nPnt){
	memcpy(&stlArray[numStl], stls, sizeof(Streamline)*nStl);
	numStl += nStl;

	memcpy(&pointArray[numPoint], pnts, sizeof(vec3f)*nPnt);

	for (int i=numPoint; i<nPnt+numPoint; ++i){
		vecAtPos3d(veloArray[i], pointArray[i]);
	}
	numPoint += nPnt;
}

void StreamlinePool3d::traceSeeds(vec3f* seeds, int numSeeds){
	for (int i=0; i<numSeeds; ++i) {
		streamlineAtPos3d(seeds[i]);
	}
}

void StreamlinePool3d::recomputeVelos(){
    for(int i=0; i<numPoint; ++i){
        vecAtPos3d(veloArray[i], pointArray[i]);
    }
}


