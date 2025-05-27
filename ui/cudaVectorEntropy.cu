#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include "typeOperation.h"
#include "definition.h"

#define PI 3.1415926f

typedef struct{
    float theta1, theta2;
    short startIdx;
    float avgAngle;
} RegInfoCollarItem;

typedef struct{
    RegInfoCollarItem *collarInfo;
    int numCollar;
    int numRegion;
} RegInfoItem;

__host__ void partitionUnitSphere(RegInfoItem &regInfo){
    float spharea = 4*PI*1*1;	//r == 1
    float vr = spharea/(float)regInfo.numRegion;
    float thetac;
    float deltai;
    float deltaf;
    float ni;
    int n;

    if(regInfo.numRegion == 1){	//only single region
        regInfo.numCollar = 1;
        regInfo.collarInfo = new RegInfoCollarItem[1];
        regInfo.collarInfo[0].theta1 = 0;
        regInfo.collarInfo[0].theta2 = PI;
        regInfo.collarInfo[0].startIdx = 0;
        regInfo.collarInfo[0].avgAngle = 2*PI;
    } else {
        //step 1:determine the colatitude of cap
        thetac = asin(sqrt(vr/(4*PI)))*2;

        //step2: determine ideal collar angle
        deltai = sqrt(vr);

        //step3: determine the ideal collar number
        ni = (PI - 2*thetac)/deltai;

        //step4: determine the actual collar number
        int n2 = floor(ni + 0.5);
        if(n2 > 1)
            n = n2;
        else
            n = 1;
        regInfo.numCollar = n+2;
        regInfo.collarInfo = new RegInfoCollarItem[regInfo.numCollar];

        //step5: creat list of ideal number of regions of each collar
        deltaf = ni/(float)n*deltai;
        float *thetaF;
        thetaF = new float[n + 1];
        for(int i = 0; i <n+1; i++){
            thetaF[i] = thetac + i*deltaf;
        }

        float *y;
        y = new float[n+1];
        for(int i = 1; i <= n; i++){
            y[i] = (4*PI*sin(thetaF[i]/2)*sin(thetaF[i]/2) - 4*PI*sin(thetaF[i-1]/2)*sin(thetaF[i-1]/2))/vr;
        }

        //step6: creat list of actual number of regions of each collar
        float *m;
        m = new float[n+1];
        float *a;
        a = new float[n+1];
        a[0] = 0;
        int max= -1;	// this is used to store the largest number of regions of the collar
        for(int i = 1; i <= n; i++){
            m[i] = floor(y[i] + a[i-1] + 0.5);
            float sum = 0 ;
            for(int j = 1; j <= i; j++)
                sum += y[j] - m[j];
            a[i] = sum;
            if(m[i]>=max)
                max = m[i];
        }

        //step7: creat list of colatitudes for each zone
        float *theta;
        theta = new float[n+3];
        theta[0] = 0;
        theta[n+2] = PI;
        for(int i = 1; i < n+2; i++){
            float sum = 0;
            for(int j = 1; j <= i-1; j++){
                sum += m[j];
            }
            sum += 1;
            float sumarea = sum*vr;
            theta[i] = asin(sqrt(sumarea/(4*PI)))*2;
        }

        float sum = 2 ;
        for(int i = 1; i <= n; i++)
            sum += m[i];

        //step8: partition each collar and store the final region information into the array ra;
        regInfo.collarInfo[0].theta1 = theta[0];
        regInfo.collarInfo[0].theta2 = theta[1];
        regInfo.collarInfo[0].startIdx = 0;
        regInfo.collarInfo[0].avgAngle = 2.0f*PI;
        int currIdx = 1;
        for(int i = 1; i <= n; i++){	//for each collar, we partition it into m[i] regions
            regInfo.collarInfo[i].theta1 = theta[i];
            regInfo.collarInfo[i].theta2 = theta[i+1];
            regInfo.collarInfo[i].startIdx = currIdx;
            regInfo.collarInfo[i].avgAngle = (2.0f*PI)/m[i];

            currIdx += m[i];
        }

        regInfo.collarInfo[n+1].theta1 = theta[n+1];
        regInfo.collarInfo[n+1].theta2 = theta[n+2];
        regInfo.collarInfo[n+1].startIdx = currIdx;
        regInfo.collarInfo[n+1].avgAngle = 2.0*PI;

        delete[] m;
        delete[] a;
        delete[] theta;
    }
}

__device__ short computeSphereBin(vec3f vec, RegInfoItem regInfo){

    float theta, gama;

    float len = vec.x*vec.x+vec.z*vec.z;

    float arcsin = asin(vec.z/sqrtf(len));
    float arcsin2 = asin(vec.y/sqrtf(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z));
    if(arcsin>=0){
        if(vec.x >= 0)
            gama = arcsin;
        else
            gama =  PI - arcsin;
    }
    else{
        if(vec.x >= 0)
            gama = 2.0*PI + arcsin;
        else
            gama = PI - arcsin;
    }

    theta = PI/2.0 - arcsin2;

    for(int i = 0; i < regInfo.numCollar; i++){
        if(theta>=regInfo.collarInfo[i].theta1&& theta<regInfo.collarInfo[i].theta2){
            short ret = (short)regInfo.collarInfo[i].startIdx+(short)(gama/regInfo.collarInfo[i].avgAngle);
            if(ret>=regInfo.collarInfo[regInfo.numCollar-1].startIdx){
                ret=(short)regInfo.collarInfo[regInfo.numCollar-1].startIdx-1;
            } else if(ret>=regInfo.collarInfo[i+1].startIdx){
                ret=(short)regInfo.collarInfo[i+1].startIdx-1;
            }
            return ret;
        }
    }
    return 0;// to avoid segmentation fault
}

__global__ void compute3dBins(vec3f *velos, short *binlist, RegInfoItem regInfo, int numMagBin, int numTotalPoints, float maxlen, float minlen){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;

    //copy regionInfo from global to share
    extern __shared__ RegInfoCollarItem collars[];
    if(threadIdx.x<regInfo.numCollar){
        collars[threadIdx.x] = regInfo.collarInfo[threadIdx.x];
    }
    regInfo.collarInfo = collars;
    __syncthreads();


    if(idx<numTotalPoints){
        short sphereBin, magBin;
        vec3f vec = velos[idx];
        float len = (vec3fLen(vec)-minlen)/(maxlen-minlen);
        if( len>0.9999f){
            magBin = numMagBin-1;
        } else {
            magBin = (short)(len*numMagBin);
        }

        sphereBin = computeSphereBin(vec, regInfo);
        binlist[idx] = sphereBin*(short)numMagBin+magBin;
    }
}

__global__ void initLogArray(float *log_arr, int size){
    int idx = blockDim.x*blockIdx.x+threadIdx.x;

    if(idx<size){
        log_arr[idx] = logf((float)idx);
    }
}

__global__ void computeVecFieldEntropy(float *entropy, short *bins, float* log_arr, int offset, int w, int h, int d, int winSize, int numBins){
    int idx = blockDim.x*blockIdx.x+threadIdx.x+offset;

    if(idx<w*h*d){
        int xup, xlow, yup, ylow, zup, zlow;
        int x = idx%w;
        int y = (idx%(w*h))/w;
        int z = idx/(w*h);
        int hist[NUM_MAG_BIN*NUM_SPHERE_BIN];

        for(int i=0; i<numBins; i++) hist[i]=0;

        xup = (x+winSize<w)?(x+winSize):(w-1);
        xlow = (x-winSize>=0)?(x-winSize):0;
        yup = (y+winSize<h)?(y+winSize):(h-1);
        ylow = (y-winSize>=0)?(y-winSize):0;
        zup = (z+winSize<d)?(z+winSize):(d-1);
        zlow = (z-winSize>=0)?(z-winSize):0;

        //histogram
        for(z=zlow; z<=zup; z++){
            for(y=ylow; y<=yup; y++){
                for(x=xlow; x<=xup; x++){
                    hist[bins[z*w*h+y*w+x]]++;
                }
            }
        }

        //entropy
        int count=(zup-zlow+1)*(yup-ylow+1)*(xup-xlow+1);
        float tmp = 0.0f;
        for(int i=0; i<numBins; i++){
            if(hist[i]!=0){
                tmp += hist[i]*log_arr[hist[i]];
            }
        }
        tmp = -(tmp/count)+log_arr[count];
        if(tmp<0.000000001f){
            entropy[idx] = 0.0f;
        } else {
            entropy[idx] = tmp;
        }
    }
}

extern "C" void compute3dBinsHost(vec3f *velos, short* bins, const int &numPoint, const int &numMagBin, const int &numSphereBin){
    //initialize region info for sphere bin computation
    RegInfoItem regInfo_h = {NULL, 0, numSphereBin};
    partitionUnitSphere(regInfo_h);
    RegInfoItem regInfo_d = {NULL, regInfo_h.numCollar, regInfo_h.numRegion};
	int err = 0;
    err |= cudaMalloc((void**)&(regInfo_d.collarInfo), sizeof(RegInfoCollarItem)*regInfo_d.numCollar);
    err |= cudaMemcpy(regInfo_d.collarInfo, regInfo_h.collarInfo, sizeof(RegInfoCollarItem)*regInfo_d.numCollar, cudaMemcpyHostToDevice);

    //find max and min
    float maxlen = vec3fLen(velos[0]);
    float minlen = maxlen;
    float len;
    for (int i=1; i<numPoint; i++){
        len = vec3fLen(velos[i]);
        if (len>maxlen) maxlen = len;
        if (len<minlen) minlen = len;
    }

    //copy velocities
    vec3f *velos_d;
    err |= cudaMalloc((void**)&velos_d, sizeof(vec3f)*numPoint);
    err |= cudaMemcpy(velos_d, velos, sizeof(vec3f)*numPoint, cudaMemcpyHostToDevice);

    //allocate memory to store binlist
    short *bins_d;
    err |= cudaMalloc((void**)&bins_d, sizeof(short)*numPoint);
    //compute bins
    int collarInfoSize = regInfo_d.numCollar*sizeof(RegInfoCollarItem);
    compute3dBins<<<ceilf(numPoint/256.0f), 256, collarInfoSize>>>(velos_d, bins_d, regInfo_d, numMagBin, numPoint, maxlen, minlen);
    err |= cudaThreadSynchronize();
    err |= cudaMemcpy(bins, bins_d, sizeof(short)*numPoint, cudaMemcpyDeviceToHost);

    //clean
    delete[] regInfo_h.collarInfo;
    err |= cudaFree(velos_d);
    err |= cudaFree(bins_d);
    err |= cudaFree((regInfo_d.collarInfo));
}

extern "C" void computeVecFieldEntropyHost(float *entropy, vec3f *vecf, const int &w, const int &h, const int &d, const int &winSize){
    int numPoint = w*h*d;

    //initialize region info for sphere bin computation
    RegInfoItem regInfo_h = {NULL, 0, NUM_SPHERE_BIN};
    partitionUnitSphere(regInfo_h);
    RegInfoItem regInfo_d = {NULL, regInfo_h.numCollar, regInfo_h.numRegion};
    cudaMalloc((void**)&(regInfo_d.collarInfo), sizeof(RegInfoCollarItem)*regInfo_d.numCollar);
    cudaMemcpy(regInfo_d.collarInfo, regInfo_h.collarInfo, sizeof(RegInfoCollarItem)*regInfo_d.numCollar, cudaMemcpyHostToDevice);

    //find max and min
    float maxlen = vec3fLen(vecf[0]);
    float minlen = maxlen;
    float len;
    for (int i=1; i<numPoint; i++){
        len = vec3fLen(vecf[i]);
        if (len>maxlen) maxlen = len;
        if (len<minlen) minlen = len;
    }

    //copy velocities
    vec3f *vecf_d;
    cudaMalloc((void**)&vecf_d, sizeof(vec3f)*numPoint);
    cudaMemcpy(vecf_d, vecf, sizeof(vec3f)*numPoint, cudaMemcpyHostToDevice);

    //allocate memory to store binlist
    short *bins_d;
    cudaMalloc((void**)&bins_d, sizeof(short)*numPoint);
    //compute bins
    int collarInfoSize = regInfo_d.numCollar*sizeof(RegInfoCollarItem);
    compute3dBins<<<ceilf(numPoint/256.0f), 256, collarInfoSize>>>(vecf_d, bins_d, regInfo_d, NUM_MAG_BIN, numPoint, maxlen, minlen);
    //clean
    delete[] regInfo_h.collarInfo;
    cudaFree(vecf_d);
    cudaFree((regInfo_d.collarInfo));

    float *log_arr_d;
    int logSize = (winSize*2+1)*(winSize*2+1)*(winSize*2+1)+1;
    cudaMalloc((void**)&log_arr_d, sizeof(float)*logSize);
    initLogArray<<<ceilf(logSize/256.0f),256>>>(log_arr_d, logSize);

    float *entropy_d;
    cudaMalloc((void**)&entropy_d, sizeof(float)*numPoint);
    for(int i=0; i<ceilf(numPoint/524288.0f); i++){
        computeVecFieldEntropy<<<16384,32>>>(entropy_d, bins_d, log_arr_d, i*524288, w, h, d, winSize, NUM_MAG_BIN*NUM_SPHERE_BIN);
        cudaThreadSynchronize();
    }
    cudaMemcpy(entropy, entropy_d, sizeof(float)*numPoint, cudaMemcpyDeviceToHost);


    //clean
    cudaFree(bins_d);
    cudaFree(log_arr_d);
    cudaFree(entropy_d);
}