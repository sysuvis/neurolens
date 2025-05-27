#include "definition.h"
#include "typeOperation.h"
#include "VectorEntropy.h"
#include <math.h>
#include <string>
#include <stdio.h>
#include <fstream>

#define abs(x) ((x>0)?(x):(-x))

extern "C" void compute3dBinsHost(vec3f *velos, short* bins, const int &numPoint, const int &numMagBin, const int &numSphereBin);
extern "C" void computeVecFieldEntropyHost(float *entropy, vec3f *vecf, const int &w, const int &h, const int &d, const int &winSize);

void entropyForStreamlinePool(StreamlinePool3d *pool, float* entropy){
	short *bins = new short[pool->numPoint];
	compute3dBinsHost(pool->veloArray, bins, pool->numPoint, NUM_MAG_BIN, NUM_SPHERE_BIN);

	float log2 = logf(2.0f), tmp;
	int hist[NUM_3D_BIN];
	for (int i=0; i<pool->numStl; ++i) {
		memset(hist, 0, sizeof(int)*NUM_3D_BIN);
		Streamline s = pool->stlArray[i];
		short *b = &bins[s.start];
		for (int j=0; j<s.numPoint; ++j) {
			++hist[b[j]];
		}
		tmp = 0.0f;
		for(int j=0; j<NUM_3D_BIN; j++){
			if(hist[j]!=0){
				tmp += hist[j]*logf(hist[j]);
			}
		}
		entropy[i] = (-(tmp/s.numPoint)+logf(s.numPoint))/log2;
	}
}

void entropyFromStreamlinePool(StreamlinePool3d *pool, float *entropy, const int &numx, const int &numy, const int &numz){
    //grid size
    float gridx = (pool->getVfWidth()-1.0f)/numx;
    float gridy = (pool->getVfHeight()-1.0f)/numy;
    float gridz = (pool->getVfDepth()-1.0f)/numz;
    int total = numx*numy*numz;
    int numxy = numx*numy;

    //compute bins
    short *bins = new short[pool->numPoint];
    compute3dBinsHost(pool->veloArray, bins, pool->numPoint, NUM_MAG_BIN, NUM_SPHERE_BIN);

    //compute hist
    int (*hist)[NUM_3D_BIN] = new int[total][NUM_3D_BIN];
    int *pcount = new int[total];
    memset(hist, 0, sizeof(int)*total*NUM_3D_BIN);
    memset(pcount, 0, sizeof(int)*total);
    int gid, xid, yid, zid;
    for(int i=0; i<pool->numPoint; i++){
        xid = pool->pointArray[i].x/gridx;
        if(xid>=numx) xid = numx-1;
		else if(xid<0) xid = 0;
        yid = pool->pointArray[i].y/gridy;
        if(yid>=numy) yid = numy-1;
		else if(yid<0) yid = 0;
        zid = pool->pointArray[i].z/gridz;
        if(zid>=numz) zid = numz-1;
		else if(zid<0) zid = 0;
        gid = zid*numxy+yid*numx+xid;
        hist[gid][bins[i]]++;
        pcount[gid]++;
    }

    //compute entropy
    memset(entropy, 0, sizeof(float)*total);
    float tmp;
    float log2 = logf(2.0f);
    for(int i=0; i<total; i++){
        if (pcount[i]==0) continue;
        tmp = 0.0f;
        for(int j=0; j<NUM_3D_BIN; j++){
            if(hist[i][j]!=0){
                tmp += hist[i][j]*logf(hist[i][j]);
            }
        }
        entropy[i] = (-(tmp/pcount[i])+logf(pcount[i]))/log2;
    }

    //normalize
    float maxv = entropy[0];
    for(int i=1; i<total; i++){
        if(entropy[i]>maxv) maxv = entropy[i];
    }
    for(int i=0; i<total; i++){
        entropy[i] /= maxv;
    }

    delete[] hist;
    delete[] pcount;
    delete[] bins;
}

void entropyFromVectorFieldSequential(float *entropy, vec3f *vecf, const int &w, const int &h, const int &d, const int &def_w, const int &def_h, const int &def_d){
    //compute bins
    short *bins = new short[w*h*d];
    compute3dBinsHost(vecf, bins, w*h*d, NUM_MAG_BIN, NUM_SPHERE_BIN);

    float *tmp_entropy = new float[w*h*d];
    //compute entropy
    int *hist = new int[NUM_MAG_BIN*NUM_SPHERE_BIN];

    int logSize = (ENTROPY_WIN_SIZE*2+1)*(ENTROPY_WIN_SIZE*2+1)*(ENTROPY_WIN_SIZE*2+1)+1;
    float *log_arr = new float[logSize];
    for(int i=0; i<logSize; i++){
        log_arr[i] = logf((float)i);
    }
    int xup, xlow, yup, ylow, zup, zlow;
    for(int z=0; z<d; z++){
        for(int y=0; y<h; y++){
            for(int x=0; x<w; x++){
                xup = (x+ENTROPY_WIN_SIZE<w)?(x+ENTROPY_WIN_SIZE):(w-1);
                xlow = (x-ENTROPY_WIN_SIZE>=0)?(x-ENTROPY_WIN_SIZE):0;
                yup = (y+ENTROPY_WIN_SIZE<h)?(y+ENTROPY_WIN_SIZE):(h-1);
                ylow = (y-ENTROPY_WIN_SIZE>=0)?(y-ENTROPY_WIN_SIZE):0;
                zup = (z+ENTROPY_WIN_SIZE<d)?(z+ENTROPY_WIN_SIZE):(d-1);
                zlow = (z-ENTROPY_WIN_SIZE>=0)?(z-ENTROPY_WIN_SIZE):0;

                //histogram
                memset(hist, 0, sizeof(int)*NUM_MAG_BIN*NUM_SPHERE_BIN);
                int count=0;
                for(int i=zlow; i<=zup; i++){
                    for(int j=ylow; j<=yup; j++){
                        for(int k=xlow; k<=xup; k++){
                            hist[bins[i*w*h+j*w+k]]++;
                            count++;
                        }
                    }
                }

                //entropy
                float tmp = 0.0f;
                for(int i=0; i<NUM_MAG_BIN*NUM_SPHERE_BIN; i++){
                    if(hist[i]!=0){
                        tmp += hist[i]*log_arr[hist[i]];
                    }
                }
                tmp_entropy[z*w*h+y*w+x] = -(tmp/count)+log_arr[count];
            }
        }
    }

    scaleSalarFieldSizeDown(entropy, tmp_entropy, def_w, def_h, def_d, w, h, d);

    //normalize
    float maxv = entropy[0], minv = entropy[0];
    for(int i=1; i<def_w*def_h*def_d; i++){
        if(entropy[i]>maxv) maxv = entropy[i];
        if(entropy[i]<minv) minv = entropy[i];
    }
    //printf("sequential %8.5f, min %8.5f\n", maxv, minv);
    for(int i=0; i<def_w*def_h*def_d; i++){
        entropy[i] = (entropy[i]-minv)/(maxv-minv);
    }

    //clean
    delete[] log_arr;
    delete[] tmp_entropy;
    delete[] hist;
    delete[] bins;
}

void entropyFromVectorFieldSequential(float *entropy, vec3f *vecf, const int &w, const int &h, const int &d){
    //compute bins
    short *bins = new short[w*h*d];
    compute3dBinsHost(vecf, bins, w*h*d, NUM_MAG_BIN, NUM_SPHERE_BIN);

    //compute entropy
    int *hist = new int[NUM_MAG_BIN*NUM_SPHERE_BIN];

    int logSize = (ENTROPY_WIN_SIZE*2+1)*(ENTROPY_WIN_SIZE*2+1)*(ENTROPY_WIN_SIZE*2+1)+1;
    float *log_arr = new float[logSize];
    for(int i=0; i<logSize; i++){
        log_arr[i] = logf((float)i);
    }

    int wh = w*h;

    int xup, xlow, yup, ylow, zup, zlow;
    for(int z=0; z<d; z++){
        for(int y=0; y<h; y++){
            for(int x=0; x<w; x++){
                xup = (x+ENTROPY_WIN_SIZE<w)?(x+ENTROPY_WIN_SIZE):(w-1);
                xlow = (x-ENTROPY_WIN_SIZE>=0)?(x-ENTROPY_WIN_SIZE):0;
                yup = (y+ENTROPY_WIN_SIZE<h)?(y+ENTROPY_WIN_SIZE):(h-1);
                ylow = (y-ENTROPY_WIN_SIZE>=0)?(y-ENTROPY_WIN_SIZE):0;
                zup = (z+ENTROPY_WIN_SIZE<d)?(z+ENTROPY_WIN_SIZE):(d-1);
                zlow = (z-ENTROPY_WIN_SIZE>=0)?(z-ENTROPY_WIN_SIZE):0;

                //histogram
                memset(hist, 0, sizeof(int)*NUM_MAG_BIN*NUM_SPHERE_BIN);
                int count=0;
                for(int i=zlow; i<=zup; i++){
                    for(int j=ylow; j<=yup; j++){
                        for(int k=xlow; k<=xup; k++){
                            hist[bins[i*wh+j*w+k]]++;
                            count++;
                        }
                    }
                }

                //entropy
                float tmp = 0.0f;
                for(int i=0; i<NUM_MAG_BIN*NUM_SPHERE_BIN; i++){
                    if(hist[i]!=0){
                        tmp += hist[i]*log_arr[hist[i]];
                    }
                }
                entropy[z*wh+y*w+x] = -(tmp/count)+log_arr[count];
            }
        }
    }

    //normalize
    float maxv = entropy[0], minv = entropy[0];
    for(int i=1; i<wh*d; i++){
        if(entropy[i]>maxv) maxv = entropy[i];
        if(entropy[i]<minv) minv = entropy[i];
    }
    //printf("sequential %8.5f, min %8.5f\n", maxv, minv);
    for(int i=0; i<wh*d; i++){
        entropy[i] = (entropy[i]-minv)/(maxv-minv);
    }

    //clean
    delete[] log_arr;
    delete[] hist;
    delete[] bins;
}

void entropyFromVectorField(float *entropy, vec3f *vecf, const int &w, const int &h, const int &d, const int &def_w, const int &def_h, const int &def_d){
    float *tmp_entropy = new float[w*h*d];
    computeVecFieldEntropyHost(tmp_entropy, vecf, w, h, d, ENTROPY_WIN_SIZE);
    //for(int i=0; i<w*h*d; i++) printf("entropy[%5d,%5d]=%8.5f\n", i, w*h*d, tmp_entropy[i]);
    float maxt = tmp_entropy[0], mint = tmp_entropy[0];
    int minid = 0;
    for(int i=1; i<w*h*d; i++){
        if(tmp_entropy[i]>maxt) maxt = tmp_entropy[i];
        if(tmp_entropy[i]<mint){ mint = tmp_entropy[i]; minid =i;}
    }
    printf("cuda entropy_tmp: max %8.5f, min %8.5f, minId%5d\n", maxt, mint, minid);
    scaleSalarFieldSizeDown(entropy, tmp_entropy, def_w, def_h, def_d, w, h, d);
    //for(int i=0; i<def_w*def_h*def_d; i++) printf("entropy[%5d,%5d]=%8.5f\n", i, def_w*def_h*def_d, entropy[i]);

    //normalize
    float maxv = entropy[0], minv = entropy[0];
    for(int i=1; i<def_w*def_h*def_d; i++){
        if(entropy[i]>maxv) maxv = entropy[i];
        if(entropy[i]<minv) minv = entropy[i];
    }
    printf("cuda entropy: max %8.5f, min %8.5f\n", maxv, minv);
    for(int i=0; i<def_w*def_h*def_d; i++){
        entropy[i] = (entropy[i]-minv)/(maxv-minv);
    }

    //clean
    delete[] tmp_entropy;
}

void entropyFromVectorField(float *entropy, vec3f *vecf, const int &w, const int &h, const int &d){
    computeVecFieldEntropyHost(entropy, vecf, w, h, d, ENTROPY_WIN_SIZE);

    //normalize
    float maxv = entropy[0], minv = entropy[0];
    for(int i=1; i<w*h*d; i++){
        if(entropy[i]>maxv) maxv = entropy[i];
        if(entropy[i]<minv) minv = entropy[i];
    }
    printf("cuda entropy: max %8.5f, min %8.5f\n", maxv, minv);
    for(int i=0; i<w*h*d; i++){
        entropy[i] = (entropy[i]-minv)/(maxv-minv);
    }
}

void scaleSalarFieldSizeDown(float *dst, float *src, const int &dst_w, const int &dst_h, const int &dst_d,\
                       const int &src_w, const int &src_h, const int &src_d)
{
    if(src_w==dst_w && src_h==dst_h && src_d==dst_d){
        memcpy(dst, src, sizeof(float)*dst_w*dst_h*dst_d);
        return;
    }
    int ylow, xlow, zlow, yup, xup, zup;
    float xlen = (src_w-1)/dst_w;
    float ylen = (src_h-1)/dst_h;
    float zlen = (src_d-1)/dst_d;
    float x, y;
    float z = zlen*0.5f;

    memset(dst, 0, sizeof(float)*dst_w*dst_h*dst_d);

    for(int i=0; i<dst_d; i++, z+=zlen){
        zlow = ceilf(z-zlen*0.5f-0.1f);
        zup = floorf(z+zlen*0.5f+0.1f);
        y = ylen*0.5f;
        for (int j=0; j<dst_h; j++, y+=ylen) {
            ylow = ceilf(y-ylen*0.5f-0.1f);
            yup = floorf(y+ylen*0.5f+0.1f);
            x = xlen*0.5f;
            for (int k=0; k<dst_w; k++, x+=xlen){
                xlow = ceilf(x-xlen*0.5f-0.1f);
                xup = floorf(x+xlen*0.5f+0.1f);
                for(int r=zlow; r<=zup; r++){
                    for(int p=ylow; p<=yup; p++){
                        for(int q=xlow; q<xup; q++){
                            dst[i*dst_w*dst_h+j*dst_w+k] += src[r*src_w*src_h+p*src_w+q];
                        }
                    }
                }
                dst[i*dst_w*dst_h+j*dst_w+k] /= (zup-zlow+1)*(yup-ylow+1)*(xup-xlow+1);
            }
        }
    }
}