#include <cuda.h>
#include <math.h>
#include "typeOperation.h"
#include "stdio.h"
#include "cudaSelection.h"

__global__ void updateControlPoint(vec3f *vertices, vec3f *points, Streamline *stls, int numStl, int offset, int numFace, float radius){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<numStl){
        int start = stls[idx].start;
        int end = start+stls[idx].numPoint;

        vec3f p1, p2, p3, v1, v2, v3, n;

        float cosv, t;
        v1 = makeVec3f(1.0f, 0.0f, 0.0f);
        p1 = points[start]-v1;

        int i, cur;
        for(cur=start; cur<end; ++cur){
            p2 = points[cur];
            if(cur!=end-1){
                v2 = points[cur+1]-p2;
            } else {
                v2 = v1;
            }
            normalize(v2);
            cosv = v1*v2;
            if(fabs(cosv)>0.999f){
                n = v2;
            } else {
                if(cosv>0.0f){
                    v3 = -v2;
                } else {
                    v3 = v2;
                }
                n = cross(cross(5.0f*v1,5.0f*v3),5.0f*(v1+v3));
            }
            for(i=0; i<numFace; i++){
                if(cur==start){
                    p3 = p1+makeVec3f(0.0f, radius*cosf(6.283185307f*i/numFace), radius*sinf(6.283185307f*i/numFace));
                } else {
                    p3 = vertices[(cur-1)*numFace+i];
                }
                t = (p2-p3)*n/(v1*n);
                p3 = p3+t*v1;
                vertices[cur*numFace+i] = p3;
                p3 = p3-p2;
                normalize(p3);
                vertices[cur*numFace+i+offset] = p3;
            }
            p1 = p2;
            v1 = v2;
        }
    }
}

__global__ void updateColor(vec4f *colors, vec3f *velos, float maxlen, float minlen, vec4f *colorMap, 
							int numPoint, int numFace, int numColor)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx<numPoint){
        //float val = (vec3fLen(velos[idx])-minlen)/(maxlen-minlen)*10.0f;
        float v =  vec3fLen(velos[idx])/maxlen*1.5f;
		v = clamp(v, 0.000001f, 0.999999f);
		v *= (numColor-1);
		int iv = (int)v;
		float f=v-iv;
		vec4f color = colorMap[iv]*(1.0f-f)+colorMap[iv+1]*f;

        for(int i=0; i<numFace; i++){
            colors[idx*numFace+i] = color;
        }
    }
}

__global__ void updateColorBronzeCoin(vec4f *colors, vec3f *velos, float maxlen, float minlen, int numPoint, int numFace){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<numPoint){
        float val = (length(velos[idx])-minlen)/(maxlen-minlen)*255.0f;
        vec4f color;
        float t;
        if(val>200.0f){
            t = (255.0f-val)/55.0f;
			color = (1-t)*makeVec4f(1.0f, 1.0f, 1.0f, 1.0f)+t*makeVec4f(1.0f, 1.0f, 0.0f, 1.0f);
        } else if(val>100.0f){
            t = (200.0f-val)*0.01f;
			color = (1-t)*makeVec4f(1.0f, 1.0f, 0.0f, 1.0f)+t*makeVec4f(1.0f, 0.0f, 0.0f, 1.0f);
        } else {
            t = (100.0f-val)*0.01f;
			color = (1-t)*makeVec4f(1.0f, 0.0f, 0.0f, 1.0f)+t*makeVec4f(0.0f, 0.0f, 0.0f, 1.0f);
        }
        for(int i=0; i<numFace; i++){
			colors[idx*numFace+i] = color;
        }
    }
}

__global__ void updateColorBronzeCoin(vec4f *colors, float *vals, float maxv, float minv, int numPoint, int numFace){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<numPoint){
        float val = (vals[idx]-minv)/(maxv-minv)*255.0f;
        vec4f color;
        float t;
        if(val>200.0f){
            t = (255.0f-val)/55.0f;
            color = (1-t)*makeVec4f(1.0f, 1.0f, 1.0f,1.0f)+t*makeVec4f(1.0f, 1.0f, 0.0f,1.0f);
        } else if(val>100.0f){
            t = (200.0f-val)*0.01f;
            color = (1-t)*makeVec4f(1.0f, 1.0f, 0.0f, 1.0f)+t*makeVec4f(1.0f, 0.0f, 0.0f, 1.0f);
        } else {
            t = (100.0f-val)*0.01f;
            color = (1-t)*makeVec4f(1.0f, 0.0f, 0.0f,1.0f)+t*makeVec4f(0.0f, 0.0f, 0.0f, 1.0f);
        }
        for(int i=0; i<numFace; i++){
            colors[idx*numFace+i] = color;
        }
    }
}

__global__ void updateColorError(vec4f *colors, float *errs, float maxv, int numPoint, int numFace){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<numPoint){
        int offset = 2*numPoint*numFace;
        float t = errs[idx];
        if(t>1.0f) t=1.0f;
        vec4f color;
        if(t>0.5f){
			color = makeVec4f(1.0f, 2.0f - 2.0f*t, 0.0f, 1.0f);
        } else {
			color = makeVec4f(0.5f + 1.0f*t, 0.5f + 1.0f*t, 0.5f + 1.0f*t, 1.0f);
        }
        for(int i=0; i<numFace; i++){
            colors[idx*numFace+i] = color;
        }
    }
}

__global__ void updateColorErrorTransparent(vec4f *colors, float *errs, float maxv, int numPoint, int numFace, float transparency){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx<numPoint){
		float t = errs[idx];
		if(t>1.0f) t=1.0f;
		vec4f color = makeVec4f(0.0f, 1.0f, 0.0f, t*transparency);

		for(int i=0; i<numFace; i++){
			colors[idx*numFace+i] = color;
		}
	}
}

__global__ void updateColorConstant(vec4f *colors, vec4f color, int num_per_thread, int num){
	int block_total = blockDim.x*num_per_thread;
	int start = blockIdx.x*block_total + threadIdx.x;
	int end = min(start + block_total, num);

	vec4f c = color;
	for (int i = start; i < end; i+=blockDim.x) {
		colors[i] = c;
	}
}

__global__ void updateColorIndex(vec4f* colors, unsigned char* color_ids, vec4f* color_map, 
	int num_per_thread, int num_points, int num_faces) 
{
	int block_total = blockDim.x*num_per_thread;
	int start = blockIdx.x*block_total + threadIdx.x;
	int end = min(start + block_total, num_points*num_faces);

	for (int i = start; i < end; i += blockDim.x) {
		colors[i] = color_map[color_ids[i/num_faces]];
	}
}

__global__ void updateColorStrip(vec4f *colors, int step, int start_pid, int end_pid, int numPoint, int numFace){
    int idx = blockIdx.x*blockDim.x + threadIdx.x+start_pid;
    if(idx<end_pid){
        vec4f color = ((idx/step)&1)?makeVec4f(1.0f,1.0f,0.0f,1.0f):makeVec4f(0.0f,0.0f,1.0f,1.0f);
        for(int i=0; i<numFace; i++){
            colors[idx*numFace+i] = color;
        }
    }
}

__global__ void updateIndex(vec4i *indices, Streamline *stls, int numStl, int numFace){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<numStl){
        int start = (stls[idx].start-idx)*numFace;
        int end = start+(stls[idx].numPoint-1)*numFace;
        for(int fid=start, vid=start+idx*numFace; fid<end; ++fid, ++vid){
            if(fid%numFace){
                indices[fid].x = vid;
                indices[fid].y = vid+numFace;
                indices[fid].z = vid-1+numFace;
                indices[fid].w = vid-1;
            } else {
				indices[fid].x = vid;
				indices[fid].y = vid + numFace;
				indices[fid].z = vid + 2 * numFace - 1;
				indices[fid].w = vid + numFace - 1;
            }
        }
    }
}

extern "C" void updateTubeVertexHost(vec3f *vertices, vec3f *points, Streamline *stls, const int &numStl, const int &numPoint, const int &numFace, const float &radius){
    updateControlPoint<<<ceilf(numStl/128.0f), 128>>>(vertices, points, stls, numStl, numPoint*numFace, numFace, radius);
    cudaThreadSynchronize();
}

extern "C" void updateTubeColorHost(vec4f *colors_d, vec3f *velos_d, 
									const float &maxlen, const float &minlen, 
									const int &numPoint, const int &numFace,
									vec4f* color_map_h, const int& numColor)
{
	vec4f *colorMap_d;
    cudaMalloc((void**)&colorMap_d, sizeof(vec4f)*numColor);
    cudaMemcpy(colorMap_d, color_map_h, sizeof(vec4f)*numColor, cudaMemcpyHostToDevice);
    
	updateColor<<<iDivUp(numPoint,128), 128>>>(colors_d, velos_d, maxlen, minlen, colorMap_d, numPoint, numFace, numColor);
    cudaThreadSynchronize();
    
	cudaFree(colorMap_d);
}

extern "C" void updateTubeColorIndexHost(vec4f* colors_d, unsigned char* color_ids_d, vec4f* color_map_d, 
	const int& num_points, const int& num_faces) 
{
	updateColorIndex<<<iDivUp(num_points*num_faces,1024),1024>>>(colors_d, color_ids_d, color_map_d, 16, num_points, num_faces);
	cudaThreadSynchronize();
}

extern "C" void updateTubeColorErrorHost(vec4f *colors, float *errs, const int &numPoint, const int &numFace){
    float *errs_d;
    cudaMalloc((char**)&errs_d, sizeof(float)*numPoint);
    cudaMemcpy(errs_d, errs, sizeof(float)*numPoint, cudaMemcpyHostToDevice);
    updateColorError<<<iDivUp(numPoint,128), 128>>>(colors, errs_d, 1.0f, numPoint, numFace);
    cudaThreadSynchronize();
    cudaFree(errs_d);
}

extern "C" void updateTubeColorErrorTransparentHost(vec4f *colors, float *errs, const int &numPoint, const int &numFace, const float &transparency){
	float *errs_d;
	cudaMalloc((char**)&errs_d, sizeof(float)*numPoint);
	cudaMemcpy(errs_d, errs, sizeof(float)*numPoint, cudaMemcpyHostToDevice);
	updateColorErrorTransparent<<<iDivUp(numPoint,128), 128>>>(colors, errs_d, 1.0f, numPoint, numFace, transparency);
	cudaThreadSynchronize();
	cudaFree(errs_d);
}

extern "C" void updateTubeColorBronzeCoinVelosHost(vec4f *colors, vec3f *velos, const float &maxlen, const float &minlen, const int &numPoint, const int &numFace){
    updateColorBronzeCoin<<<iDivUp(numPoint,128), 128>>>(colors, velos, maxlen, minlen, numPoint, numFace);
    cudaThreadSynchronize();
}

extern "C" void updateTubeColorBronzeCoinValsHost(vec4f *colors, float *vals, const float &maxv, const float &minv, const int &numPoint, const int &numFace){
    updateColorBronzeCoin<<<iDivUp(numPoint,128), 128>>>(colors, vals, maxv, minv, numPoint, numFace);
    cudaThreadSynchronize();
}

extern "C" void updateTubeColorConstantHost(vec4f *colors, vec4f color, const int &start_pid, const int &end_pid, const int &num_faces){
	int num_point = (end_pid - start_pid)*num_faces;
    updateColorConstant<<<iDivUp(num_point,1024),1024>>>(colors+start_pid*num_faces, color, 16, num_point);
    cudaThreadSynchronize();
}

extern "C" void setStreamlineTubeColorStripHost(vec4f *colors, const int &step, const int &start_pid, const int &end_pid, const int &numPoint, const int &numFace){
    updateColorStrip<<<iDivUp(end_pid-start_pid,256),256>>>(colors, step, start_pid, end_pid, numPoint, numFace);
    cudaThreadSynchronize();
}

extern "C" void updateTubeIndexHost(vec4i *indices, Streamline *stls, const int &numStl, const int &numFace){
    updateIndex<<<iDivUp(numStl,128), 128>>>(indices, stls, numStl, numFace);
    cudaThreadSynchronize();
}