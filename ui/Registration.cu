#include <cuda.h>
#include "Registration.h"
#include "typeOperation.h"

__global__ void computeProcrustesDistanceGlobal(vec3f* pts, int numPoint, int winsize, float* retMatrix, int offset){
	int idx = blockDim.x*blockIdx.x+threadIdx.x+offset;

	if (idx<numPoint*numPoint){
		int pid1, pid2;
		pid1 = idx/numPoint;
		pid2 = idx%numPoint;
                if(pid1<pid2) return;
		if(pid1<winsize||pid2<winsize||(pid1+winsize>=numPoint)||(pid2+winsize>=numPoint)){
			return;
		}

		if (pid1==pid2){
			retMatrix[idx] = 0.0f;
		} else {
			retMatrix[idx] = computeProcrustesDistanceWithoutOrder(&pts[pid1-winsize], &pts[pid2-winsize], winsize*2+1);
		}
	}
}

__host__ float* genProcrustesDistanceMatrixHost(Streamline* stls, const int& numStl, vec3f* pts, const int& numPoint, const int& winsize){
	float* mat = new float[numPoint*numPoint];
	float* mat_d;
	int err = 0;
	err |= cudaMalloc((void**)&mat_d, sizeof(float)*numPoint*numPoint);

	vec3f* pts_d;
	err |= cudaMalloc((void**)&pts_d, sizeof(vec3f)*numPoint);
	err |= cudaMemcpy(pts_d, pts, sizeof(vec3f)*numPoint, cudaMemcpyHostToDevice);

	for (int offset=0; offset<numPoint*numPoint; offset+=32768){
		computeProcrustesDistanceGlobal<<<1024,32>>>(pts_d, numPoint, winsize, mat_d, offset);
		cudaThreadSynchronize();
	}

	err |= cudaMemcpy(mat, mat_d, sizeof(float)*numPoint*numPoint, cudaMemcpyDeviceToHost);
	err |= cudaFree(pts_d);
	err |= cudaFree(mat_d);

	for (int i=0; i<numPoint; ++i){
		for (int j=0; j<i; ++j){
			mat[j*numPoint+i] = mat[i*numPoint+j];
		}
	}

	Streamline s;
	int start, end;
	for (int i=0; i<numStl; ++i){
		s = stls[i];
		start = s.start;
		end = s.start+s.numPoint-1;
		for (int j=0, k=start, p=end; j<winsize; ++j, ++k, --p){
			for (int q=1; q<=winsize; ++q){
				mat[(start+winsize)*numPoint+start+winsize-q] = 0.0f;
				mat[(end-winsize)*numPoint+end-winsize+q] = 0.0f;
			}
			for (int q=0; q<numPoint; ++q){
				mat[k*numPoint+q] = mat[(start+winsize)*numPoint+q];
				mat[q*numPoint+k] = mat[q*numPoint+start+winsize];
				mat[p*numPoint+q] = mat[(end-winsize)*numPoint+q];
				mat[q*numPoint+p] = mat[q*numPoint+end-winsize];
			}
		}
	}

	return mat;
}
