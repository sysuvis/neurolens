#ifndef CUDA_VBO_INTEROP
#define CUDA_VBO_INTEROP

#include <GL/glew.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cstdio>

void createVBO(GLuint* vbo, const unsigned int &size, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags);
void deleteVBO(GLuint* vbo, struct cudaGraphicsResource *vbo_res);

class cudaVBO {
public:
	cudaVBO(const unsigned int& _size=0, unsigned int vbo_res_flags=cudaGraphicsMapFlagsWriteDiscard) {
		if (_size > 0) {
			createVBO(&vbo, _size, &res, vbo_res_flags);
			size = _size;
		}
		else {
			vbo = 0;
			res = NULL;
		}
		flags = vbo_res_flags;
	}
	
	~cudaVBO() {
		deleteVBO(&vbo, res);
	}

	void resize(const unsigned int& _size) {
		if (vbo!=0) {
			deleteVBO(&vbo, res);
		}
		createVBO(&vbo, _size, &res, flags);
		size = _size;
	}

	void* map() {
		void* ret;
		size_t num_bytes;
		cudaGraphicsMapResources(1, &res, 0);
		cudaGraphicsResourceGetMappedPointer(&ret, &num_bytes, res);
		return ret;
	}

	void unmap() {
		cudaGraphicsUnmapResources(1, &res, 0);
	}

	void dump(void* ret_h) {
		void *ret_d;
		size_t num_bytes;
		cudaGraphicsMapResources(1, &res, 0);
		cudaGraphicsResourceGetMappedPointer(&ret_d, &num_bytes, res);
		cudaMemcpy(ret_h, ret_d, num_bytes, cudaMemcpyDeviceToHost);
		cudaGraphicsUnmapResources(1, &res, 0);
	}

	void load(void* org_h) {
		void *dst_d;
		size_t num_bytes;
		cudaGraphicsMapResources(1, &res, 0);
		cudaGraphicsResourceGetMappedPointer(&dst_d, &num_bytes, res);
		cudaMemcpy(dst_d, org_h, num_bytes, cudaMemcpyHostToDevice);
		cudaGraphicsUnmapResources(1, &res, 0);
	}

	GLuint getVBO() { return vbo; }

	GLuint vbo;
	cudaGraphicsResource *res;
	unsigned int flags;
	unsigned int size;
};

#endif //CUDA_VBO_INTEROP