#include "CudaVBOInterop.h"
#include <cuda_runtime.h>
#include "DisplayWidget.h"

void createVBO(GLuint* vbo, const unsigned int &size, 
				struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags)
{
	*vbo = 0;
	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));
}

void deleteVBO(GLuint* vbo, struct cudaGraphicsResource *vbo_res){
	if (vbo) {
		// unregister this buffer object with CUDA
		cudaGraphicsUnregisterResource(vbo_res);

		glBindBuffer(1, *vbo);
		glDeleteBuffers(1, vbo);

		*vbo = 0;
	}
}