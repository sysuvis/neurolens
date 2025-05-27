#ifndef CUDA_STREAMLINE_RENDERER_H
#define CUDA_STREAMLINE_RENDERER_H

#include <GL/glew.h>
#include "typeOperation.h"
#include "ColorMap.h"
#include <cuda.h>
#include <cuda_gl_interop.h>
#include "CudaVBOInterop.h"
#include "cudaDeviceMem.h"
#include <iostream>
#include <vector>


//use typedef instead of template for backward compatibility

class cudaStreamlineRenderer{
public:
	typedef vec4f ColorType;
	typedef vec3f VertexType;
	typedef vec3f NormalType;
	typedef vec4i IndexType;

	cudaStreamlineRenderer(Streamline* stls, vec3f* points, const int& num_stls, const int& num_faces=8, const float& radius=1.0f);
	~cudaStreamlineRenderer();

	int getNumPoint(){return mNumPoint;}
	int getNumStreamline(){return mNumStl;}
	vec3f* getPoints_d(){return mPoints_d.data_d;}
	Streamline* getStreamlines_d(){return mStreamlines_d.data_d;}
	
	void updateTubeRadius(const float& radius);
	void updateStreamlineColorVelocity(vec3f* velos, const LinearColorMapType& map_name=COLOR_MAP_PERCEPTUAL);
	void updateTubeColors(const vec4f& color);
	void updateStreamlineColor(const int& streamline_id, const vec4f& color);
	void updateColor(const std::vector<unsigned char>& color_ids, const std::vector<vec4f>& color_map);
	//void updateSegmentColor(const int& streamline_id, const vec4f& color, IndexRange range);
	void updateSegColor(std::vector<StreamlineSegment> SegMask, std::vector<std::vector<int>> ColorMask);
	void updateSegColor(std::vector<StreamlineSegment> edges);

	void drawSingleStreamline(const int& sid, const vec4f& color);
	void drawSingleStreamline(const int& sid);
	void drawStreamlineSegment(const int& sid, const IndexRange& seg);
	void drawStreamlineSegment(const int& sid, const IndexRange& seg, const vec4f& color);
	void drawAllStreamline();	

	void resetIndices();
	void sortQuadByDepth(float* modelview, float* projection);

	void enableRenderProgram();
	void disableRenderProgram();

private:
	int mNumPoint;
	int mNumStl;
	float mRadius;
	int mNumFace;
	int mNumVertices;
	int mNumQuads;
	int mVerticesSize;
	int mNormalsSize;
	int mColorsSize;
	int mIndicesSize;
	int mNormalOffset;
	int mColorOffset;
	int mNumColorComponent;

	cudaVBO mVerticesCuVBO;
	cudaVBO mIndicesCuVBO;
	cudaVBO mColorCuVBO;

	cudaDeviceMem<vec3f> mPoints_d;
	cudaDeviceMem<Streamline> mStreamlines_d;
	Streamline *mStlArray_h;

	GLuint mRenderProgram;
	bool mbUseRenderProgram;
};

#endif //CUDA_STREAMLINE_RENDERER_H