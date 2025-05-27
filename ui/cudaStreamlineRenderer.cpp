#include "cudaStreamlineRenderer.h"
#include "CudaVBOInterop.h"
#include "SurfaceSorting.h"
#include "ShaderUtilities.h"

const char *renderVertexShaderSource = STRINGIFY(
#version 450 compatibility\n

out vec4 color;
out vec3 normal;
out vec3 mv_vertex;

void main()
{
	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
	normal = gl_NormalMatrix * gl_Normal;
	mv_vertex = vec3(gl_ModelViewMatrix * gl_Vertex);
	color = gl_Color;
}
);

const char *renderFragmentShaderSource = STRINGIFY(
#version 450 compatibility\n

in vec4 color;
in vec3 normal;
in vec3 mv_vertex;

void main() {
	vec3 n = normalize(normal);
	vec3 l = normalize(mv_vertex - vec3(gl_LightSource[0].position));
	vec3 v = normalize(mv_vertex);

	float dot_nv = dot(n, v);

	vec4 ret_color;
	if (dot_nv<0.1 && dot_nv>-0.1) {
		ret_color = vec4(0.0, 0.0, 0.0, 1.0);
		if (color.w < 0.1f){
			ret_color.w = color.w*10.0f;
		}
	} else {
		vec3 ld = normalize(gl_LightSource[0].spotDirection);

		float diffuse_coefficient = max(0.0, dot(n, -l));
		float specular_coefficient;
		if (dot(l, ld) < 0.95f) {
			specular_coefficient = pow(max(0.0, dot(reflect(-l, n), v)), 0.85f);
		} else {
			specular_coefficient = 0.0;
		}

		ret_color = gl_LightSource[0].ambient*color
			+ diffuse_coefficient*gl_LightSource[0].diffuse*color
			+ specular_coefficient*gl_LightSource[0].specular;

		ret_color.w = color.w;
	}

	gl_FragColor = ret_color;
}
);

extern "C" void updateTubeVertexHost(vec3f *vertices, vec3f *points, Streamline *stls, const int &numStl, const int &numPoint, const int &numFace, const float &radius);
extern "C" void updateTubeIndexHost(vec4i *indices, Streamline *stls, const int &numStl, const int &numFace);
extern "C" void updateTubeColorHost(vec4f *colors_d, vec3f *velos_d, const float &maxlen, const float &minlen, const int &numPoint, const int &numFace, vec4f* color_map, const int& numColor);
extern "C" void updateTubeColorConstantHost(vec4f *colors, vec4f color, const int &start_pid, const int &end_pid, const int &num_faces);
extern "C" void updateTubeColorIndexHost(vec4f* colors_d, unsigned char* color_ids_d, vec4f* color_map_d,
	const int& num_points, const int& num_faces);

cudaStreamlineRenderer::cudaStreamlineRenderer(Streamline* stls, vec3f* points, const int& num_stls, const int& num_faces, const float& radius)
:mNumPoint(stls[num_stls-1].start+stls[num_stls-1].numPoint),
mNumStl(num_stls),
mNumFace(num_faces),
mNumVertices(mNumPoint*num_faces),
mNumQuads((mNumPoint-mNumStl)*mNumFace),
mVerticesSize(sizeof(VertexType)*mNumVertices),
mNormalsSize(sizeof(NormalType)*mNumVertices),
mColorsSize(sizeof(ColorType)*mNumVertices),
mIndicesSize(sizeof(IndexType)*mNumQuads),
mNormalOffset(mVerticesSize),
mColorOffset(mVerticesSize+mNormalsSize),
mNumColorComponent(4),
mVerticesCuVBO(mVerticesSize+mNormalsSize+mColorsSize),
mIndicesCuVBO(mIndicesSize),
mPoints_d(points, mNumPoint),
mStreamlines_d(stls, num_stls),
mStlArray_h(stls),
mRenderProgram(0),
mbUseRenderProgram(false)
{
	updateTubeRadius(0.2);
	resetIndices();
}

cudaStreamlineRenderer::~cudaStreamlineRenderer(){
}

void cudaStreamlineRenderer::updateTubeRadius( const float& radius ){
	mRadius =  radius;
	//compute tube vertices positions
	VertexType* vertices_d = (VertexType*)mVerticesCuVBO.map();
	updateTubeVertexHost(vertices_d, mPoints_d.data_d, mStreamlines_d.data_d, mNumStl, mNumPoint, mNumFace, mRadius);
	mVerticesCuVBO.unmap();
}

void cudaStreamlineRenderer::updateStreamlineColorVelocity(vec3f* velos, const LinearColorMapType& map_name){
	
	// map OpenGL buffer object for writing from CUDA
	char* vertices_d = (char*)mVerticesCuVBO.map();
	vec4f* colors_d = (vec4f*)(&vertices_d[mColorOffset]);
	//copy velocities to gpu
	cudaDeviceMem<vec3f> velos_d(velos, mNumPoint);

	float minv, maxv, len;
	minv = maxv = velos[0] * velos[0];
	for(int i=1; i<mNumPoint; i++){
		len = velos[i]*velos[i];
		if(len>maxv) maxv = len;
		if(len<minv) minv = len;
	}
	maxv = sqrtf(maxv);
	minv = sqrtf(minv);

	std::vector<vec4f> color_map;
	ColorMap::getColorMap(color_map, map_name);

	//call cuda warper here
	updateTubeColorHost(colors_d, velos_d.data_d, maxv, minv, mNumPoint, mNumFace, &color_map[0], color_map.size());
	
	// unmap buffer object
	mVerticesCuVBO.unmap();
}

void cudaStreamlineRenderer::updateTubeColors(const vec4f& color){
	char* vertices_d = (char*)mVerticesCuVBO.map();
	ColorType* colors_d = (ColorType*)(&vertices_d[mColorOffset]);
	updateTubeColorConstantHost(colors_d, color, 0, mNumPoint, mNumFace);
	mVerticesCuVBO.unmap();
}

void cudaStreamlineRenderer::updateStreamlineColor(const int& streamline_id, const vec4f& color){
	char* vertices_d = (char*)mVerticesCuVBO.map();
	ColorType* colors_d = (ColorType*)(&vertices_d[mColorOffset]);
	const Streamline& s = mStlArray_h[streamline_id];
	updateTubeColorConstantHost(colors_d, color, s.start, s.start+s.numPoint, mNumFace);
	mVerticesCuVBO.unmap();
}

void cudaStreamlineRenderer::updateSegColor(std::vector<StreamlineSegment> SegMask,std::vector<std::vector<int>> ColorMask) {
	char* vertices_d = (char*)mVerticesCuVBO.map();
	ColorType* colors_d = (ColorType*)(&vertices_d[mColorOffset]);
	for (auto seg : SegMask) {
		int color_id = ColorMask[seg.streamline_id][seg.segment.lower];
		vec4f color = ColorMap::getD3ColorNoGray(color_id);
		const Streamline& s = mStlArray_h[seg.streamline_id];
		int scale = 20,sample_num= mNumPoint/mNumStl;
		int start = seg.segment.lower > scale ? (seg.segment.lower - scale) : 0;
		int end = (sample_num - seg.segment.upper) > scale ? (seg.segment.upper + scale) : sample_num;
		updateTubeColorConstantHost(colors_d, color, s.start+start, s.start + end, mNumFace);
	}
	
	mVerticesCuVBO.unmap();
}

void cudaStreamlineRenderer::updateSegColor(std::vector<StreamlineSegment> edges) {
	char* vertices_d = (char*)mVerticesCuVBO.map();
	ColorType* colors_d = (ColorType*)(&vertices_d[mColorOffset]);
	for (auto seg : edges) {
		//int color_id = seg.segment.lower;
		vec4f color = ColorMap::getColorByName(ColorMap::Air_Force_blue);
		const Streamline& s = mStlArray_h[seg.streamline_id];
		int scale = 10, sample_num = mNumPoint / mNumStl;
		int segment_start = (seg.segment.lower) % sample_num;
		int segment_end = (seg.segment.upper) % sample_num;
		int start = segment_start > scale ? (segment_start - scale) : 0;
		int end = (sample_num - segment_end) > scale ? (segment_end + scale) : sample_num;
		updateTubeColorConstantHost(colors_d, color, s.start + start, s.start + end, mNumFace);
	}

	mVerticesCuVBO.unmap();
}

void cudaStreamlineRenderer::updateColor(const std::vector<unsigned char>& color_ids, const std::vector<vec4f>& color_map)
{
	char* vertices_d = (char*)mVerticesCuVBO.map();
	ColorType* colors_d = (ColorType*)(&vertices_d[mColorOffset]);

	cudaDeviceMem<unsigned char> color_ids_d(color_ids.data(), color_ids.size());
	cudaDeviceMem<vec4f> color_map_d(color_map.data(), color_map.size());

	updateTubeColorIndexHost(colors_d, color_ids_d.data_d, color_map_d.data_d, mNumPoint, mNumFace);

	mVerticesCuVBO.unmap();
}


void cudaStreamlineRenderer::drawStreamlineSegment(const int& sid, const IndexRange& seg, const vec4f& color){
	if (seg.lower<0) {
		drawSingleStreamline(sid, color);
		return;
	}
	if(sid<0 || sid>=mNumStl || seg.upper<seg.lower) return;
	
	if (mbUseRenderProgram) glUseProgram(mRenderProgram);

	glColor4f(color.x, color.y, color.z, color.w);

	glBindBuffer(GL_ARRAY_BUFFER, mVerticesCuVBO.getVBO());//vertices
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, mIndicesCuVBO.getVBO());//indices
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glNormalPointer(GL_FLOAT, 0, (void*)(mNormalOffset));
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	
	//for each streamline
	glDrawElements(GL_QUADS, 4*(seg.upper-seg.lower)*mNumFace, GL_UNSIGNED_INT, (void*)(sizeof(unsigned int)*4*(mStlArray_h[sid].start+seg.lower-sid)*mNumFace));

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);

	glUseProgram(0);
}

void cudaStreamlineRenderer::drawStreamlineSegment(const int& sid, const IndexRange& seg){
	if (seg.lower<0) {
		drawSingleStreamline(sid);
		return;
	}
	if(sid<0||sid>=mNumStl && seg.upper>seg.lower) return;

	if (mbUseRenderProgram) glUseProgram(mRenderProgram);
	glBindBuffer(GL_ARRAY_BUFFER, mVerticesCuVBO.getVBO());//vertices
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, mIndicesCuVBO.getVBO());//indices
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glNormalPointer(GL_FLOAT, 0, (void*)(mNormalOffset));
	glColorPointer(mNumColorComponent, GL_FLOAT, 0, (void*)(mColorOffset));
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);

	//for each streamline
	glDrawElements(GL_QUADS, 4*(seg.upper-seg.lower)*mNumFace, GL_UNSIGNED_INT, (void*)(sizeof(unsigned int)*4*(mStlArray_h[sid].start+seg.lower-sid)*mNumFace));

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);
	glUseProgram(0);
}

void cudaStreamlineRenderer::drawSingleStreamline(const int& sid, const vec4f& color){
	glColor4f(color.x, color.y, color.z, color.w);

	if(sid<0||sid>=mNumStl) return;

	if (mbUseRenderProgram) glUseProgram(mRenderProgram);
	glBindBuffer(GL_ARRAY_BUFFER, mVerticesCuVBO.getVBO());//vertices
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, mIndicesCuVBO.getVBO());//indices
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glNormalPointer(GL_FLOAT, 0, (void*)(mNormalOffset));
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);

	//for each streamline
	glDrawElements(GL_QUADS, 4*(mStlArray_h[sid].numPoint-1)*mNumFace, GL_UNSIGNED_INT, (void*)(sizeof(unsigned int)*4*(mStlArray_h[sid].start-sid)*mNumFace));

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);

	glUseProgram(0);
}

void cudaStreamlineRenderer::drawSingleStreamline(const int& sid){
	if(sid<0||sid>=mNumStl) return;

	if (mbUseRenderProgram) glUseProgram(mRenderProgram);
	glBindBuffer(GL_ARRAY_BUFFER, mVerticesCuVBO.getVBO());//vertices
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, mIndicesCuVBO.getVBO());//indices
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glNormalPointer(GL_FLOAT, 0, (void*)(mNormalOffset));
	glColorPointer(mNumColorComponent, GL_FLOAT, 0, (void*)(mColorOffset));
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);

	//for each streamline
	glDrawElements(GL_QUADS, 4*(mStlArray_h[sid].numPoint-1)*mNumFace, GL_UNSIGNED_INT, (void*)(sizeof(unsigned int)*4*(mStlArray_h[sid].start-sid)*mNumFace));

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);
	glUseProgram(0);
}

void cudaStreamlineRenderer::drawAllStreamline(){
	if (mbUseRenderProgram) glUseProgram(mRenderProgram);
	glBindBuffer(GL_ARRAY_BUFFER, mVerticesCuVBO.getVBO());//vertices
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, mIndicesCuVBO.getVBO());//indices
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glNormalPointer(GL_FLOAT, 0, (void*)(mNormalOffset));
	glColorPointer(mNumColorComponent, GL_FLOAT, 0, (void*)(mColorOffset));
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);

	glDrawElements(GL_QUADS, 4*(mNumPoint-mNumStl)*mNumFace, GL_UNSIGNED_INT, 0);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);
	glUseProgram(0);
}

void cudaStreamlineRenderer::resetIndices(){
	//compute tube vetices indices
	IndexType* indices_d = (IndexType*)mIndicesCuVBO.map();
	updateTubeIndexHost(indices_d, mStreamlines_d.data_d, mNumStl, mNumFace);
	mIndicesCuVBO.unmap();
}

void cudaStreamlineRenderer::sortQuadByDepth(float* modelview, float* projection) {
	VertexType* vertices_d = (VertexType*)mVerticesCuVBO.map();
	IndexType* indices_d = (IndexType*)mIndicesCuVBO.map();
	cudaSortDeviceQuadSurface(vertices_d, indices_d, mNumQuads, modelview, projection);
	mVerticesCuVBO.unmap();
	mIndicesCuVBO.unmap();
}

void cudaStreamlineRenderer::enableRenderProgram(){
	if (mRenderProgram == 0) {
		mRenderProgram = compileProgram(renderVertexShaderSource, renderFragmentShaderSource);
	}
	mbUseRenderProgram = true;
}

void cudaStreamlineRenderer::disableRenderProgram(){
	mbUseRenderProgram = false;
}
