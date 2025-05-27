#pragma once

#include "VolumeData.h"
#include "NCData.h"
#include <cuda.h>
#include <curand_kernel.h>
#include "CudaVBOInterop.h"
#include "cudaDeviceMem.h"
#include "EarthModel.h"
#include "RandomSample.h"
#include "cudaSelection.h"
#include "earth.h"

extern "C"
void cudaEarthTrace_h(vec4f* points_d, vec4f* ret_points, int total, int num_per_thread, int num_steps, 
	float w_scale, vec3f dim, float t, float delta_t, float interval);

extern "C"
void generateRandPoints_h(vec4f* points, int total, int num_per_thread, vec3i dim);

extern "C"
void updatePointsRand_h(vec4f* points, int total, int num_per_thread, vec3i dim, float t_upper, float t_lower);

extern "C"
void cudaEarthUpdateVectorFieldTex_h(cudaArray* vec_time_prev, cudaArray* vec_time_next);

extern "C"
void cudaEarthUpdateIndex_h(int* indices_d, int current_point, int num_per_thread, int num_point_per_particle, int num_particle);

extern "C"
void cudaEarthSetIndex_h(int* indices_d, int num_per_thread, int num_point_per_particle, int num_particle);

extern "C"
void earthUpdatePointsRandSampleEntries_h(vec4f* points_d, vec2f* triangle_vertices_d, vec3i* triangles_d,
	int* sample_entries, int num_sample_entries, float z_up,
	int total, int num_per_thread, float t_upper, float t_lower);

extern "C"
void earthBlockParticleCount_h(int* count_d, vec4f* points_d,
	vec2i grid_dim, vec2f block_size, int num_per_thread, int num_particles);

extern "C"
void earthBlockImportance_h(float* importance_d, int* count_d, float* area_d,
	float default_count, int num_per_thread, int num_blocks);

extern "C"
curandState* earthCreateRNG(const int& total);

extern "C"
void earthRecycleParticles_h(vec4f* points_d, int num_particles,
	const int& current_point, const float& z_up, const float& z_low, vec2i& grid_dim, vec2f& block_size,
	NUSBItem* distributions_d, const int& num_bitems, const float& t_up, const float& t_low,
	float* importance_d, const float& imp_up, const float& imp_low, const float& curv_weight,
	const int& prev_point_offset, const int& next_point_offset, curandState* states_d);

class cudaEarthTracer {
public:
	cudaEarthTracer(const std::string& u_file_name, const std::string& v_file_name, const std::string& w_file_name,
		const std::string& u_var_name="U", const std::string& v_var_name="V", const std::string& w_var_name="W",
		const std::string& lat_var_name = "lat", const std::string& lon_var_name = "lon", const std::string& lev_var_name = "lev") :
		u(NCSliceData(u_file_name, u_var_name)),
		v(NCSliceData(v_file_name, v_var_name)),
		w(NCSliceData(w_file_name, w_var_name)),
		dim(makeVec3i(u.getSliceDimension(3), u.getSliceDimension(2), u.getSliceDimension(1))),
		u_data(VolumeData<float>(dim.x, dim.y, dim.z)),
		v_data(VolumeData<float>(dim.x, dim.y, dim.z)),
		w_data(VolumeData<float>(dim.x, dim.y, dim.z)),
		vf_data(VolumeData<vec4f>(dim.x, dim.y, dim.z)),
		trace_interval(0.05f),
		num_timestep(u.getSliceDimension(0)),
		current_time(0),
		current_trace(0),
		current_point(0),
		num_step_per_trace(2),
		num_trace_per_time_step(64),
		num_particle_per_thread(64),
		num_points_per_particle(16),
		num_particles(1<<12),
		max_num_particles(1<<16),
		//num_points_per_particle(10),
		//num_particles(100),
		//max_num_particles(100),
		earth_step_size(5.0f),
		w_scale(70.0f),
		z_up(dim.z-1),
		z_low(0.0f),
		recycle_max_imp_scale(0.1f),
		recycle_min_imp_scale(0.0f),
		recycle_curvature_weight(30.0f),
		recycle_max_time_scale(40.0f),
		recycle_min_time_scale(4.0f),
		particles_cuvbo(num_points_per_particle*max_num_particles *sizeof(vec4f)),
		streamlet_indices_cuvbo((2*num_points_per_particle-1)*max_num_particles*sizeof(int))
	{
		vec4f* vf_data_array = vf_data.getData();
		for (int i = 0; i < u_data.volumeSize(); ++i) {
			vf_data_array[i].z = 0.0f;
			vf_data_array[i].w = 0.0f;
		}

		prev_vec = allocateCudaVectorField(dim);
		next_vec = allocateCudaVectorField(dim);

		NCSliceData lat_nc(u_file_name, lat_var_name);
		NCSliceData lon_nc(u_file_name, lon_var_name);
		NCSliceData lev_nc(u_file_name, lev_var_name);

		get1DData(lat, lat_nc);
		get1DData(lon, lon_nc);
		lon.push_back(lon[0] + 360.0f);
		get1DData(lev, lev_nc);

		float dtor = 0.01745329252f;
		pos_offset = dtor*makeVec3f(lon[0], lat[0], 0.0f);
		float lon_step = lon[1] - lon[0];
		float lat_step = lat[1] - lat[0];
		float lev_step = 1.0f / (lev.size() - 1);
		pos_to_radian = makeVec3f(dtor*lon_step, dtor*lat_step, lev_step);

		loadTimeStep(7);
		initTracer();
	};

	~cudaEarthTracer() {
		if (particles_RNG != NULL) {
			cudaFree(particles_RNG);
		}
	}

	void get1DData(std::vector<float>& arr, NCSliceData& nc_data) {
		int n = nc_data.getSliceDimension(0);
		std::vector<double> arr_d(n);
		nc_data.get1DData(&arr_d[0]);
		arr.resize(n);
		for (int i = 0; i < n; ++i) {
			arr[i] = static_cast<float>(arr_d[i]);
		}
	}

	void loadTimeStep(const int& i, cudaArray* vf_d) {
		float *udata = u_data.getData(), *vdata = v_data.getData(), *wdata;
		vec4f *vfdata = vf_data.getData();
		u.get3DSlice(udata, i);
		v.get3DSlice(vdata, i);
		int size = u_data.volumeSize();
		if (w.isValid()) {
			float *wdata = w_data.getData();
			w.get3DSlice(wdata, i);
			for (int i = 0; i < size; ++i) {
				vfdata[i].x = udata[i];
				vfdata[i].y = vdata[i];
				vfdata[i].z = -wdata[i];
			}
		} else {
			for (int i = 0; i < size; ++i) {
				vfdata[i].x = udata[i];
				vfdata[i].y = vdata[i];
			}
		}

		updateCudaVectorField(vf_d, vfdata, dim);
	}

	void loadTimeStep(const int& i) {
		if (i == 0) {
			loadTimeStep(0, prev_vec);
			loadTimeStep(1, next_vec);
		} else {
			std::swap(prev_vec, next_vec);
			if (i < num_timestep - 1) {
				loadTimeStep(i + 1, next_vec);
			} else if (i == num_timestep - 1) {
				next_vec = prev_vec;
			}
		}
		cudaEarthUpdateVectorFieldTex_h(prev_vec, next_vec);
	}

	void updateTime() {
		if (current_time<num_timestep-1) {
			++current_trace;
			if (current_trace==num_trace_per_time_step) {
				current_trace = 0;
				++current_time;
				loadTimeStep(current_time);
			}
		}
	}

	void initTracer() {
		current_point = 0;
		
		//init random states
		particles_RNG = earthCreateRNG(max_num_particles);

		//init point positions
		vec4f* points_d = (vec4f*)particles_cuvbo.map();
		cudaMemset(points_d, 0xee, sizeof(vec4f)*max_num_particles*num_points_per_particle);
		generateRandPoints_h(points_d, num_particles, num_particle_per_thread, dim);
		particles_cuvbo.unmap();

		//init indices
		int* indices_d = (int*)streamlet_indices_cuvbo.map();
		cudaEarthSetIndex_h(indices_d, num_particle_per_thread, num_points_per_particle, max_num_particles);
		streamlet_indices_cuvbo.unmap();

		//init counts and starts for glMultiDraw
		streamlet_counts.assign(max_num_particles, num_points_per_particle);
		streamlet_starts.resize(num_points_per_particle);
		for (int i = 0; i < num_points_per_particle; ++i) {
			streamlet_starts[i].resize(max_num_particles);
			for (int j = 0; j < max_num_particles; ++j) {
				streamlet_starts[i][j] = (GLvoid*)(sizeof(unsigned int)*((2*num_points_per_particle-1)*j+(i+1)%num_points_per_particle));
			}
		}
	}

	void getParticles(std::vector<vec4f>& particles) {
		particles.resize(num_particles*num_points_per_particle);
		vec4f* points_d = (vec4f*)particles_cuvbo.map();
		cudaMemcpy(&particles[0], points_d, num_particles*num_points_per_particle *sizeof(vec4f), cudaMemcpyDeviceToHost);
		particles_cuvbo.unmap();
	}

	void printPoints() {
		std::vector<unsigned int> indices(streamlet_indices_cuvbo.size / sizeof(unsigned int));
		streamlet_indices_cuvbo.dump(indices.data());
		std::vector<vec4f> points(particles_cuvbo.size / sizeof(vec4f));
		particles_cuvbo.dump(points.data());
		for (int i = 0; i < num_particles; ++i) {
			printf("%i particle of iter %i\n", i, current_point);
			for (int j = 0; j < streamlet_counts[i]; ++j) {
				int idx = indices[((int)streamlet_starts[current_point][i]) / sizeof(unsigned int) + j];
				vec4f p = points[idx];
				printf("%5i: %8.4f %8.4f %8.4f %5i %5i ", idx, p.x, p.y, p.z, earthGetTime(p.w), earthGetCurrentPoint(p.w));
				if (earthIsOutOfBound(p.w)) {
					printf("out of bound.\n");
				} else if (earthIsInvalid(p.w)) {
					printf("invalid.\n");
				} else {
					printf("valid.\n");
				}
 			}
		}
	}

	void trace() {
		int next_point = (current_point + 1 == num_points_per_particle) ? 0 : (current_point + 1);
		
		//update point positions
		if (current_point%4==0) 
			recycleParticles();

		//trace particles
		vec4f* points_d = (vec4f*)particles_cuvbo.map();
		cudaEarthTrace_h(&points_d[max_num_particles*current_point], &points_d[max_num_particles*next_point], 
			num_particles, num_particle_per_thread, num_step_per_trace, w_scale, makeVec3f(dim), 
			current_trace/(float)num_trace_per_time_step, 1.0f/(num_step_per_trace*num_trace_per_time_step), 
			trace_interval);
		particles_cuvbo.unmap();
		current_point = next_point;
		//printPoints();
		updateTime();
	}

	void updateArea(float *modelview_mat, float *projection_mat) {
		cudaQuadAreaOnScreen(earth_area_d.data_d, earth_vertices_d.data_d, earth_quads_d.data_d,
			modelview_mat, projection_mat, num_particle_per_thread, num_earth_quads);
		std::vector<float> area(earth_area_d.size);
		earth_area_d.dump(area.data());
	}

	void updateImportance(const bool& b_use_particle_count) {
		earth_count_d.memset(0);
		if (b_use_particle_count) {
			vec4f* points_d = (vec4f*)particles_cuvbo.map();
			for (int i = 0; i < num_points_per_particle; ++i) {
				earthBlockParticleCount_h(earth_count_d.data_d, &points_d[max_num_particles*current_point],
					earth_grid_dim, earth_block_size, num_particle_per_thread, num_particles);
			}
			particles_cuvbo.unmap();
		}
		earthBlockImportance_h(earth_importance_d.data_d, earth_count_d.data_d, earth_area_d.data_d, 0.5f,
			num_particle_per_thread, earth_importance_d.size);
		
		NonUniformSampleBinary nusb;
		std::vector<float> importance(earth_importance_d.size);
		earth_importance_d.dump(importance.data());
		nusb.updateDistributions(importance.data(), importance.size());
		earth_importance_sum = nusb.sum_prob;
		earth_importance_max = nusb.max_prob;
		num_earth_bitems = nusb.bitems.size();
		earth_distribution_d.load(nusb.bitems.data(), num_earth_bitems);
	}

	void recycleParticles() {
		updateImportance(true);
		vec4f* points_d = (vec4f*)particles_cuvbo.map();
		int prev_point_offset = (current_point == 0) ? ((num_points_per_particle - 1)*max_num_particles) : -max_num_particles;
		int next_piont_offset = (current_point == num_points_per_particle - 1) ? ((1 - num_points_per_particle)*max_num_particles) : max_num_particles;
		earthRecycleParticles_h(&points_d[current_point*max_num_particles], num_particles, current_point,
			z_up, z_low, earth_grid_dim, earth_block_size, earth_distribution_d.data_d, num_earth_bitems, 
			num_points_per_particle*recycle_max_time_scale, num_points_per_particle*recycle_min_time_scale,
			earth_importance_d.data_d, earth_importance_max*recycle_max_imp_scale, earth_importance_max*recycle_min_imp_scale,
			recycle_curvature_weight, prev_point_offset, next_piont_offset, particles_RNG);
		particles_cuvbo.unmap();
	}

	void resetAllParticles() {
		updateImportance(false);
		vec4f* points_d = (vec4f*)particles_cuvbo.map();
		current_point = 0;
		checkCudaErrors(cudaMemset(points_d, 0xee, sizeof(vec4f)*max_num_particles*num_points_per_particle));
		int prev_point_offset = (num_points_per_particle - 1)*max_num_particles;
		earthRecycleParticles_h(&points_d[current_point], num_particles, current_point, z_up, z_low,
			earth_grid_dim, earth_block_size, earth_distribution_d.data_d, num_earth_bitems, -1e30, 1e30,
			earth_importance_d.data_d, -1e30, 1e30, prev_point_offset, recycle_curvature_weight, 
			max_num_particles, particles_RNG);
		//printPoints();
		particles_cuvbo.unmap();
	}

	void updateEarthDeviceData(const float& earth_radius) {
		std::vector<vec3f> earth_vertices;
		std::vector<vec4i> earth_quads;

		vec2f pos_range = makeVec2f(dim.x, dim.y - 1.0f);
		earth_grid_dim = makeVec2i(pos_range.x / 5.0f, pos_range.y / 5.0f);
		earth_block_size = makeVec2f(pos_range.x / earth_grid_dim.x, pos_range.y / earth_grid_dim.y);
		vec2f index_to_radian = makeVec2f(earth_block_size.x*pos_to_radian.x, earth_block_size.y*pos_to_radian.y);

		int rw = earth_grid_dim.x + 1;
		for (int i = 0; i <= earth_grid_dim.y; ++i) {
			float lat = i*index_to_radian.y + pos_offset.y;
			for (int j = 0; j <= earth_grid_dim.x; ++j) {
				float lon = j*index_to_radian.x + pos_offset.x;
				earth_vertices.push_back(earth_radius*lonLatTo3D(makeVec2f(lon, lat)));
				if (i != 0 && j!=0) {
					earth_quads.push_back(makeVec4i((i - 1)*rw + j-1, i*rw + j-1, i*rw + j, (i - 1)*rw + j));
				}
			}
		}

		//allocate memory
		if (earth_vertices_d.empty()) {
			num_earth_vertices = earth_vertices.size();
			num_earth_quads = earth_quads.size();
			earth_vertices_d.allocate(num_earth_vertices);
			earth_quads_d.allocate(num_earth_quads);
			earth_area_d.allocate(num_earth_quads);
			earth_count_d.allocate(num_earth_quads);
			earth_importance_d.allocate(num_earth_quads);
			earth_distribution_d.allocate(num_earth_quads);
		}
		
		//copy to device
		earth_vertices_d.load(earth_vertices.data());
		earth_quads_d.load(earth_quads.data());
	}

	vec3i getDimension() { return dim; }
	int getNumParticles() { return num_particles; }
	int getParticleVBO() { return particles_cuvbo.getVBO(); }
	int getIndicesVBO() { return streamlet_indices_cuvbo.getVBO(); }
	float getCurrentTimestep() { return (current_time + current_trace / (float)num_trace_per_time_step); }

	const GLsizei* getStreamletCounts() { return (const GLsizei*)streamlet_counts.data(); }
	const GLvoid** getStreamletStarts() { return (const GLvoid**)(streamlet_starts[current_point].data()); }

	NCSliceData u, v, w;
	vec3i dim;
	VolumeData<float> u_data, v_data, w_data;
	VolumeData<vec4f> vf_data;
	std::vector<float> lat, lon, lev;

	//particle settings
	int max_num_particles;
	int num_particles;
	int num_points_per_particle;
	int current_point;

	//tracing parameters
	float trace_interval;
	int num_particle_per_thread;
	int num_timestep;
	int num_step_per_trace;
	int num_trace_per_time_step;
	int current_time;
	int current_trace;
	int seg_len;
	float w_scale;
	float z_up;
	float z_low;

	//recycle parameters
	float recycle_max_imp_scale;
	float recycle_min_imp_scale;
	float recycle_max_time_scale;
	float recycle_min_time_scale;
	float recycle_curvature_weight;

	vec3f pos_to_radian;
	vec3f pos_offset;

	float earth_step_size;
	cudaDeviceMem<vec3f> earth_vertices_d;
	cudaDeviceMem<vec4i> earth_quads_d;
	cudaDeviceMem<float> earth_area_d;
	cudaDeviceMem<float> earth_importance_d;
	cudaDeviceMem<int> earth_count_d;
	cudaDeviceMem<NUSBItem> earth_distribution_d;
	int num_earth_bitems;
	int num_earth_vertices;
	int num_earth_quads;
	float earth_importance_sum;
	float earth_importance_max;
	vec2i earth_grid_dim;
	vec2f earth_block_size;

	cudaArray *prev_vec;
	cudaArray *next_vec;

	curandState* particles_RNG;
	cudaVBO particles_cuvbo;
	cudaVBO streamlet_indices_cuvbo;
	std::vector<GLsizei> streamlet_counts;
	std::vector<std::vector<GLvoid*>> streamlet_starts;
};