#include "FeatureFlowField.h"
#include "CriticalPointDetection.h"

VolumeData<vec4f>* computeFeatureFlow(VolumeData<vec3f>* prev, VolumeData<vec3f>* curr, VolumeData<vec3f>* next, const float& scale_diff){
	vec3i dim =  makeVec3i(prev->width(), prev->height(), prev->depth());
	vec3f jacob[3];
	vec3f d;
	vec4f ret_vec;
	VolumeData<vec4f>* ret = new VolumeData<vec4f>(dim.x, dim.y, dim.z);
	float fac = 1.0f/scale_diff;

	vec3i p;
	for (p.z=0; p.z<dim.z; ++p.z) {
		for (p.y=0; p.y<dim.y; ++p.y) {
			for (p.x=0; p.x<dim.x; ++p.x) {
				getJacobianAtGridPoint(p, curr->getVolumeData(), dim, jacob);
				d = fac*(next->getValueQuick(p.x, p.y, p.z)-prev->getValueQuick(p.x, p.y, p.z));
				ret_vec.x = determinantOfColumnMatrix(jacob[1], jacob[2], d);//-determinantOfColumnMatrix(jacob[1], jacob[2], d);
				ret_vec.y = -determinantOfColumnMatrix(jacob[2], d, jacob[0]);//determinantOfColumnMatrix(jacob[2], d, jacob[1]);
				ret_vec.z = determinantOfColumnMatrix(d, jacob[0], jacob[1]);//-determinantOfColumnMatrix(d, jacob[0], jacob[1]);
				ret_vec.w = -determinantOfColumnMatrix(jacob[0], jacob[1], jacob[2]);
				ret->setValueQuick(p.x, p.y, p.z, ret_vec);
			}
		} 
	}

	return ret;
}

VolumeData<vec4f>* computeFeatureFlowFirst(VolumeData<vec3f>* curr, VolumeData<vec3f>* next, const float& scale_diff){
	vec3i dim =  makeVec3i(curr->width(), curr->height(), curr->depth());
	vec3f jacob[3];
	vec3f d;
	vec4f ret_vec;
	VolumeData<vec4f>* ret = new VolumeData<vec4f>(dim.x, dim.y, dim.z);
	float fac = 2.0f/scale_diff;

	vec3i p;
	for (p.z=0; p.z<dim.z; ++p.z) {
		for (p.y=0; p.y<dim.y; ++p.y) {
			for (p.x=0; p.x<dim.x; ++p.x) {
				getJacobianAtGridPoint(p, curr->getVolumeData(), dim, jacob);
				d = fac*(next->getValueQuick(p.x, p.y, p.z)-curr->getValueQuick(p.x, p.y, p.z));
				ret_vec.x = determinantOfColumnMatrix(jacob[1], jacob[2], d);//-determinantOfColumnMatrix(jacob[1], jacob[2], d);
				ret_vec.y = -determinantOfColumnMatrix(jacob[2], d, jacob[0]);//determinantOfColumnMatrix(jacob[2], d, jacob[1]);
				ret_vec.z = determinantOfColumnMatrix(d, jacob[0], jacob[1]);//-determinantOfColumnMatrix(d, jacob[0], jacob[1]);
				ret_vec.w = -determinantOfColumnMatrix(jacob[0], jacob[1], jacob[2]);
				ret->setValueQuick(p.x, p.y, p.z, ret_vec);
			}
		} 
	}

	return ret;
}

VolumeData<vec4f>* computeFeatureFlowLast(VolumeData<vec3f>* prev, VolumeData<vec3f>* curr, const float& scale_diff){
	vec3i dim =  makeVec3i(curr->width(), curr->height(), curr->depth());
	vec3f jacob[3];
	vec3f d;
	vec4f ret_vec;
	VolumeData<vec4f>* ret = new VolumeData<vec4f>(dim.x, dim.y, dim.z);
	float fac = 2.0f/scale_diff;

	vec3i p;
	for (p.z=0; p.z<dim.z; ++p.z) {
		for (p.y=0; p.y<dim.y; ++p.y) {
			for (p.x=0; p.x<dim.x; ++p.x) {
				getJacobianAtGridPoint(p, curr->getVolumeData(), dim, jacob);
				d = fac*(curr->getValueQuick(p.x, p.y, p.z)-prev->getValueQuick(p.x, p.y, p.z));
				ret_vec.x = determinantOfColumnMatrix(jacob[1], jacob[2], d);//-determinantOfColumnMatrix(jacob[1], jacob[2], d);
				ret_vec.y = -determinantOfColumnMatrix(jacob[2], d, jacob[0]);//determinantOfColumnMatrix(jacob[2], d, jacob[1]);
				ret_vec.z = determinantOfColumnMatrix(d, jacob[0], jacob[1]);//-determinantOfColumnMatrix(d, jacob[0], jacob[1]);
				ret_vec.w = -determinantOfColumnMatrix(jacob[0], jacob[1], jacob[2]);
				ret->setValueQuick(p.x, p.y, p.z, ret_vec);
			}
		} 
	}

	return ret;
}