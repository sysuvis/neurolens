#ifndef FR_LAYOUT_H
#define FR_LAYOUT_H

#include "GplGraph.h"

class FRLayout : public GplGraph{
public:
	FRLayout(const float& c, const float& k);
	~FRLayout();
	void setC(const float& c){mC=c;}
	void setK(const float& k){mK=k;}
	void updateLayout();
private:
	vec2f getNodeMovement(const int& nid, const float& stepSize);

	float mC, mK;
};

#endif//FR_LAYOUT_H