#include "FRLayout.h"
#include <cmath>
#include <ctime>

FRLayout::FRLayout(const float& c, const float& k)
:GplGraph(),
mC(c),
mK(k)
{
}

FRLayout::~FRLayout(){
}

void FRLayout::updateLayout(){
	vec2f m;
	float t = mK;
	float e = mK*nodes.size();

	int count=0;
	while (e>0.001f*mK*nodes.size() || count<1000){
		e = 0.0f;
		for (int i=0; i<nodes.size(); ++i){
			m = getNodeMovement(i, t);
			nodes[i].pos = nodes[i].pos+m;
			e += vec2dLen(m);
		}
		t*=0.95f;
		++count;
	}

	updateRange();
}

vec2f FRLayout::getNodeMovement(const int& nid, const float& stepSize){
	vec2f m = makeVec2f(0.0f, 0.0f);
	vec2f& npos = nodes[nid].pos, ipos, d;
	float w, ld, lm;
	float n = (float)nodes[nid].numNeighbors;

	for (int i=0; i<nodes.size(); ++i){
		if (i!=nid){
			ipos = nodes[i].pos;
			d = ipos-npos;
			ld = vec2dLen(d);
			if (dist[nid][i]>=GPL_INFINITE){//not neighbors
				w = -mC*mK*mK/ld;
			} else {
				w = (ld-mK)/n;
			}
			m = m+(1.0f/ld*w)*d;
		}
	}

	if((lm=vec2dLen(m))>stepSize){
		m = stepSize/lm*m;
	}

	return m;
}