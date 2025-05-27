#include "MatrixDisplay.h"
#include "DisplayWidget.h"

void MatrixDisplay<float>::drawCell(const float& val, const RectDisplayArea& area){
	if (val<mFilter[0] || val>mFilter[1]) {
		vec4f color = ColorMap::getGrayScale(0.3f);
		glColor(color);
		
		glBegin(GL_LINE_STRIP);
		glVertex(area.origin);
		glVertex(area.origin + area.row_axis);
		glVertex(area.origin + area.row_axis + area.col_axis);
		//glVertex(area.origin);
		glVertex(area.origin + area.col_axis);
		//glVertex(area.origin + area.row_axis + area.col_axis);
		glEnd();
	}
	else {
		float nv = interpolate(mNormalizationDomain[0], mNormalizationDomain[1], 
			val, mNormalizationRange[0], mNormalizationRange[1]);

		vec4f color = ColorMap::getLinearColor(nv, mColorSheme);
		glColor4f(color.r, color.g, color.b, color.a);
		glBegin(GL_QUADS);
		glVertex(area.origin);
		glVertex(area.origin + area.row_axis);
		glVertex(area.origin + area.row_axis + area.col_axis);
		glVertex(area.origin + area.col_axis);
		glEnd();
	}
}

void MatrixDisplay<int>::drawCell(const int& val, const RectDisplayArea& area){
	if (val<mFilter[0] || val>mFilter[1]) return;

	vec4f color = ColorMap::getD3ColorNoGray(val);
	glColor4f(color.r, color.g, color.b, color.a);
	glBegin(GL_QUADS);
	glVertex(area.origin);
	glVertex(area.origin+area.row_axis);
	glVertex(area.origin+area.row_axis+area.col_axis);
	glVertex(area.origin+area.col_axis);
	glEnd();
} 