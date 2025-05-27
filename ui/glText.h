#pragma once

#include <string>
#include <QImage>
#include <QOpenGLTexture>
#include <gl/glew.h>

class glText {
public:
	glText() {
		QImage text_img(QString("D:/PROJECT/data/lucida_console.png"));
		if (text_img.isNull()) {
			printf("Fail to read font texture: ./data/lucida_console.png.\n");
		}
		else {
			mFontTex = new QOpenGLTexture(text_img);
			mFontTex->setMinificationFilter(QOpenGLTexture::Linear);
			mFontTex->setMagnificationFilter(QOpenGLTexture::Linear);
			mCharPerLine = 16;
			mTexXUnit = 1.0f / mCharPerLine;
			mTexYUnit = 1.0f / 6.0f;
		}

	}

	~glText() {
		delete mFontTex;
	}

	int get_char_id(const char& c) {
		if (c < 32 || c>126) return -1;
		return (c - 33);
	}

	void drawText(const int& x, const int& y, const std::string& str, const float& scale) {
		glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		glDisable(GL_LIGHTING);
		mFontTex->bind(0);
		glEnable(GL_TEXTURE_2D);

		float w = scale, h = 2.0f * scale;
		float cur_x = x, cur_y = y;

		//glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
		glBegin(GL_QUADS);

		for (int i = 0; i < str.size(); ++i) {
			drawChar(str[i], cur_x, cur_y, w, h);
			cur_x += w;
		}
		glEnd();

		mFontTex->release();
		glPopAttrib();
	}

	void drawRotatedText(const int& x, const int& y, const std::string& str, const float& scale, const float& angle) {
		glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		glDisable(GL_LIGHTING);
		mFontTex->bind(0);
		glEnable(GL_TEXTURE_2D);

		float w = scale, h = 2.0f * scale;
		float cur_x = 0, cur_y = 0; // 旋转后的文本应相对于原点进行位置计算

		// 保存当前变换矩阵状态
		glPushMatrix();

		// 将旋转中心移动到文本的左下角位置
		glTranslatef(x, y, 0);
		// 应用旋转变换
		glRotatef(angle, 0.0f, 0.0f, 1.0f);
		// 将旋转中心移回到原点
		glTranslatef(-x, -y, 0);

		glBegin(GL_QUADS);

		for (int i = 0; i < str.size(); ++i) {
			drawChar(str[i], x + cur_x, y + cur_y, w, h);
			cur_x += w; // 移动到下一个字符的位置
		}

		glEnd();

		// 恢复保存的变换矩阵状态
		glPopMatrix();

		mFontTex->release();
		glPopAttrib();
	}


	void drawChar(const char& c, const float& x, const float& y, const float& w, const float& h) {
		int cid = get_char_id(c);
		if (cid < 0) return;

		float tex_x = (cid % mCharPerLine) * mTexXUnit;
		float tex_y = (cid / mCharPerLine) * mTexYUnit;
		glTexCoord2f(tex_x, tex_y);
		glVertex2f(x, y + h);
		glTexCoord2f(tex_x + mTexXUnit, tex_y);
		glVertex2f(x + w, y + h);
		glTexCoord2f(tex_x + mTexXUnit, tex_y + mTexYUnit);
		glVertex2f(x + w, y);
		glTexCoord2f(tex_x, tex_y + mTexYUnit);
		glVertex2f(x, y);
	}

	void drawFigure(const float& x, const float& y, std::string path, const float& scale) {
		QImage img(QString::fromStdString(path));
		QOpenGLTexture* ImgTex = new QOpenGLTexture(img);
		ImgTex->setMinificationFilter(QOpenGLTexture::Linear);
		ImgTex->setMagnificationFilter(QOpenGLTexture::Linear);

		glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		glDisable(GL_DEPTH_TEST);

		glDisable(GL_LIGHTING);
		ImgTex->bind(0);
		glEnable(GL_TEXTURE_2D);
		glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
		glBegin(GL_QUADS);

		float w = scale;
		float h = scale * 1.35;
		
		float z = 1.0f; 

		glTexCoord2f(0, 0);
		glVertex3f(x, y + h, z);
		glTexCoord2f(1, 0);
		glVertex3f(x + w, y + h, z);
		glTexCoord2f(1, 1);
		glVertex3f(x + w, y, z);
		glTexCoord2f(0, 1);
		glVertex3f(x, y, z);

		glEnd();

		ImgTex->release();
		glEnable(GL_DEPTH_TEST);
		glPopAttrib();
	}


	float mTexXUnit, mTexYUnit;
	int mCharPerLine;
	QOpenGLTexture* mFontTex;
	//QOpenGLTexture* mImgTex;
};