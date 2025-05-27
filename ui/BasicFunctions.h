#ifndef BASIC_FUNCTIONS_H
#define BASIC_FUNCTIONS_H

#include <QWidget>
#include <QFileDialog>
#include <QPixmap>
#include <cassert>

static bool saveScreenshot(QWidget* w, const QString& filePath){
	QString fileName =
		QFileDialog::getSaveFileName(w, "Save Image", filePath, "Images (*.bmp)");
	if (fileName.length()==0) return false;
	QPixmap p = QPixmap::grabWidget(w);
	bool ret = p.save(fileName);
	return ret;
}

static bool savePPM(const char *ppmFilename, unsigned char *image_ptr, const int& image_x, const int& image_y){
	FILE *out = fopen(ppmFilename, "wb");

	if (out == NULL) {
		printf(" can't open file %s\n", ppmFilename);
		return false;
	}

	fprintf(out, "P3\n");
	fprintf(out, "%d  %d\n", image_x, image_y);
	fprintf(out, "%d\n", 256);

	unsigned char *buffer;

	// save the data
	for (int i = 0; i < image_y; i++) {
		buffer = image_ptr + (image_y - i - 1) * image_x * 3;

		for (int j = 0; j < image_x; j++) {
			fprintf(out, "%d %d %d\n",
				(int)(buffer[j*3]),
				(int)(buffer[j*3+1]),
				(int)(buffer[j*3+2]));
		}
	}

	fclose(out);

	return true;
}

#define BMP_Header_Length 54
static bool saveBMP(const char *bmpFilename, const int& image_x, const int& image_y){
	FILE*    pDummyFile;
	FILE*    pWritingFile;
	GLubyte* pPixelData;
	GLubyte  BMP_Header[BMP_Header_Length];
	GLint    i, j;
	GLint    PixelDataLength;

	pDummyFile = fopen("dummy.bmp", "rb");
	if( pDummyFile == 0 ){
		printf("can't read dummy.bmp");
		return false;
	}

	pWritingFile = fopen(bmpFilename, "wb");
	if( pWritingFile == 0 ){
		printf("can't write file");
		return false;
	}

	i = image_x * 3;   
	while( i%4 != 0 ) ++i;               
	PixelDataLength = i * image_y;
	pPixelData = new GLubyte[PixelDataLength];

	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
	glReadPixels(0, 0, image_x, image_y,
		GL_BGR_EXT, GL_UNSIGNED_BYTE, pPixelData);

	fread(BMP_Header, sizeof(BMP_Header), 1, pDummyFile);
	fwrite(BMP_Header, sizeof(BMP_Header), 1, pWritingFile);
	fseek(pWritingFile, 0x0012, SEEK_SET);
	i = image_x;
	j = image_y;
	fwrite(&i, sizeof(i), 1, pWritingFile);
	fwrite(&j, sizeof(j), 1, pWritingFile);

	fseek(pWritingFile, 0, SEEK_END);
	fwrite(pPixelData, PixelDataLength, 1, pWritingFile);

	fclose(pDummyFile);
	fclose(pWritingFile);
	delete[] pPixelData;

	return true;
}

#endif//BASIC_FUNCTIONS_H