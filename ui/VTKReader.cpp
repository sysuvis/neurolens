#include "VTKReader.h"
#include <fstream>

VTKReader::VTKReader(char* file_path)
:mVectorField(NULL)
{
	readFile(file_path);
	adjustVectorField();
}

VTKReader::~VTKReader(){
	std::map<std::string, VolumeData<float>*>::iterator it;
	for (it=mScalarFields.begin(); it!=mScalarFields.end(); ++it) {
		delete it->second;
	}
	if (mVectorField) delete mVectorField;
}

bool VTKReader::readFile(char* file_path){
	std::ifstream inFile;
	inFile.open(file_path, std::ios_base::in|std::ios_base::binary);
	if (!inFile.is_open()) return false;

	inFile.seekg (0, inFile.end);
	int fileLen = inFile.tellg();
	inFile.seekg (0, inFile.beg);

	//read header
	char buf[2048];
	inFile.read(buf, 2048);
	std::string str = buf;

	//read dimension
	size_t p = str.find("DIMENSIONS")+10, p2;
	int dim[3];
	for(int i=0; i<3; ++i){
		dim[i] = atoi(&buf[p]);
		p = str.find(' ', p+1);
	}
	int numPoint = dim[0]*dim[1]*dim[2];
	mDim = makeVec3i(dim[0], dim[1], dim[2]);
	int data_size;

	//read aspect ratio
	float aspect[3];
	p = str.find("ASPECT_RATIO")+12;
	for(int i=0; i<3; ++i){
		aspect[i] = atof(&buf[p]);
		p = str.find(' ', p+1);
	}
	mAspects = makeVec3f(aspect[0], aspect[1], aspect[2]);

	//read origin
	float org[3];
	p = str.find("ORIGIN")+6;
	for(int i=0; i<3; ++i){
		org[i] = atof(&buf[p]);
		p = str.find(' ', p+1);
	}
	mOrigin = makeVec3f(org[0], org[1], org[2]);

	////model matrix
	//IvtMatrix mat(aspect[0]*dim[0], 0, 0, org[0], 
	//	0, aspect[1]*dim[1], 0, org[1], 
	//	0, 0, aspect[2]*dim[2], org[2], 
	//	0, 0, 0, 1);

	//read fields
	int fpos = 0;
	while(1){
		if ((p=str.find("SCALARS"))!=std::string::npos) {
			data_size = numPoint*sizeof(float);
			p+=7;

			//find attribute name
			p2 = str.find(' ', p+1);
			std::string attrib(&buf[p+1], p2-p-1);

			//find the start of data
			if((p2=str.find("LOOKUP_TABLE", p+1))!=std::string::npos){
				p = str.find((char)0x0a, p2+1);
			} else {
				p = str.find((char)0x0a, p+1);
			}
			fpos += p+1;
			inFile.seekg(fpos);

			//read data
			VolumeData<float>* volume = new VolumeData<float>(dim[0], dim[1], dim[2]);
			float* data = volume->getData();
			char* cdata = (char*)data;

			inFile.read((char*)data, numPoint*sizeof(float));
			switchEndian4(cdata, numPoint);
			float minv=1e30, maxv=-1e30;
			for (int i=0; i<numPoint; ++i) {
				if (minv>data[i]) minv = data[i];
				if (maxv<data[i]) maxv = data[i];
			}

			mScalarFields[attrib] = volume;

			mScalarMin[attrib] = minv;
			mScalarMax[attrib] = maxv;

		} else if ((p=str.find("VECTORS"))!=std::string::npos){
			data_size = numPoint*sizeof(float3);
			fpos += str.find((char)0x0a, p+1)+1;
			inFile.seekg(fpos);

			//allocate volume
			mVectorField = new VolumeData<vec3f>(dim[0], dim[1], dim[2]);
			vec3f* data = mVectorField->getData();

			//read velocity field data
			inFile.read((char*)data, numPoint*sizeof(float3));
			char* cdata = (char*)data;
			switchEndian4(cdata, numPoint*3);
		} else {
			break;
		}
		fpos += data_size+1;
		if(fpos>=fileLen) break;
		inFile.seekg(fpos);
		inFile.read(buf, 2048);
		str = buf;
	}

	inFile.close();

	return true;
}

void VTKReader::adjustVectorField(){
	if (!mVectorField) return;

	vec3f* vecf = mVectorField->getData();
	int whd = mVectorField->volumeSize();
	float maxlen = 0.0f;
	float minlen = vec3fLen(vecf[0]);
	float len, avglen = 0.0f;
	int cnt = 0;
	for(int i=0; i<whd; i++){
		len = vec3fLen(vecf[i]);
		if(len>0){
			if(len>maxlen) maxlen = len;
			if(len<minlen) minlen = len;
			avglen += len;
			cnt++;
		}
	}
	maxlen /= avglen;
	minlen /= avglen;
	len = cnt/avglen;
	for(int i=0; i<whd; i++) vecf[i] = len*vecf[i];
}

void VTKReader::switchEndian4(char* data, const int& num_item){
	char tmp;
	for (int i=0; i<num_item*4; i+=4) {
		tmp = data[i];
		data[i] = data[i+3];
		data[i+3] = tmp;
		tmp = data[i+1];
		data[i+1] = data[i+2];
		data[i+2] = tmp;
	}
}