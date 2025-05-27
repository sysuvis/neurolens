#ifndef SHADER_UTILITIES_H
#define SHADER_UTILITIES_H


#include <GL/glew.h>

#define STRINGIFY(A) #A
#define GLSL(version, shader)  "#version " #version "\n" #shader

GLuint compileProgram(const char *vsource);
GLuint compileProgram(const char *vsource, const char *fsource);
GLuint compileProgram(const char* gsource, const char *vsource, const char *fsource);

#endif //SHADER_UTILITIES_H