#include <cstdio>
#include "ShaderUtilities.h"

GLuint compileProgram(const char *vsource){
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);

	glShaderSource(vertexShader, 1, &vsource, 0);

	glCompileShader(vertexShader);

	GLuint program = glCreateProgram();

	glAttachShader(program, vertexShader);

	glLinkProgram(program);

	// check if program linked
	GLint success = 0;
	glGetProgramiv(program, GL_LINK_STATUS, &success);

	if (!success) {
		char temp[256];
		glGetProgramInfoLog(program, 256, 0, temp);
		printf("Failed to link program:\n%s\n", temp);
		glDeleteProgram(program);
		program = 0;
	}

	return program;
}

GLuint compileProgram(const char *vsource, const char *fsource){
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(vertexShader, 1, &vsource, 0);
	glShaderSource(fragmentShader, 1, &fsource, 0);

	glCompileShader(vertexShader);
	glCompileShader(fragmentShader);

	GLuint program = glCreateProgram();

	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);

	glLinkProgram(program);

	// check if program linked
	GLint success = 0;
	glGetProgramiv(program, GL_LINK_STATUS, &success);

	if (!success)
	{
		char temp[256];
		glGetProgramInfoLog(program, 256, 0, temp);
		printf("Failed to link program:\n%s\n", temp);
		glDeleteProgram(program);
		program = 0;
	}

	return program;
}

GLuint compileProgram(const char* gsource, const char *vsource, const char *fsource){
	GLuint geometryShader  = glCreateShader(GL_GEOMETRY_SHADER);
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(vertexShader, 1, &vsource, 0);
	glShaderSource(geometryShader, 1, &gsource, 0);
	glShaderSource(fragmentShader, 1, &fsource, 0);

	glCompileShader(vertexShader);
	glCompileShader(geometryShader);
	glCompileShader(fragmentShader);

	GLuint program = glCreateProgram();

	glAttachShader(program, vertexShader);
	glAttachShader(program, geometryShader);
	glAttachShader(program, fragmentShader);

	glLinkProgram(program);

	// check if program linked
	GLint success = 0;
	glGetProgramiv(program, GL_LINK_STATUS, &success);

	if (!success)
	{
		char temp[256];
		glGetProgramInfoLog(program, 256, 0, temp);
		printf("Failed to link program:\n%s\n", temp);
		glDeleteProgram(program);
		program = 0;
	}

	return program;
}