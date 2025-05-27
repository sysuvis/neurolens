#pragma once

#include <vector>
#include <cuda.h>
#include "helper_cuda.h"

template<typename T>
class cudaDeviceMem{
public:
	cudaDeviceMem():
	data_d(NULL),
	size(0)
	{
	}

	cudaDeviceMem(const int& _size):
	data_d(NULL)
	{
		allocate(_size);
	}

	cudaDeviceMem(const T* data, const int& _size) :
	data_d(NULL)
	{
		allocate(_size);
		load(data);
	}

	cudaDeviceMem(const std::vector<T>& data) :
	data_d(NULL)
	{
		allocate(data.size());
		load(data.data());
	}

	~cudaDeviceMem() {
		free();
	}

	void allocate(const int& _size) {
		if (_size!=size){
			size = _size;
			if (data_d != NULL) {
				checkCudaErrors(cudaFree(data_d));
			}
			checkCudaErrors(cudaMalloc(&data_d, sizeof(T)*size));	
		}
	}

	void free() {
		if (data_d != NULL) {
			checkCudaErrors(cudaFree(data_d));
			size = 0;
		}
	}

	void dump(T* host_pointer) {
		checkCudaErrors(cudaMemcpy(host_pointer, data_d, sizeof(T)*size, cudaMemcpyDeviceToHost));
	}

	void dump(T* host_pointer, const int& _size, const int& dev_offset=0) {
		checkCudaErrors(cudaMemcpy(host_pointer, &data_d[dev_offset], sizeof(T)*_size, cudaMemcpyDeviceToHost));
	}

	void load(const T* host_pointer) {
		checkCudaErrors(cudaMemcpy(data_d, host_pointer, sizeof(T)*size, cudaMemcpyHostToDevice));
	}

	void load(const T* host_pointer, const int& _size, const int& dev_offset = 0) {
		checkCudaErrors(cudaMemcpy(&data_d[dev_offset], host_pointer, sizeof(T)*_size, cudaMemcpyHostToDevice));
	}
	
	void copy(T* dev_pointer) {
		checkCudaErrors(cudaMemcpy(dev_pointer, data_d, sizeof(T)*size, cudaMemcpyDeviceToDevice));
	}

	void copy(T* dev_pointer, const int& _size, const int& dev_offset = 0) {
		checkCudaErrors(cudaMemcpy(dev_pointer, &data_d[dev_offset], sizeof(T)*_size, cudaMemcpyDeviceToDevice));
	}

	void copy(cudaDeviceMem* dev_mem) {
		copy(dev_mem->data_d, dev_mem->size, 0);
	}

	void memset(const char& c) {
		checkCudaErrors(cudaMemset(data_d, c, sizeof(T)*size));
	}

	void memset(const char& c, const int& _size, const int& offset = 0) {
		checkCudaErrors(cudaMemset(&data_d[offset], c, sizeof(T)*_size));
	}

	bool empty(){
		return (size == 0);
	}

	T* data_d;
	int size;
};