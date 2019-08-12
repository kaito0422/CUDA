#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <ctime>

//#define MAX_NUMBER	32768		// 32 * 1024
//#define MAX_NUMBER	65536		// 64 * 1024
//#define MAX_NUMBER	131072		// 128 * 1024
#define MAX_NUMBER	(1024 * 1024 * 16)		// 128 * 1024

typedef int		cuda_int;
typedef int		cpu_int;

const int device = 0;			// 在Matebook 14上只有一个MX250
int deviceNum;
int maxBlockNumX;
int maxBlockNumY;
int maxBlockNumZ;
int maxThdPerBlock;
cudaDeviceProp deviceProp;
cuda_int *dev_a = NULL;
cuda_int *dev_b = NULL;
cuda_int *dev_c = NULL;

cpu_int a[MAX_NUMBER];
cpu_int b[MAX_NUMBER];
cpu_int c[MAX_NUMBER];

cudaError_t cudaEnvInit()
{
	cudaError_t cudaStatus;
	int value;

	/* 在硬件平台上选择一个支持cuda的设备 */
	cudaStatus = cudaGetDeviceCount(&deviceNum);
	cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
	cudaStatus = cudaSetDevice(device);
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaSetDevice failed!" << std::endl;
		return cudaStatus;
	}
	cudaStatus = cudaDeviceGetAttribute(&value, cudaDevAttrMaxBlockDimX, device);\
	maxBlockNumX = value;
	cudaStatus = cudaDeviceGetAttribute(&value, cudaDevAttrMaxBlockDimY, device);
	maxBlockNumY = value;
	cudaStatus = cudaDeviceGetAttribute(&value, cudaDevAttrMaxBlockDimZ, device);
	maxBlockNumZ = value;

	cudaStatus = cudaMalloc((void **)&dev_a, MAX_NUMBER * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMalloc dev_a failed!" << std::endl;
		return cudaStatus;
	}
	cudaStatus = cudaMalloc((void **)&dev_b, MAX_NUMBER * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMalloc dev_b failed!" << std::endl;
		cudaFree(dev_a);
		return cudaStatus;
	}
	cudaStatus = cudaMalloc((void **)&dev_c, MAX_NUMBER * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMalloc dev_c failed!" << std::endl;
		cudaFree(dev_a);
		cudaFree(dev_b);
		return cudaStatus;
	}
	maxThdPerBlock = deviceProp.maxThreadsPerBlock;
	return cudaSuccess;
}

void cudaRelaseApp()
{
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}

void cudaShowDevInfo()
{
	std::cout << "CUDA device number: " << deviceNum << std::endl;
	std::cout << "CUDA device name: " << deviceProp.name << std::endl;
	std::cout << "CUDA device is " << (deviceProp.integrated == 1 ? "integrated" : "discreted") << std::endl;
	std::cout << "Multiprocessor number: " << deviceProp.multiProcessorCount << std::endl;
	std::cout << "register number of each Multiprocessor: " << deviceProp.regsPerMultiprocessor << std::endl;
	std::cout << "Global L1 cache supported: " << (deviceProp.globalL1CacheSupported == 1 ? "Yes" : "No") << std::endl;
	std::cout << "Local L1 cache supported: " << (deviceProp.localL1CacheSupported == 1 ? "Yes" : "No") << std::endl;
	std::cout << "L2 cache size: " << deviceProp.l2CacheSize << std::endl;
	std::cout << "warp size: " << deviceProp.warpSize << std::endl;
	std::cout << "Max threads dimension: " << deviceProp.maxThreadsDim << std::endl;
	std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
	std::cout << "Max threads per multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "registers per block: " << deviceProp.regsPerBlock << std::endl;
	std::cout << "Global memory available on device: " << (double)deviceProp.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
	std::cout << "Max X blocks: " << maxBlockNumX << std::endl;
	std::cout << "Max Y blocks: " << maxBlockNumY << std::endl;
	std::cout << "Max Z blocks: " << maxBlockNumZ << std::endl;
	std::cout << "Max threads per block: " << maxThdPerBlock << std::endl;
	std::cout << "Clock rate: " << (double)deviceProp.clockRate / 1024 / 1024 << " GHz" << std::endl;
}

__global__ void kernel(int *C, const int *A, const int *B)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	C[i] = (A[i] + B[i]) * (A[i] - B[i]);
	C[i] = 2 * (C[i] + i) / i;
	C[i] = C[i] * C[i];
	C[i] = C[i] + A[i] + B[i];
	C[i] = C[i] / A[i];
	C[i] = C[i] / B[i];
	C[i] = C[i] * C[i];
	C[i] = C[i] + A[i] + B[i];
	C[i] = C[i] / A[i];
	C[i] = C[i] / B[i];
	C[i] = C[i] * C[i];
	C[i] = C[i] + A[i] + B[i];
	C[i] = C[i] / A[i];
	C[i] = C[i] / B[i];
	C[i] = C[i] * C[i];
	C[i] = C[i] + A[i] + B[i];
	C[i] = C[i] / A[i];
	C[i] = C[i] / B[i];
}

int main()
{
	for (int i = 0; i < MAX_NUMBER; i++) {
		a[i] = 2;
		b[i] = 1;
	}

	clock_t start_t, end_t;
	cudaError_t res;
	res = cudaEnvInit();
	if (res != cudaSuccess)
		return 0;
	cudaShowDevInfo();

	res = cudaMemcpy(dev_a, a, sizeof(cpu_int) * MAX_NUMBER, cudaMemcpyHostToDevice);
	if (res != cudaSuccess)
		std::cout << "cudaMemcpy a failed!" << std::endl;
	res = cudaMemcpy(dev_b, b, sizeof(cpu_int) * MAX_NUMBER, cudaMemcpyHostToDevice);
	if (res != cudaSuccess)
		std::cout << "cudaMemcpy b failed!" << std::endl;

	int usedBlockNum = maxBlockNumZ;
	int usedThdPerBlock = maxThdPerBlock / 2;
	int iter_lenth = usedBlockNum * usedThdPerBlock;
	int iter_times = (MAX_NUMBER % iter_lenth == 0 ? MAX_NUMBER / iter_lenth : (MAX_NUMBER + iter_lenth) / iter_lenth);
	std::cout << "iter_lenth: " << iter_lenth << std::endl;
	std::cout << "iter_times: " << iter_times << std::endl;
	start_t = clock();
	for (int i = 0; i < iter_times; i++) {
		kernel<<<usedBlockNum, usedThdPerBlock>>>(dev_c + iter_lenth * i, dev_a + iter_lenth * i, dev_b + iter_lenth * i);
		res = cudaGetLastError();
		if (res != cudaSuccess)
			std::cout << "GetLastError failed!" << std::endl;
		res = cudaDeviceSynchronize();
		if (res != cudaSuccess)
			std::cout << "DeviceSynchronize failed!" << std::endl;
		if (i == iter_times - 1) {
			int copy_len = (MAX_NUMBER % iter_lenth == 0 ? iter_lenth : MAX_NUMBER % iter_lenth);
			res = cudaMemcpy(c + iter_lenth * i, dev_c + iter_lenth * i, sizeof(cpu_int) * copy_len, cudaMemcpyDeviceToHost);
		}
		else
			res = cudaMemcpy(c + iter_lenth * i, dev_c + iter_lenth * i, sizeof(cpu_int) * iter_lenth, cudaMemcpyDeviceToHost);
		if (res != cudaSuccess)
			std::cout << "cudaMemcpy c failed!" << std::endl;
	}
	end_t = clock();
	std::cout << "GPU used time: " << (double)(end_t - start_t) * 1000 / CLOCKS_PER_SEC << "ms" << std::endl;

	
	start_t = clock();
	for (int i = 1; i < MAX_NUMBER; i++) {
		c[i] = (a[i] + b[i]) * (a[i] - b[i]);
		c[i] = 2 * (c[i] + i) / i;
		c[i] = c[i] * c[i];
		c[i] = c[i] + a[i] + b[i];
		c[i] = c[i] / a[i];
		c[i] = c[i] / a[i];
		c[i] = c[i] * c[i];
		c[i] = c[i] + a[i] + b[i];
		c[i] = c[i] / a[i];
		c[i] = c[i] / a[i];
		c[i] = c[i] * c[i];
		c[i] = c[i] + a[i] + b[i];
		c[i] = c[i] / a[i];
		c[i] = c[i] / a[i];
		c[i] = c[i] * c[i];
		c[i] = c[i] + a[i] + b[i];
		c[i] = c[i] / a[i];
		c[i] = c[i] / a[i];
	}
	end_t = clock();
	std::cout << "CPU used time: " << (double)(end_t - start_t) * 1000 / CLOCKS_PER_SEC << "ms" << std::endl;
	
	cudaRelaseApp();

	for (int i = 0; i < 100; i++)
		std::cout << c[i] << ' ';
	std::cout << std::endl;

	system("pause");
	return 0;
}