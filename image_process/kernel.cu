#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <ctime>

#include "FreeImage.h"
#pragma commet(lib,"FreeImage.lib")
#pragma warning(disable : 4096)


//#define MAX_NUMBER	32768		// 32 * 1024
//#define MAX_NUMBER	65536		// 64 * 1024
//#define MAX_NUMBER	131072		// 128 * 1024
#define MAX_NUMBER	(1024 * 1024 * 16)		// 128 * 1024
#define TEST_NUMBER	512

typedef unsigned int		cuda_int;
typedef unsigned int		cpu_int;

const int device = 0;			// 在Matebook 14上只有一个MX250
int deviceNum;
int maxBlockNumX;
int maxBlockNumY;
int maxBlockNumZ;
int maxThdPerBlock;
cudaDeviceProp deviceProp;

void printChangePhase()
{
	std::cout << std::endl << "-------------------------------------------" << std::endl << std::endl;
}

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

	/* 记录GPU的硬件参数 */
	cudaStatus = cudaDeviceGetAttribute(&value, cudaDevAttrMaxBlockDimX, device);\
	maxBlockNumX = value;
	cudaStatus = cudaDeviceGetAttribute(&value, cudaDevAttrMaxBlockDimY, device);
	maxBlockNumY = value;
	cudaStatus = cudaDeviceGetAttribute(&value, cudaDevAttrMaxBlockDimZ, device);
	maxBlockNumZ = value;

	/* 在GPU端分配memory */
	
	maxThdPerBlock = deviceProp.maxThreadsPerBlock;
	return cudaSuccess;
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
	printChangePhase();
}

/***********************************************************************************************
*										GPU vs CPU kernel									   *
***********************************************************************************************/
cuda_int *dev_a = NULL;
cuda_int *dev_b = NULL;
cuda_int *dev_c = NULL;
cpu_int a[MAX_NUMBER];
cpu_int b[MAX_NUMBER];
cpu_int c[MAX_NUMBER];

__global__ void kernel(cuda_int *C, const cuda_int *A, const cuda_int *B)
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

void GPUvsCPU()
{
	std::cout << "GPU vs CPU: " << std::endl;
	cudaMalloc((void **)&dev_a, MAX_NUMBER * sizeof(int));
	cudaMalloc((void **)&dev_b, MAX_NUMBER * sizeof(int));
	cudaMalloc((void **)&dev_c, MAX_NUMBER * sizeof(int));

	/* 把我们需要进行计算的数据从CPU端搬运到GPU端 */
	cudaMemcpy(dev_a, a, sizeof(cpu_int) * MAX_NUMBER, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, sizeof(cpu_int) * MAX_NUMBER, cudaMemcpyHostToDevice);

	/* 根据GPU的硬件参数，确定线程空间（维度） */
	int usedBlockNum = maxBlockNumZ;
	int usedThdPerBlock = maxThdPerBlock;
	int iter_lenth = usedBlockNum * usedThdPerBlock;		// 软件层面，能同时处理多少个thread
	// 要完成所有的thread，需要多少次迭代
	int iter_times = (MAX_NUMBER % iter_lenth == 0 ? MAX_NUMBER / iter_lenth : (MAX_NUMBER + iter_lenth) / iter_lenth);
	std::cout << "iter_lenth: " << iter_lenth << std::endl;
	std::cout << "iter_times: " << iter_times << std::endl;

	clock_t start_t, end_t;
	/* 调用GPU来计算，并把计算完的数据搬运回CPU端 */
	start_t = clock();
	for (int i = 0; i < iter_times; i++) {
		/* 调用GPU计算 */
		kernel<<<usedBlockNum, usedThdPerBlock>>>(dev_c + iter_lenth * i, dev_a + iter_lenth * i, dev_b + iter_lenth * i);
		// kernel<<<blockNum, threadsPerBlock>>> (Param1, Param2, Param3, ...)
		// block中（MX250），每个block最多可包含1024个thread
		// 软件上的block抽象是在硬件上SM上调度的基本单元
		// 一个block中包含多个warp，每个warp有32个thead

		cudaDeviceSynchronize();
		if (i == iter_times - 1) {
			int copy_len = (MAX_NUMBER % iter_lenth == 0 ? iter_lenth : MAX_NUMBER % iter_lenth);
			cudaMemcpy(c + iter_lenth * i, dev_c + iter_lenth * i, sizeof(cpu_int) * copy_len, cudaMemcpyDeviceToHost);
		}
		else
			cudaMemcpy(c + iter_lenth * i, dev_c + iter_lenth * i, sizeof(cpu_int) * iter_lenth, cudaMemcpyDeviceToHost);
	}
	end_t = clock();
	std::cout << "GPU used time: " << (double)(end_t - start_t) * 1000 / CLOCKS_PER_SEC << "ms" << std::endl;

	/* 使用CPU来计算相同的程序量，记录时间，和GPU性能进行对比 */
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

	for (int i = 0; i < 100; i++)
		std::cout << c[i] << ' ';
	std::cout << std::endl;

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	printChangePhase();
}

/***********************************************************************************************
*										test 1D kernel										   *
***********************************************************************************************/
cpu_int block1D[MAX_NUMBER];
cpu_int warp1D[MAX_NUMBER];
cpu_int localthread1D[MAX_NUMBER];
cpu_int globalthread1D[MAX_NUMBER];
cuda_int *dev_block1D;
cuda_int *dev_warp1D;
cuda_int *dev_localthread1D;
cuda_int *dev_globalthread1D;

__global__ void kernelInfo1D(cuda_int *blockId, cuda_int *warpId, cuda_int *localThreadId, cuda_int *globalThreadId)
{
	int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
	blockId[thread_x] = blockIdx.x;
	warpId[thread_x] = threadIdx.x / warpSize;
	localThreadId[thread_x] = threadIdx.x;
	globalThreadId[thread_x] = thread_x;
}

void test1DKernel()
{
	std::cout << "1D kernel info: " << std::endl;
	cudaMalloc((void **)&dev_block1D, TEST_NUMBER * sizeof(int));
	cudaMalloc((void **)&dev_warp1D, TEST_NUMBER * sizeof(int));
	cudaMalloc((void **)&dev_localthread1D, TEST_NUMBER * sizeof(int));
	cudaMalloc((void **)&dev_globalthread1D, TEST_NUMBER * sizeof(int));

	kernelInfo1D<<<8, 64>>>(dev_block1D, dev_warp1D, dev_localthread1D, dev_globalthread1D);

	cudaMemcpy(block1D, dev_block1D, sizeof(cpu_int) * TEST_NUMBER, cudaMemcpyDeviceToHost);
	cudaMemcpy(warp1D, dev_warp1D, sizeof(cpu_int) * TEST_NUMBER, cudaMemcpyDeviceToHost);
	cudaMemcpy(localthread1D, dev_localthread1D, sizeof(cpu_int) * TEST_NUMBER, cudaMemcpyDeviceToHost);
	cudaMemcpy(globalthread1D, dev_globalthread1D, sizeof(cpu_int) * TEST_NUMBER, cudaMemcpyDeviceToHost);
	for (int i = 0; i < TEST_NUMBER; i++) {
		std::cout << "blockId:\t" << block1D[i] << "\twarpId:\t" << warp1D[i] \
			<< "\tlocalthreadId:\t" << localthread1D[i] << "\tglobalthreaId:\t" << globalthread1D[i] << std::endl;
	}

	cudaFree(dev_block1D);
	cudaFree(dev_warp1D);
	cudaFree(dev_localthread1D);
	cudaFree(dev_globalthread1D);

	printChangePhase();
}

/***********************************************************************************************
*										test 2D kernel										   *
***********************************************************************************************/
struct BlockPosition {
	int x;
	int y;
};

typedef struct BlockPosition cuda_pos;
typedef struct BlockPosition cpu_pos;

cpu_pos block2D[MAX_NUMBER / 2];
cpu_int localthread2D[MAX_NUMBER / 2];
cpu_int globalthread2D[MAX_NUMBER / 2];
cuda_pos *dev_block2D = NULL;
cuda_int *dev_localthread2D = NULL;
cuda_int *dev_globalthread2D = NULL;

__global__ void kernelInfo2D(cuda_pos *threadInfo, cuda_int *localThreadId, cuda_int *globalThreadId)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_idx = gridDim.x * blockDim.x * blockDim.y * blockIdx.y + blockDim.x * blockDim.y * blockIdx.x \
				   + blockDim.x * threadIdx.y + threadIdx.x;
	threadInfo[thread_idx].x = blockIdx.x;
	threadInfo[thread_idx].y = blockIdx.y;
	localThreadId[thread_idx] = threadIdx.y * blockDim.x + threadIdx.x;
	globalThreadId[thread_idx] = thread_idx;
}

void test2DKernel()
{
	std::cout << "2D kernel info: " << std::endl;
	cudaMalloc((void **)&dev_block2D, TEST_NUMBER * sizeof(cuda_pos) / 2);
	cudaMalloc((void **)&dev_localthread2D, TEST_NUMBER * sizeof(cuda_int) / 2);
	cudaMalloc((void **)&dev_globalthread2D, TEST_NUMBER * sizeof(cuda_int) / 2);

	dim3 threads_rect(32, 2);	// block的维度，每个block有32x2个thread
	dim3 blocks_rect(2, 2);		// grid的维度，每个grid有2x2个block

	kernelInfo2D<<<blocks_rect, threads_rect>>>(dev_block2D, dev_localthread2D, dev_globalthread2D);

	cudaMemcpy(block2D, dev_block2D, TEST_NUMBER * sizeof(cuda_pos) / 2, cudaMemcpyDeviceToHost);
	cudaMemcpy(localthread2D, dev_localthread2D, TEST_NUMBER * sizeof(cuda_int) / 2, cudaMemcpyDeviceToHost);
	cudaMemcpy(globalthread2D, dev_globalthread2D, TEST_NUMBER * sizeof(cuda_int) / 2, cudaMemcpyDeviceToHost);

	for (int i = 0; i < TEST_NUMBER / 2; i++) {
		std::cout << "block_x:\t" << block2D[i].x << "\tblock_y:\t" << block2D[i].y \
			<< "\tlocalthreadId:\t" << localthread2D[i] << "\tglobalthreaId:\t" << globalthread2D[i] << std::endl;
	}

	cudaFree(dev_block2D);
	cudaFree(dev_localthread2D);
	cudaFree(dev_globalthread2D);

	printChangePhase();;
}

/***********************************************************************************************
*										FreeImage   										   *
***********************************************************************************************/

void loadImage(const char *filename, int *width, int *height, unsigned int **buffer)
{
	FREE_IMAGE_FORMAT format = FreeImage_GetFileType(filename, 0);
	FIBITMAP *image = FreeImage_Load(format, filename);
	FIBITMAP *temp = image;
	image = FreeImage_ConvertTo32Bits(temp);
	*width = FreeImage_GetWidth(image);
	*height = FreeImage_GetHeight(image);
	*buffer = (unsigned int *)malloc(sizeof(unsigned int) * (*width) * (*height));
	if (!*buffer)
		std::cout << "malloc image buffer failed" << std::endl;
	else
		std::cout << "malloc image buffer success" << std::endl;
	memcpy(*buffer, FreeImage_GetBits(image), (*width) * (*height) * sizeof(unsigned int));

	FreeImage_Unload(image);
}

void saveImage(const char *filename, unsigned int *buffer, int width, int height)
{
	FREE_IMAGE_FORMAT format = FreeImage_GetFIFFromFilename(filename);
	FIBITMAP *image = FreeImage_ConvertFromRawBits((BYTE*)buffer, width,
		height, width * 4, 32, 0xFF000000, 0x00FF0000, 0x0000FF00);
	FreeImage_Save(format, image, filename);
}

/***********************************************************************************************
*								image processing kernel   									   *
***********************************************************************************************/
unsigned int *imageBuf;
unsigned int *outBuf;
cuda_int *dev_image;
cuda_int *dev_out;

__global__ void kernelImageProcessing(cuda_int *image, cuda_int *out)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
//	unsigned int thread_idx = (blockDim.x * blockDim.y) * (blockIdx.y * gridDim.x + blockIdx.y) + blockDim.x * threadIdx.y + threadIdx.x;
	unsigned int thread_idx = gridDim.x * blockDim.x * idy + idx;

	unsigned int data = image[thread_idx];
	data ^= 0x00FFFFFF;
	int red, green, blue;
	red = (data & 0x00FF0000) >> 16;
	green = (data & 0x0000FF00) >> 8;
	blue = (data & 0x000000FF) >> 0;
	int dif1, dif2, dif3;
	dif1 = red > green ? red - green : green - red;
	dif2 = red > blue ? red - blue : blue - red;
	dif3 = blue > green ? blue - green : green - blue;
	if (dif1 < 0x25 && dif2 < 0x25 && dif3 < 0x25)
		data = 0xFF000000;
	out[thread_idx] = data;
}

void processImage(const char *filename)
{
	int width, height;
	loadImage(filename, &width, &height, &imageBuf);
	outBuf = (unsigned int *)malloc(sizeof(unsigned int) * width * height);

	std::cout << filename << " processing: " << std::endl;
	cudaMalloc((void **)&dev_image, width * height * sizeof(cuda_int));
	cudaMalloc((void **)&dev_out, width * height * sizeof(cuda_int));

	cudaMemcpy(dev_image, imageBuf, sizeof(cpu_int) * width * height, cudaMemcpyHostToDevice);

	int block_x, block_y, thread_x, thread_y;
	block_x = width % 32 ? width / 32 + 1 : width / 32;
	block_y = height % 32 ? height / 32 + 1 : height / 32;
	dim3 threads_rect(1, 1);				// block的维度，每个block有32x32个thread
	dim3 blocks_rect(width, height);		// grid的维度

	kernelImageProcessing<<<blocks_rect, threads_rect>>>(dev_image, dev_out);

	cudaMemcpy(outBuf, dev_out, sizeof(cpu_int) * width * height, cudaMemcpyDeviceToHost);

	saveImage("kaito.png", outBuf, width, height);
}


int main()
{
	for (int i = 0; i < MAX_NUMBER; i++) {
		a[i] = 2;
		b[i] = 1;
	}

	clock_t start_t, end_t;
	cudaEnvInit();		// CUDA硬件使用环境初始化

	cudaShowDevInfo();
	GPUvsCPU();
//	test1DKernel();
//	test2DKernel();
	processImage("hyperbeast01.png");


	system("pause");
	return 0;
}
