#include "sha.h"

struct GPU_Properties {
    int concurrentKernels;
    int maxThreadsPerMultiProcessor;
    int multiProcessorCount;
    int warpSize;
};

extern "C" const GPU_Properties *get_gpu_props() {
    cudaDeviceReset(); // clear all existing allcations on device in case exception happens.
    
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    GPU_Properties *props = new GPU_Properties;
    props->concurrentKernels = device_prop.concurrentKernels;
    props->maxThreadsPerMultiProcessor = device_prop.maxThreadsPerMultiProcessor;
    props->multiProcessorCount = device_prop.multiProcessorCount;
    props->warpSize = device_prop.warpSize;
    return props;
}