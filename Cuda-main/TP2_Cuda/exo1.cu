#include <iostream>

int main()
{
    int count = 0;

    cudaGetDeviceCount( &count );

    std::cout << count << " device(s) found.\n";

    cudaDeviceProp deviceProp;
    int dev = 0;

    cudaGetDeviceProperties(&deviceProp, dev);

    std::cout << deviceProp.multiProcessorCount << " multiprocesseur.\n";
    std::cout << deviceProp.maxThreadsPerBlock << " threads max par blocs.\n";
    
    std::cout << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << "\n";
    std::cout << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << "\n";
    return 0;
}