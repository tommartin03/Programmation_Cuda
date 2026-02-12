#include <iostream>
#include <vector>


__global__ void vecadd( int * v1, int * v2, int size )
{
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
         v2[tid] += v1[tid];
    }
}


int main()
{
    size_t vec_size = 10;
    std::vector< int > v1( vec_size );
    std::vector< int > v2( vec_size );
    
    int * v1_d = nullptr;
    int * v2_d = nullptr;

    for( std::size_t i = 0 ; i < v1.size() ; ++i )
    {
        v1[ i ] =  i;
        v2[ i ] =  i;
    }
    
    cudaMalloc( &v1_d, v1.size() * sizeof( int ) );
    cudaMalloc( &v2_d, v2.size() * sizeof( int ) );


    cudaMemcpyAsync(v1_d, v1.data(), vec_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(v2_d, v2.data(), vec_size * sizeof(int), cudaMemcpyHostToDevice);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    int threadsPerBlock = 256;
    int numBlocks = (v1.size() + threadsPerBlock - 1) / threadsPerBlock;
    int mid = vec_size / 2;

    vecadd<<<numBlocks, threadsPerBlock, 0, stream1>>>(v1_d, v2_d, mid);
    vecadd<<<numBlocks, threadsPerBlock, 0, stream2>>>(v1_d + mid, v2_d + mid, vec_size - mid);

    cudaMemcpyAsync(v2.data(), v2_d, vec_size * sizeof(int), cudaMemcpyDeviceToHost, stream1);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    for (size_t idex = 0; idex < v1.size(); idex++)
        std::cout <<   v2[idex] << " ";
        std::cout << std::endl;

        
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(v1_d);
    cudaFree(v2_d);

    return 0;
}
