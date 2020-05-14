#include <cstddef>
#include <algorithm>
#include <array>
#include <cmath>

//Aguide is called p in the paper, W is called c_k, centers is called mu_k 
void calcW(float* const Aguide_ptr, float* const centers_ptr,
           ptrdiff_t sizeY, ptrdiff_t sizeX, ptrdiff_t rangeDims, ptrdiff_t numClusters, float const h,
           float* W_ptr){

    constexpr ptrdiff_t blockSize = 512;

    #pragma omp parallel for  
    for (ptrdiff_t x = 0; x < sizeX; x++){
        for (ptrdiff_t yBlockStart = 0; yBlockStart < sizeY; yBlockStart += blockSize){
            ptrdiff_t yBlockEnd = std::min(yBlockStart + blockSize, sizeY);
            
            for (ptrdiff_t c = 0; c < numClusters; c++){
                std::array<float, blockSize> W_buffer{};
                
                for (ptrdiff_t pIdx = 0; pIdx < rangeDims; pIdx++){
                    for (ptrdiff_t y = yBlockStart; y < yBlockEnd; y++){
                        
                        ptrdiff_t A_idx = y + x*sizeY + pIdx*sizeY*sizeX;
                        ptrdiff_t centers_idx = c + pIdx*numClusters;
                        ptrdiff_t buffer_idx = y - yBlockStart;

                        float centerDist = Aguide_ptr[A_idx] - centers_ptr[centers_idx];
                        W_buffer[buffer_idx] = W_buffer[buffer_idx] + centerDist*centerDist;
                    }
                }
                
                for (ptrdiff_t y = yBlockStart; y < yBlockEnd; y++){
                    ptrdiff_t W_idx = y + x*sizeY + c*sizeY*sizeX;
                    ptrdiff_t buffer_idx = y - yBlockStart;

                    W_ptr[W_idx] = std::exp( -W_buffer[buffer_idx] / (2*h*h) );
                }
            }
        }
    }

    // #pragma omp parallel for  
    // for (size_t x = 0; x < sizeX; x++){
    //     for (size_t c = 0; c < numClusters; c++){
    //         for (size_t y = 0; y < sizeY; y++){
    //         }
    //     }
    // }
}