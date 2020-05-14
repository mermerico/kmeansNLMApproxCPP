#include <cstddef>
#include <algorithm>
#include <array>
#include <vector>
#include <cmath>

//A is the input image- called f in the paper, W is called c_k, numClusters is K
//S is the radius of the neighborhood over which we are averaging (neighborhood size (2S+1)^2)
//We're going to column major order right now to match MATLAB, TODO: switch to row major
//TODO: fixup int vs ptrdiff_t indexing.
void highDimBoxFilter(float* const A, float* const W, float* const C1chanT,
                      ptrdiff_t const sizeY, ptrdiff_t const sizeX, ptrdiff_t const numClusters, float const S_ignore,
                      float* output){

    constexpr ptrdiff_t blockSize = 128;
    constexpr signed int S = 8;

    ptrdiff_t numBlocksX = std::ceil(float(sizeX) / float(blockSize));
    ptrdiff_t numBlocksY = std::ceil((float(sizeY) / float(blockSize)));
    ptrdiff_t numBlocks = numBlocksX * numBlocksY;
    #pragma omp parallel for 
    for (ptrdiff_t blockIdx = 0; blockIdx < numBlocks; blockIdx++){
        //We're using column major order across the blocks
        ptrdiff_t xBlockStart = blockSize*(blockIdx/numBlocksY);
        ptrdiff_t yBlockStart = blockSize*(blockIdx % numBlocksY);

        //Don't compute past the end of the image
        ptrdiff_t xIdxEnd = std::min(blockSize,sizeX - xBlockStart);
        ptrdiff_t yIdxEnd = std::min(blockSize,sizeY - yBlockStart);

        //Calculate the weighting (Wt) in this block across all clusters to preserve locallity. TODO: test moving this into the per-cluster for loop
        std::vector<float> Wt(blockSize*blockSize*numClusters, 0.0f);
        for (ptrdiff_t xIdx = 0; xIdx < xIdxEnd; xIdx++){
            for (ptrdiff_t destClusterIdx = 0; destClusterIdx < numClusters; destClusterIdx++){
                for(ptrdiff_t sourceClusterIdx = 0; sourceClusterIdx < numClusters; sourceClusterIdx++){
                    for(ptrdiff_t yIdx = 0; yIdx < yIdxEnd; yIdx++){
                        ptrdiff_t Wt_idx = yIdx + xIdx*blockSize + destClusterIdx*blockSize*blockSize;
                        ptrdiff_t C1chanT_idx = sourceClusterIdx + destClusterIdx*numClusters;
                        ptrdiff_t W_idx = (yBlockStart+yIdx) + (xBlockStart+xIdx)*sizeY + sourceClusterIdx*sizeY*sizeX;

                        Wt[Wt_idx] += W[W_idx]*C1chanT[C1chanT_idx];
                    }
                }
            }
            
        }

        //Calculate sum (across clusters) of Wt*boxFilter(W) and Wt*boxFilter(W*A), yield Wb and B
        std::vector<float> W_summedAreaTable(blockSize*(blockSize+(2*S)), 0.0f);
        std::vector<float> WA_summedAreaTable(blockSize*(blockSize+(2*S))*3, 0.0f);
        std::vector<float> Wb(blockSize*blockSize, 0.0f);
        std::vector<float> B(blockSize*blockSize*3, 0.0f);
        for (ptrdiff_t clusterIdx = 0; clusterIdx < numClusters; clusterIdx++){

            //We need to blur in both the x and y directions, but only the x direction is vectorizeable
            //Therefore, we first do a blur in the x direction and store it in a temporary array.
            //Then we calculate the cumulative sum (summed area table) in the y direction using scalars
            //Lastly, when we read out the values, we will do the appropriate subtractions to calculate the blur in the y direction

            //First create temporary summed area table for a blur in the x direction
            std::fill(W_summedAreaTable.begin(), W_summedAreaTable.end(), 0);//TODO: I probably only need to zero out the first column
            std::fill(WA_summedAreaTable.begin(), WA_summedAreaTable.end(), 0); //TODO: might want to set to epsilon instead of 0 

            
            //Preamble: generate the first column of the summed area table in the x direction.
            //TODO: When I make this code row major, change comment to say first row.
            for (int xReadIdx = -S; xReadIdx < (1+S); xReadIdx++){
                for (ptrdiff_t c = 0; c < 3; c++){
                    for (int yReadIdx = -S; yReadIdx < (int(blockSize)+S); yReadIdx++){
                        ptrdiff_t xWriteIdx = 0;
                        ptrdiff_t yWriteIdx = yReadIdx + S;

                        int yImage = int(yBlockStart)+yReadIdx;
                        int xImage = int(xBlockStart)+xReadIdx;
                        if ((xImage < 0) | (xImage >= int(sizeX)) | (yImage < 0) | (yImage >= int(sizeY))) //We are trying to read beyond the image dimensions. TODO: work to take this out of the loop? Hopefully the compiler hoists it
                            continue; //Assume area beyond image is equal to 0. TODO: is this faster if I set the addition to 0 instead?

                        ptrdiff_t W_sat_idx = yWriteIdx + xWriteIdx*(blockSize+(2*S));
                        ptrdiff_t W_idx = yImage + xImage*sizeY + clusterIdx*sizeY*sizeX;
                        if (c == 0){ //W is a single channel, so we only have to compute it once
                            W_summedAreaTable[W_sat_idx] += W[W_idx];
                        }

                        ptrdiff_t A_idx = yImage + xImage*sizeY + c*sizeY*sizeX;
                        ptrdiff_t WA_sat_idx = W_sat_idx + c*blockSize*(blockSize+(2*S));
                        WA_summedAreaTable[WA_sat_idx] += W[W_idx]*A[A_idx];
                    }
                }
            }

            //Main: generate the rest of the summed area table in the x direction.
            //TODO: When I make this code row major, change comment to say first row.
            for (int xReadPosIdx = 1+S; xReadPosIdx < (int(blockSize)+S); xReadPosIdx++){
                for (ptrdiff_t c = 0; c < 3; c++){
                    for (int yReadIdx = -S; yReadIdx < (int(blockSize)+S); yReadIdx++){
                        ptrdiff_t xWriteIdx = xReadPosIdx - S;
                        ptrdiff_t yWriteIdx = yReadIdx + S;

                        int yImage = int(yBlockStart) + yReadIdx;
                        int xImageReadCenter = int(xBlockStart) + xReadPosIdx-S;
                        int xImageReadAdd = xImageReadCenter + S;
                        int xImageReadSub = xImageReadCenter - S - 1;

                        ptrdiff_t W_sat_idx_write = yWriteIdx + xWriteIdx*(blockSize+(2*S));
                        ptrdiff_t W_sat_idx_read = W_sat_idx_write - (blockSize+(2*S)); //read from previous column
                        ptrdiff_t W_idx_add = yImage + xImageReadAdd*sizeY + clusterIdx*sizeY*sizeX;
                        ptrdiff_t W_idx_sub = yImage + xImageReadSub*sizeY + clusterIdx*sizeY*sizeX;

                        ptrdiff_t WA_sat_idx_write = W_sat_idx_write + c*blockSize*(blockSize+(2*S));
                        ptrdiff_t WA_sat_idx_read = WA_sat_idx_write - (blockSize+(2*S));
                        ptrdiff_t A_idx_add = yImage + xImageReadAdd*sizeY + c*sizeY*sizeX;
                        ptrdiff_t A_idx_sub = yImage + xImageReadSub*sizeY + c*sizeY*sizeX;
                        if ((xImageReadCenter >= int(sizeX)) | (yImage < 0) | (yImage >= int(sizeY))){ //We are trying to read beyond the image dimensions. TODO: work to take this out of the loop? Hopefully the compiler hoists it
                            continue; //Keep values at 0
                        }
                        else if (xImageReadSub < 0){
                            if (c == 0){
                                W_summedAreaTable[W_sat_idx_write] = W_summedAreaTable[W_sat_idx_read] + W[W_idx_add];
                            }
                            WA_summedAreaTable[WA_sat_idx_write] = WA_summedAreaTable[WA_sat_idx_read] + W[W_idx_add]*A[A_idx_add];
                        }
                        else if (xImageReadAdd >= int(sizeX)){
                            if (c == 0){
                                W_summedAreaTable[W_sat_idx_write] = W_summedAreaTable[W_sat_idx_read] - W[W_idx_sub];
                            }
                            WA_summedAreaTable[WA_sat_idx_write] = WA_summedAreaTable[WA_sat_idx_read] - W[W_idx_sub]*A[A_idx_sub];
                        }
                        else {
                            if (c == 0){
                                W_summedAreaTable[W_sat_idx_write] = W_summedAreaTable[W_sat_idx_read] + W[W_idx_add] - W[W_idx_sub];
                            }
                            WA_summedAreaTable[WA_sat_idx_write] = WA_summedAreaTable[WA_sat_idx_read] + W[W_idx_add]*A[A_idx_add] - W[W_idx_sub]*A[A_idx_sub];
                        }
                    }     
                }       
            }
            

            //Cumulative sum in the vertical direction. TODO: change this comment when flipping order
            for (ptrdiff_t c = 0; c < 3; c++){
                for (ptrdiff_t xIdx = 0; xIdx < blockSize; xIdx++){            
                    for (ptrdiff_t yWriteIdx = 1; yWriteIdx < blockSize+2*S; yWriteIdx++){ // the cumsum of the first row is equal to the first row, so skip it. TODO: change this comment when filpping order 
                        
                        ptrdiff_t W_sat_idx = yWriteIdx + xIdx*(blockSize+(2*S));
                        if (c == 0){ //W is a single channel, so we only have to compute it once
                            W_summedAreaTable[W_sat_idx] += W_summedAreaTable[W_sat_idx - 1];
                        }

                        ptrdiff_t WA_sat_idx = W_sat_idx + c*blockSize*(blockSize+(2*S));
                        WA_summedAreaTable[WA_sat_idx] += WA_summedAreaTable[WA_sat_idx - 1];
                    }
                }
            }


            //Weight by Wt and write out to Wb and B
            for (ptrdiff_t c = 0; c < 3; c++){
                for (ptrdiff_t xIdx = 0; xIdx < blockSize; xIdx++){
                    for (ptrdiff_t yIdx = 0; yIdx < blockSize; yIdx++){
                        ptrdiff_t Wt_idx = yIdx + xIdx*blockSize + clusterIdx*blockSize*blockSize;
                        ptrdiff_t Wb_idx = yIdx + xIdx*blockSize;
                        ptrdiff_t B_idx = yIdx + xIdx*blockSize + c*blockSize*blockSize;

                        ptrdiff_t W_sat_idx = yIdx+S + xIdx*(blockSize+2*S);
                        ptrdiff_t WA_sat_idx = yIdx+S + xIdx*(blockSize+2*S) + c*blockSize*(blockSize+2*S);

                        if (yIdx == 0){
                            if (c == 0){
                                Wb[Wb_idx] += Wt[Wt_idx]*W_summedAreaTable[W_sat_idx+S];
                            }
                            B[B_idx] += Wt[Wt_idx]*WA_summedAreaTable[WA_sat_idx+S];
                        }
                        else{
                            if (c == 0){
                                Wb[Wb_idx] += Wt[Wt_idx]*(W_summedAreaTable[W_sat_idx+S] - W_summedAreaTable[W_sat_idx-S-1]); //We read out a blurred value using the summed area table. blur = cumsum(x+r) - cumsum(x-r)
                            }
                            B[B_idx] += Wt[Wt_idx]*(WA_summedAreaTable[WA_sat_idx+S] - WA_summedAreaTable[WA_sat_idx-S-1]);                            
                        }
                    }
                }
            }
        }


        //Divide B by Wb
        for (ptrdiff_t chanIdx = 0; chanIdx < 3; chanIdx++){
            for (ptrdiff_t xIdx = 0; xIdx < xIdxEnd; xIdx++){
                for(ptrdiff_t yIdx = 0; yIdx < yIdxEnd; yIdx++){
                    ptrdiff_t output_idx = (yBlockStart+yIdx) + (xBlockStart+xIdx)*sizeY + chanIdx*sizeY*sizeX;
                    ptrdiff_t Wb_idx = yIdx + xIdx*blockSize;
                    ptrdiff_t B_idx = Wb_idx + chanIdx*blockSize*blockSize;

                    output[output_idx] = B[B_idx] / Wb[Wb_idx];
                }
            }
        }

    }
}