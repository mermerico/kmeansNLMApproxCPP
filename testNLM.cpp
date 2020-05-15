// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use of the GUI API as well as some 
    aspects of image manipulation from the dlib C++ Library.


    This is a pretty simple example.  It takes a BMP file on the command line
    and opens it up, runs a simple edge detection algorithm on it, and 
    displays the results on the screen.  
*/



#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include "kMeansNLMApprox.cpp"
#include <algorithm>
#include <iostream>
#include <chrono>
//#include <gperftools/profiler.h>

using namespace std;
using namespace dlib;

//  ----------------------------------------------------------------------------

int main(int argc, char** argv)
{
    try
    {
        // make sure the user entered an argument to this program
        if (argc != 3)
        {
            cout << "error, enter and input image and a reference output" << endl;
            return 1;
        }
  
        array2d<rgb_pixel> img;
        load_image(img, argv[1]);

        array2d<rgb_pixel> reference_img;
        load_image(reference_img, argv[2]);

        ptrdiff_t sizeY = img.nr();
        ptrdiff_t sizeX = img.nc();
        ptrdiff_t numChannels = 3;
        std::vector<float> inputImgVector(sizeY * sizeX * numChannels);
        for (int row = 0; row < sizeY; row++) {
            for (int col = 0; col < sizeX; col++) {
                rgb_pixel inPx = img[row][col];
                inputImgVector[row + col*sizeY + 0L*sizeY*sizeX] = float(inPx.red)/256.0f;
                inputImgVector[row + col*sizeY + 1L*sizeY*sizeX] = float(inPx.green)/256.0f;
                inputImgVector[row + col*sizeY + 2L*sizeY*sizeX] = float(inPx.blue)/256.0f;
            }
        }

        constexpr int numClusters = 10;
        float clusterThreshold = 3e-3;
        float h = 0.4184;
        std::vector<float> outputImgVector(sizeY * sizeX * numChannels);

        auto begin = std::chrono::high_resolution_clock::now();
        //ProfilerStart("NLM.prof");
        kMeansNLMApprox(inputImgVector.data(), numClusters, clusterThreshold, h, sizeX, sizeY, outputImgVector.data());
        //ProfilerStop();
        auto end = std::chrono::high_resolution_clock::now();
        auto dur = end - begin;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
        std::cout << "Wall time passed: "
            << ms << " ms.\n";

        array2d<rgb_pixel> output_img(sizeY, sizeX);
        for (int row = 0; row < sizeY; row++) {
            for (int col = 0; col < sizeX; col++) {
                rgb_pixel outPx;
                outPx.red   = std::clamp(outputImgVector[row + col * sizeY + 0L * sizeY * sizeX], 0.0f, 1.0f) * 255;
                outPx.green = std::clamp(outputImgVector[row + col * sizeY + 1L * sizeY * sizeX], 0.0f, 1.0f) * 255;
                outPx.blue  = std::clamp(outputImgVector[row + col * sizeY + 2L * sizeY * sizeX], 0.0f, 1.0f) * 255;
                output_img[row][col] = outPx;
            }
        }

        array2d<rgb_pixel> difference_img(sizeY, sizeX);
        double summedDiff = 0;
        double imageSum = 0;
        for (int row = 0; row < sizeY; row++) {
            for (int col = 0; col < sizeX; col++) {
                rgb_pixel diffPx;
                diffPx.red = 128 + int(reference_img[row][col].red) - int(output_img[row][col].red);
                diffPx.green = 128 + int(reference_img[row][col].green) - int(output_img[row][col].green);
                diffPx.blue = 128 + int(reference_img[row][col].blue) - int(output_img[row][col].blue);
                difference_img[row][col] = diffPx;
                summedDiff += std::abs(double(diffPx.red) + double(diffPx.green) + double(diffPx.blue) - 3 * 128.0);
                imageSum += double(reference_img[row][col].red) + double(reference_img[row][col].green) + double(reference_img[row][col].blue);
            }
        }
        std::cout << "percent difference is " << 100*summedDiff/imageSum << endl;

        image_window diffImgWindow(difference_img, "difference image");
        diffImgWindow.set_size(std::min(sizeY, ptrdiff_t(1000)), std::min(sizeX, ptrdiff_t(1000)));

        image_window my_windowOrig(img, "Original Image");
        image_window my_windowNLM(output_img, "Output Image");
        my_windowOrig.set_size(std::min(sizeY, ptrdiff_t(1000)), std::min(sizeX, ptrdiff_t(1000)));
        my_windowNLM.set_size(std::min(sizeY, ptrdiff_t(1000)), std::min(sizeX, ptrdiff_t(1000)));
        
        my_windowOrig.wait_until_closed();
        my_windowNLM.wait_until_closed();
        diffImgWindow.wait_until_closed();

    }
    catch (exception& e)
    {
        cout << "exception thrown: " << e.what() << endl;
    }
}

//  ----------------------------------------------------------------------------

