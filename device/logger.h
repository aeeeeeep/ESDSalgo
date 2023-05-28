#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"

using namespace std;
using namespace cv;
using namespace nvinfer1;
using namespace nvonnxparser;

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kVERBOSE)
            std::cout << msg << std::endl;
    }
};

static Logger gLogger;

static const int INPUT_W = 64;
static const int INPUT_H = 64;
static const char* INPUT_BLOB_NAME = "input";
static const char* OUTPUT_BLOB_NAME = "output";

float* blobFromImage(cv::Mat& img){
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++)
    {
        for (size_t  h = 0; h < img_h; h++)
        {
            for (size_t w = 0; w < img_w; w++)
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (((float)img.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    return blob;
}
