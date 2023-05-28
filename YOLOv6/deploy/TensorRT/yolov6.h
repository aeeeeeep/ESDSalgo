#ifndef YOLOV6_H
#define YOLOV6_H
#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <nlohmann/json.hpp>
#include <cmath>

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

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.5

using namespace nvinfer1;
using json = nlohmann::json;

// stuff we know about the network and the input/output blobs
const int num_class = 6;
static const int INPUT_W = 640;
static const int INPUT_H = 640;
static const char* INPUT_BLOB_NAME = "images";
static const char* OUTPUT_BLOB_NAME = "outputs";
static const char* class_names[] = {
        "helmet", "nohelmet", "glove", "noglove", "coat", "nocoat"
};

cv::Mat static_resize(cv::Mat&);

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static inline float intersection_area(const Object&, const Object&);
static void qsort_descent_inplace(std::vector<Object>&, int, int);
static void qsort_descent_inplace(std::vector<Object>&);
static void nms_sorted_bboxes(const std::vector<Object>&, std::vector<int>&, float);
static void generate_yolo_proposals(float*, int, float, std::vector<Object>&);
float* blobFromImage(cv::Mat&);
static void decode_outputs(float*, int, std::vector<Object>&, float, const int, const int);
static void print_objects(const std::vector<Object>&);
IExecutionContext* Init(char*, int&);
json doInference(IExecutionContext&, cv::Mat, const int);

#endif
