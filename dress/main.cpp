#include "yolov6.h"

using namespace nvinfer1;
using json = nlohmann::json;

int main(int argc, char **argv) {
    // cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream

    int output_size = 1;
    IExecutionContext *context = Init(argv[1], output_size);

    cv::Mat img = cv::imread(argv[3]);
    // run inference
    json result = doInference(*context, img, output_size);
    std::cout << result.dump() << std::endl;

    // delete the pointer to the float
    // destroy the engine
    // context->destroy();
    // engine->destroy();
    // runtime->destroy();
    return 0;
}
