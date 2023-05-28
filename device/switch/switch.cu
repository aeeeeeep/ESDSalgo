#include "/algo/device/logger.h"
using namespace std;
using namespace cv;
using namespace nvinfer1;
using namespace nvonnxparser;

namespace SWITCH{
        IExecutionContext* Init(char* path, cudaStream_t& stream) {
                char *trtModelStream{nullptr};
                size_t size{0};

                const std::string engine_file_path {path};
                std::ifstream file(engine_file_path, std::ios::binary);
                if (file.good()) {
                        file.seekg(0, file.end);
                        size = file.tellg();
                        file.seekg(0, file.beg);
                        trtModelStream = new char[size];
                        assert(trtModelStream);
                        file.read(trtModelStream, size);
                        file.close();
                }
                IRuntime* runtime = createInferRuntime(gLogger);
                assert(runtime != nullptr);
                ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
                assert(engine != nullptr);
                IExecutionContext* context = engine->createExecutionContext();
                assert(context != nullptr);

                delete[] trtModelStream;
                // Pointers to input and output device buffers to pass to engine.
                // Engine requires exactly IEngine::getNbBindings() number of buffers.
                assert(engine->getNbBindings() == 2);

                // In order to bind the buffers, we need to know the names of the input and output tensors.
                // Note that indices are guaranteed to be less than IEngine::getNbBindings()
                int mBatchSize = engine->getMaxBatchSize();

                // Create stream
                CHECK(cudaStreamCreate(&stream));

                return context;
        }

        float* doInference(IExecutionContext& context, cudaStream_t& stream, cv::Mat img) {
                cv::resize(img, img, Size(64,64));
                float* input = blobFromImage(img);
                float* output = new float[1];

                // Create GPU buffers on device
                void* buffers[2];
                CHECK(cudaMalloc(&buffers[0], 3 * 64 * 64 * sizeof(float)));
                CHECK(cudaMalloc(&buffers[1], 3 * sizeof(float)));

                // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
                CHECK(cudaMemcpyAsync(buffers[0], input, 3 * 64 * 64 * sizeof(float), cudaMemcpyHostToDevice, stream));
                context.enqueue(1, buffers, stream, nullptr);
                CHECK(cudaMemcpyAsync(output, buffers[1], 3 * sizeof(float), cudaMemcpyDeviceToHost, stream));
                cudaStreamSynchronize(stream);
                CHECK(cudaFree(buffers[0]));
                CHECK(cudaFree(buffers[1]));
                free(input);
                return output;
        }
}
