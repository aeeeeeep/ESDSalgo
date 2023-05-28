#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <pthread.h>
#include "mat2base64.h"
#include "yolov6.h"
#include <iostream>
#include <list>
#include <unistd.h>
using websocketpp::lib::bind;
using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;
typedef websocketpp::server<websocketpp::config::asio> server;
using namespace std;
using namespace cv;
pthread_mutex_t plock,quelock;
server print_server;
list<Mat> mats;
list<websocketpp::connection_hdl> sers;
void *pushMat(void *argc)
{
    VideoCapture cap;
    cap.open("rtsp://localhost:8554/stream1");
    double fps = cap.get(cv::CAP_PROP_FPS);
    int interval = int(fps + 0.5);
    int count = 0;
    int output_size = 1;
    IExecutionContext* context = Init("./weights/last_ckpt.trt", output_size);
    while (true)
    {
        Mat frame;
        cap >> frame;
        count++;
        json result = doInference(*context, frame, output_size);
        cout<<result.dump()<<std::endl;
        cv::resize(frame, frame, cv::Size(960, 540));
        pthread_mutex_lock(&plock);
        mats.push_back(frame);
        pthread_mutex_unlock(&plock);
        usleep(1000000);
    }
    cap.release();
    return NULL;
}
void on_message(websocketpp::connection_hdl hld, server::message_ptr msg)
{
    // std::cout << msg->get_payload() << std::endl;
}
void onOpen(server *s, websocketpp::connection_hdl hdl)
{
    pthread_mutex_lock(&quelock);
	sers.push_back(hdl);
	pthread_mutex_unlock(&quelock);
}
void *run(void *argc)
{
    print_server.set_open_handler(bind(&onOpen, &print_server, ::_1));
    print_server.set_message_handler(&on_message);
    print_server.set_access_channels(websocketpp::log::alevel::none);
    print_server.set_error_channels(websocketpp::log::elevel::none);
    print_server.init_asio();
    print_server.listen(9002);
    while(true)
    {
        print_server.start_accept();
        print_server.run();
    }
    return NULL;
}
void* sendimg(void* argc){
    websocketpp::lib::error_code ec;
    Mat frame;
    String str;
    while (true)
    {
        pthread_mutex_lock(&plock);
        if (!mats.empty())
        {
            frame = mats.front();
            mats.pop_front();
            pthread_mutex_unlock(&plock);
            str = Mat2Base64(frame, "png");
            for(auto p=sers.begin();p!=sers.end();p++)
		    {
			    print_server.send((*p), str, websocketpp::frame::opcode::text, ec);
		    }
        }
        else
        {
            pthread_mutex_unlock(&plock);
        }
        if (ec)
        {
            break;
        }
    }
    return NULL;
}
int main()
{
    pthread_t pid, pidd,pid1;
    pthread_mutex_init(&plock, NULL);
    pthread_create(&pid, NULL, run, NULL);
    pthread_create(&pidd, NULL, pushMat, NULL);
    pthread_create(&pid1, NULL, sendimg, NULL);
    pthread_join(pid1, NULL);
    pthread_join(pid, NULL);
    pthread_join(pidd, NULL);
    return 0;
}
