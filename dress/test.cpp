#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <pthread.h>
#include "mat2base64.h"
#include "yolov6.h"
#include <iostream>
#include <list>
#include <unistd.h>
#include"./drelab.h"
using websocketpp::lib::bind;
using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;
typedef websocketpp::server<websocketpp::config::asio> server;
using namespace std;
using namespace cv;
pthread_mutex_t plock,quelock;
server print_server;
list<drelab> lists;
std::set<websocketpp::connection_hdl,std::owner_less<websocketpp::connection_hdl>> connections;
#include"./drelab.h"
using namespace std;
void *pushMat(void *argc)
{
    VideoCapture cap;
    cap.open("rtsp://localhost:8554/stream1");
    double fps = cap.get(cv::CAP_PROP_FPS);
    int interval = int(fps + 0.5);
    int count = 0;
    int output_size = 1;
    IExecutionContext* context = Init("./weights/last_ckpt.trt", output_size);
    int temp=0;
    while (true)
    {
        Mat frame;
        cap >> frame;
        if(temp%30==0)
        {   
            string str;
            drelab d;
            json result = doInference(*context, frame, output_size);
            cv::resize(frame, frame, cv::Size(960, 540));
            str = Mat2Base64(frame, "png");
            d.createimage(str);
            d.createmassage(result);
            pthread_mutex_lock(&plock);
            lists.push_back(d);
            pthread_mutex_unlock(&plock);
        }
        temp++;
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
    connections.insert(hdl);
	pthread_mutex_unlock(&quelock);
}
void on_close(websocketpp::connection_hdl hdl)
{
    pthread_mutex_lock(&quelock);
    connections.erase(hdl);
	pthread_mutex_unlock(&quelock);
}
void *run(void *argc)
{
    print_server.set_open_handler(bind(&onOpen, &print_server, ::_1));
    print_server.set_message_handler(&on_message);
    print_server.set_close_handler(&on_close);
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
    String str;
    while (true)
    {
        pthread_mutex_lock(&plock);
        if (!lists.empty())
        {
            drelab result=lists.front();
            lists.pop_front();
            pthread_mutex_unlock(&plock);
            for (auto it = connections.begin(); it != connections.end(); ++it) {
                    print_server.send(*it,result.getstring(),websocketpp::frame::opcode::text, ec);
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
int main(){
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
