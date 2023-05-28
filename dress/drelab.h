#pragma once
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <pthread.h>
#include "mat2base64.h"
#include "yolov6.h"
#include <iostream>
#include <list>
#include <unistd.h>
#include <string>
using namespace std;
class drelab
{
private:
    json result;
    json message;
public:
    drelab(/* args */);
    void createimage(string str){
        result["image"]=str;
    }
    void createmassage(json message){
        result.update(message);
    }
    string getstring(){
        return result.dump();
    }
    ~drelab();
};

drelab::drelab(/* args */)
{
}

drelab::~drelab()
{
}
