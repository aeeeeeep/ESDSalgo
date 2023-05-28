#include"plate/plate.cu"
#include"switch/switch.cu"
#include"light/light.h"
#include"power/power.h"
#include"MysqlConn.cpp"
#include<nlohmann/json.hpp>
#include<iostream>
using namespace std;
using namespace nlohmann;
int main(){
    string _plate= "plate";
    string _switch= "switch";
    string _power= "power";
    Scalar lower_red = Scalar(11, 175, 200);
    Scalar upper_red = Scalar(22, 255, 255);
    Scalar lower_red1 = Scalar(0, 43, 46);
    Scalar upper_red1 = Scalar(3, 255, 255);
    Scalar lower_red2 = Scalar(156, 43, 46);
    Scalar upper_red2 = Scalar(180, 255, 255);

    cudaStream_t stream_plate;
    CHECK(cudaStreamCreate(&stream_plate));
    cudaStream_t stream_switch;
    CHECK(cudaStreamCreate(&stream_switch));
    IExecutionContext* context_plate = PLATE::Init("/algo/tools/model_plate.trt", stream_plate);
    IExecutionContext* context_switch = SWITCH::Init("/algo/tools/model_switch.trt", stream_switch);

    MysqlConn mysqlConn;
    mysqlConn.connect();
    list<list<string>> lists;
    vector<string> devices_str;
    vector<json> devices_json;
    mysqlConn.sqlQuery("select type,box from Image",lists);
    for (auto i = lists.begin(); i != lists.end(); i++) {
        string str = *(i->begin());
        json box = json::parse(*(i->rbegin()));
        devices_str.push_back(str);
        devices_json.push_back(box);
    }
    VideoCapture capture;
    capture.open("rtsp://192.168.10.105:8554/stream1");
    Mat frame;
    int height;
    int width;
    int xMin;
    int yMin;
    Mat crop_img;
    float* output;
    
    while(1) {
    	capture>>frame;
    	if (frame.empty()) {
	    break;
    	}
    	size_t i=0;
	vector<int> status;
    	while(i < devices_str.size()) {
	    height = devices_json[i]["height"];
	    width = devices_json[i]["width"];
	    xMin = devices_json[i]["xMin"];
	    yMin = devices_json[i]["yMin"];
	    Rect crop_region(xMin, yMin, width, height);
	    crop_img = frame(crop_region);
	    resize(crop_img, crop_img, Size(64,64));
	    if(devices_str[i] == _plate) { // plate
	    	output = PLATE::doInference(*context_plate, stream_plate, crop_img);
	    	status.push_back(std::max_element(output, output + 2) - output);
	    } else if(devices_str[i] == _switch) { // switch
	    	output = SWITCH::doInference(*context_switch, stream_switch, crop_img);
	    	status.push_back(std::max_element(output, output + 3) - output);
	    } else if(devices_str[i] == _power) { // power
	    	if(power(crop_img, lower_red, upper_red)) {
	    		status.push_back(1);
	    	} else {
	    		status.push_back(0);
	    	}
	    } else { // light
	    	if(light(crop_img,lower_red1,upper_red1,lower_red2,upper_red2)) {
	    		status.push_back(1);
	    	} else {
	    		status.push_back(0);
	    	}
	    }
	    i++;
    	}
	for(int j=0; j<status.size(); j++) {
		cout<<status[j]<<" ";
	}
	cout<<endl;
    }
    return 0;
}
