# ESDSalgo

Employee Safety Detection System Algorithm part

## 部署指南

状态灯

```shell
cd device/light
g++ -o liblight.so -fPIC -shared ./light.cpp --std=c++17 `pkg-config --cflags --libs opencv4` -O3
```

电源灯

```shell
cd device/power
g++ -o libpower.so -fPIC -shared ./power.cpp --std=c++17 `pkg-config --cflags --libs opencv4` -O3
```

压板开关

```shell
cd tools
python torch2onnx.py
trtexec --onnx=model_plate.onnx --saveEngine=model_plate.trt --explicitBatch --workspace=1024 --best --fp16
cd ../device/plate
nvcc -lib plate.cu -o libplate.so `pkg-config --cflags --libs opencv4`
```

旋转开关

```shell
cd tools
python torch2onnx.py
trtexec --onnx=model_switch.onnx --saveEngine=model_switch.trt --explicitBatch --workspace=1024 --best --fp16
cd ../device/switch
nvcc -lib switch.cu -o libswitch.so `pkg-config --cflags --libs opencv4`
```

设备检测

```shell
cd device
nvcc test.cu -o device -L./plate -L./switch -L./power -L./light -lplate -lswitch -lpower -llight  -lnvinfer -lmysqlclient `pkg-config --cflags --libs opencv4`
```

着装检测

```shell
cd dress
nvcc main.cpp -L./build -lyolov6 -lnvinfer  `pkg-config --cflags --libs opencv4`
```
