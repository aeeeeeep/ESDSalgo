import cv2
import os
import pandas as pd

for root, dirs, files in os.walk('./'):
    if dirs != []:
        for file in files:
            if file[-4:] == '.mp4':
                video_path = root + '/' + file
                times=0 
                frameFrequency = 30
                outPutDirName = root + '/images/'
                if not os.path.exists(outPutDirName):
                    os.makedirs(outPutDirName)
                camera = cv2.VideoCapture(video_path)
                while True:
                    times += 1
                    res, image = camera.read()
                    if not res:
                        print('not res , not image')
                        break
                    if times%frameFrequency==0:
                        cv2.imwrite(outPutDirName + file[:-4] + str(times)+'.jpg', image)
                        # print(outPutDirName + file[:-4] + str(times)+'.jpg')
                print('图片提取结束')
                camera.release()
