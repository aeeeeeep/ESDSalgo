import cv2
import os
import pandas as pd

dt = pd.read_excel('/home/aep/Datasets/FWWB/补充仪表视频数据/仪表视频数据清单.xlsx')
videos = dt.iloc[:,0].tolist()

for video_name in videos: 
    # video_name = videos[0]
    video_path = '/home/aep/Datasets/FWWB/补充仪表视频数据/' + video_name
    i = 0
    flag = True
    frameFrequency = 1800
    while flag:
        i += 1
        times=0
        camera = cv2.VideoCapture(video_path)
        res, image = camera.read()
        roi = cv2.selectROI(windowName="original", img=image, showCrosshair=True, fromCenter=False)
        x, y, w, h = roi
        cv2.destroyAllWindows()

        print("select class name:")
        print("0) exit")
        print("1) switch_left")
        print("2) switch_center")
        print("3) switch_right")
        print("4) plate_open")
        print("5) plate_close")
        cls_list = ['switch_left','switch_center','switch_right','plate_open','plate_close']
        cls_select = int(input("input class select: "))
        if cls_select == 0:
            flag = False
            continue
        cls_name = cls_list[cls_select-1]
        if not os.path.exists(cls_name):
            os.makedirs(cls_name)

        while True:
            times += 1800
            res, image = camera.read()
            if not res:
                print('not res , not image')
                break
            if times % frameFrequency==0:
                if roi != (0, 0, 0, 0):
                    crop = image[y:y+h, x:x+w]
                    crop = cv2.resize(crop, dsize=(64,64), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(cls_name + '/' + video_name[:-4] + '_' + str(times)+ '_' + cls_name + '_' + str(i) + '.png', crop)
                    print(cls_name + '/' + video_name[:-4] + '_' + str(times)+ '_' + cls_name + '_' + str(i) + '.png')

        print('图片提取结束')
        camera.release()
