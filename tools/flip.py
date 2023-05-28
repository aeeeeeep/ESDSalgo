# -*- coding: UTF-8 -*-

import glob
import os
from PIL import Image

output_path_l = './switch_right'

img_list_l = os.listdir(output_path_l)

for img_path in img_list_l:
    img_name = os.path.splitext(img_path)[0]  # 获取不加后缀名的文件名
    img_name = img_name.replace('right', 'left')
    print(img_name)  # 打印查看文件名
    im = Image.open(output_path_l + '/' + img_path)
    im_flip = im.transpose(Image.FLIP_LEFT_RIGHT)
    # 判断输出文件夹是否已存在，不存在则创建。
    im_flip.save('./switch_left/' + img_name + '.png')

print('所有图片均已旋转完毕，并存入输出文件夹')
