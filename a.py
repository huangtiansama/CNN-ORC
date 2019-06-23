import numpy as np
import cv2 as cv
import math
import tensorflow as tf
import random
from os import listdir
from matplotlib import pyplot as plt
def enhance(load):#数据增强模块
    with tf.Session() as sess:
        for i in load:
            for s in range(0,80):
                raw_img = tf.gfile.FastGFile(i,'rb').read()
                n=random.randint(0,11)
                img_data = tf.image.decode_image(raw_img)
                if n==0:         #随机进行翻转,裁剪,缩放,调整对比度,色调,亮度
                    img_data=np.rot90(sess.run(img_data))
                    strload=i[0:i.find('.',-5,-1)-1]+'_'+str(s)+str(n)+'.jpg'
                    cv.imwrite(strload,img_data.eval()) 
                elif n==1:
                    img_data = tf.image.rgb_to_grayscale(img_data)
                elif n==2:
                    img_data = tf.image.convert_image_dtype(img_data, tf.float32)
                    img_data = tf.image.adjust_brightness(img_data, delta=-.7)
                elif n==3:
                    img_data = tf.image.convert_image_dtype(img_data, tf.float32)
                    img_data = tf.image.random_brightness(img_data, max_delta=0.6)
                elif n==4:
                    img_data = tf.image.convert_image_dtype(img_data, tf.float32)
                    img_data = tf.image.random_contrast(img_data, lower=0, upper=4)
                elif n==5:
                    img_data = tf.image.convert_image_dtype(img_data, tf.float32)
                    img_data = tf.image.random_hue(img_data, 0.5)
                elif n==6:
                    img_data = tf.image.convert_image_dtype(img_data, tf.float32)
                    img_data = tf.image.random_saturation(img_data, lower=0, upper=2)
                elif n==7:
                    img_data = tf.image.central_crop(sess.run(img_data),random.random())
                elif n==8:
                    img_data = tf.image.resize_image_with_pad(img_data,random.randint(sess.run(tf.shape(img_data))[0]/2,sess.run(tf.shape(img_data))[0]*2),random.randint(sess.run(tf.shape(img_data))[1]/2,sess.run(tf.shape(img_data))[1]*2))
                elif n==9:
                    img_data = tf.image.flip_left_right(img_data) 
                elif n== 10:
                    img_data = tf.image.flip_up_down(img_data)
                img_data = tf.image.convert_image_dtype(img_data, tf.int16)
                strload=i[0:i.find('.',-5,-1)-1]+'_'+str(s)+str(n)+'.jpg'
                cv.imwrite(strload,img_data.eval())
def cutimg(img_value,ROI_w,ROI_h,ROI_x,ROI_y,type):#裁剪图片
    img=[]
    t=0
    for i in range(0,math.ceil(ROI_w/25)):
        if type!=3 and i%4==0 and i>0:
                t+=10
        n=i*25+t    
        x=np.zeros((ROI_h,25,img_value.shape[2]),dtype=np.int16)
        for j in range(0,ROI_h):
            if ROI_w-n<25:
                return img
            else :
                x[j][0:]=img_value[ROI_y+j][n+ROI_x:n+ROI_x+25]
        img.append(x)
    return img
def tool1(type,imgout,kernel,light_num,thresholdvalue):#卡号定位处理
    num=t=0
    ROI_w=ROI_h=ROI_x=ROI_y=0
    if type==1:
        retval, dst=cv.threshold(imgout,thresholdvalue+light_num,255,cv.THRESH_BINARY)
    elif type==2:
        retval, dst=cv.threshold(imgout,thresholdvalue-15,255,cv.THRESH_BINARY)
    elif type==3:
        retval, dst=cv.threshold(imgout,thresholdvalue-light_num-30,255,cv.THRESH_BINARY_INV)
    dst = cv.morphologyEx(dst,cv.MORPH_GRADIENT,kernel)
    contours, hierarchy=cv.findContours(dst,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    for i in range(0,len(contours)):  
        x, y, w, h = cv.boundingRect(contours[i]) 
        # print('ROI:',ROI_x,ROI_y,ROI_w,ROI_h,"\n")
        if w>150 and h>120 and y>300:
            ROI_y=0
            num=0
            t=10
            continue
        # print(x,y,w,h,"\n")
        if y+h <= 480*0.75-t and y>=200 and h<=46:
            if ROI_y==0:  
                ROI_h=46
                ROI_y=y-(60-h)
                ROI_x=x 
                ROI_w=w
            elif y>=ROI_y and y+h<=ROI_y+46:
                    if x>ROI_x:
                        if x>ROI_x+ROI_w:
                            ROI_w=x-ROI_x+w
                    else:
                        ROI_w+=ROI_x-x
                        ROI_x=x
                    num+=(ROI_h/20+1)*(ROI_w/30+1)
            elif ROI_w/640>0.7 and num>20:
                break
            else :
                ROI_h=46
                ROI_y=y-(46-h)
                ROI_x=x
                ROI_w=w
                num=0
        else:
            continue
    return ROI_w,ROI_h,ROI_x,ROI_y,num
def imghandle(img_name):#图片处理
    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img=cv.resize(img,(640,480))#准备参数
    imgout = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    thresholdvalue=imgout[20,20]
    light_num=30
    kernel = np.ones((9,9), np.uint8)
    print(cv.__version__,thresholdvalue)
    if thresholdvalue>190:
        light_num=50
    elif thresholdvalue>200:
        light_num=100
    dst = cv.morphologyEx(imgout,cv.MORPH_TOPHAT,kernel)#形态处理
    gradX = cv.Sobel(dst,cv.CV_32F,1,0,-1)
    gradX = np.absolute(gradX)
    gradX = (255 * ((gradX -np.min(gradX)) / (np.max(gradX) -np.min(gradX))))  
    dst = gradX.astype("uint8")#梯度处理
    dst = cv.morphologyEx(dst,cv.MORPH_CLOSE,kernel)
    retval, dst=cv.threshold(imgout,thresholdvalue+light_num,255,cv.THRESH_BINARY)
    ROI_w,ROI_h,ROI_x,ROI_y,num=tool1(1,imgout,kernel,light_num,thresholdvalue)#卡号定位处理
    if ROI_w/640>0.7 and num>20:
        ROI_w+=abs(640-ROI_w-ROI_x)
        ROI_x-=10
        handle=cutimg(img,ROI_w,ROI_h,ROI_x,ROI_y,1)
        cv.rectangle(img, (ROI_x,ROI_y), (ROI_x+ROI_w,ROI_y+ROI_h), (165,165,255), 2)
        plt.imshow(img)
        plt.show()
        return handle
    ROI_w,ROI_h,ROI_x,ROI_y,num=tool1(2,imgout,kernel,light_num,thresholdvalue)
    if ROI_w/640>0.7 and num>20:
        ROI_w+=abs(640-ROI_w-ROI_x)
        ROI_x-=10
        handle=cutimg(img,ROI_w,ROI_h,ROI_x,ROI_y,2)
        cv.rectangle(img, (ROI_x,ROI_y), (ROI_x+ROI_w,ROI_y+ROI_h), (165,165,255), 2)
        plt.imshow(img)
        plt.show()
        return handle
    ROI_w,ROI_h,ROI_x,ROI_y,num=tool1(3,imgout,kernel,light_num,thresholdvalue)
    if ROI_w/640>0.7 and num>20:
        ROI_w+=abs(640-ROI_w-ROI_x)
        ROI_x-=10
        handle=cutimg(img,ROI_w,ROI_h,ROI_x,ROI_y,3)
        cv.rectangle(img, (ROI_x,ROI_y), (ROI_x+ROI_w,ROI_y+ROI_h), (165,165,255), 2)
        plt.imshow(img)
        plt.show()
        return handle#返回裁剪好的数字列表
    return 0,0,0,0