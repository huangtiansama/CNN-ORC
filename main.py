import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from tkinter import *
import tkinter.filedialog
import a
import pred
def xz():
    filename=tkinter.filedialog.askopenfilename()
    img=a.imghandle(filename)
    pred.pred(img)
def en():
    filename=tkinter.filedialog.askopenfilenames()
    a.enhance(filename)
root = Tk()
root.title('demo')
root.geometry('640x480')
lb = Label(root,text='')
lb.pack()
btn1=Button(root,text='数据增强',command=en)
btn2=Button(root,text='图片识别',command=xz)
btn2.pack()
btn1.pack()
root.mainloop()