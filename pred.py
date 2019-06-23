import numpy as np
import tensorflow as tf
import cv2 as cv
from matplotlib import pyplot as plt
x=[]
q=[]
def resize(img):
    for i in img:
        x.append(cv.resize(i,(30,46)))
def pred(img):
    with tf.Session() as sess:
        resize(img)
        saver = tf.train.import_meta_graph('./save_mode-1707.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
        y =  graph.get_operation_by_name('pred').outputs[0]
        t=tf.nn.softmax(y)
        img_input=graph.get_operation_by_name('img_input').outputs[0]
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
        for i in sess.run(t,feed_dict={img_input:x,keep_prob:1}):
               n=i.argmax(axis=0)
               if n!=10:
                       q.append(n)
        print(q)