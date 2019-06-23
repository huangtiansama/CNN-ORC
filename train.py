from os import listdir
import cv2 as cv
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
#本程序参考了TensorFlow中文官方教程
img=[]#图片列表
test=[]#测试数据下标列表
#parm
learning_rate=0.001 #学习率
training_iters=20 #训练周期
batch_size=50 #批处理
display_step=50 #迭代次数用于统计精准度
#network
np_input=30*46*3
out_class=11
dropout=0.8
img_input=tf.placeholder(tf.float32,[None,46,30,3],name='img_input')#输入数据
test_input=tf.placeholder(tf.float32,[None,out_class],name='test_input')#训练样本下标
keep_prob=tf.placeholder(tf.float32,name='keep_prob')
def conv2d(x_input,W,b,strides=1):#cnn模型
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x_input,W,strides=[1,strides,strides,1],padding='SAME'),b))
def maxpool2d(x_input,k):
        return tf.nn.max_pool(x_input,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')
def convnet(x,weights,biases,dropout):
        x=tf.reshape(x,shape=[-1,46,30,3])
        conv1= conv2d(x,weights['wc1'],biases['bc1'])
        conv1=maxpool2d(conv1,k=2)
        conv2=conv2d(conv1,weights['wc2'],biases['bc2'])
        conv2=maxpool2d(conv2,k=2)
        fc1=tf.reshape(conv2,shape=[-1,weights['wd1'].get_shape().as_list()[0]])
        fc1=tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
        fc1=tf.nn.relu(fc1)
        fc1=tf.nn.dropout(fc1,dropout)
        return tf.add(tf.matmul(fc1,weights['wd2']),biases['bd2'],name='pred')
weights={#权重
        # conv1 5*5*3,out=64
        'wc1':tf.Variable(tf.random_normal([5,5,3,32]),name='wc1'),#卷积层权重
        # conv2 5*5*32*3,out=64*3
        'wc2':tf.Variable(tf.random_normal([5,5,32,64]),name='wc2'),
        # full connected,out=1024*3
        'wd1':tf.Variable(tf.random_normal([6144,1024]),name='wd1'),#全连接层权重
        'wd2':tf.Variable(tf.random_normal([1024,out_class]),name='wd2')
}
biases={
        # conv1
        'bc1':tf.Variable(tf.random_normal([32]),name='bc1'),
        'bc2':tf.Variable(tf.random_normal([64]),name='bc2'),#卷积层权重
        'bd1':tf.Variable(tf.random_normal([1024]),name='bd1'),#全连接层权重
        'bd2':tf.Variable(tf.random_normal([out_class]),name='bd2'),
}
def split_str(str):#字符串切割用来提取测试数据的正确下标
        for i in range(4,8):
                x=np.zeros((out_class))
                if str[i]=='_':
                       x[out_class-1]=1
                       test.append(x)
                else:
                       x[int(str[i])]=1
                       test.append(x)
def cutimg(img_value):#切割图片
    x=np.zeros((img_value.shape[0],30,img_value.shape[2]))
    for i in range(0,(int)(img_value.shape[1]/30)):
        n=i*30
        for j in range(0,x.shape[0]):
                x[j][0:]=img_value[j][n:n+30]
        img.append(np.array(x))
def img_load(key_list):#加载图片
    with tf.Session() as se:
        for imgname in key_list:
            split_str(imgname)
            image_string = tf.read_file(imgname)
            cutimg( se.run(tf.image.decode_image(image_string)))
def String_add(string_list):
    n=0
    for i in string_list:
        string_list[n]='img/'+i
        n+=1
img_list=listdir('img/')#获取训练数据
String_add(img_list)
img_load(img_list)#加载数据集
x1=tf.constant(np.array(img))
t1=tf.constant(np.array(test))
dataset=tf.data.Dataset.from_tensor_slices((x1, t1))#建立dataset集
datasets=dataset.shuffle(10).batch(batch_size).repeat(training_iters)
pred=convnet(img_input,weights,biases,keep_prob)#推理
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=test_input),name='cost')#计算损失函数
optimizer= tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)#权重优化
correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(test_input,1),name='correct_prediction')
accuracy= tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name='accuracy')#计算精确度
saver=tf.train.Saver(max_to_keep=1)#保存模型
init=tf.global_variables_initializer()
train_loss=[]
training_acc=[]
test_acc=[]
with tf.Session() as sess:#开始训练
        sess.run(init)
        iterator = datasets.make_initializable_iterator()
        init_op = iterator.make_initializer(datasets)
        step=1
        sess.run(init_op)
        iterator = iterator.get_next()
        try:
                while True:
                        x_out,y_out=sess.run(iterator)
                        sess.run(optimizer,feed_dict={img_input:x_out,test_input:y_out,keep_prob:dropout})
                        if step%display_step==0:
                                x_out,y_out=sess.run(iterator)
                                loss_train,acc_train=sess.run([cost,accuracy],feed_dict={img_input:x_out,test_input:y_out,keep_prob:1})
                                train_loss.append(loss_train)
                                training_acc.append(acc_train)

                        step+=1
        except tf.errors.OutOfRangeError:#训练结束,保存结果并绘制训练情况
                saver.save(sess,"save_mode",global_step=step)
                print(step)
                eval_indices=range(0,step*batch_size,display_step*batch_size)
                plt.plot(train_loss,eval_indices[0:len(train_loss)],'k-')
                plt.title('Softmax Loss per iteration')
                plt.xlabel('iteration')
                plt.ylabel('Softmax Loss')
                plt.show()
                plt.plot(training_acc,eval_indices[0:len(training_acc)],'r--',label='Tranin Accuracy')
                plt.title('Train Accuracy')
                plt.xlabel('Genration')
                plt.ylabel('Accuracy')
                plt.legend(loc='lower right')
                plt.show()
