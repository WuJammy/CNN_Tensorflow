import numpy as np
import tensorflow as tf
from PIL import Image  
#download mnist datasets
#55000 * 28 * 28 55000image
#from tensorflow.examples.tutorials.mnist import input_data
import random
import numpy as np
import pandas as pd



def ResizeAndConToNumpyArray(pic,X_size,Y_size):
    gray = pic.convert('L')
    gray_rz = gray.resize((X_size,Y_size), Image.ANTIALIAS)
    np_gray= np.array(gray_rz,float)
    np_gray = np_gray.ravel().reshape(1,-1)
    np_gray /= 255
    return np_gray

train =[]
train_label =[]
#train.append()
path = 'C:\\cnn\\training-images'
seq = {'0','1','2','3','4','5','6','7','8','9'}

for i ,element in enumerate(seq):
    for j in range(84):
        num = str(j+1)
        trainpath =path+'\\'+element+'\\'+' '+'('+num+')'+'.jpg'
        
        img = Image.open(trainpath)
        
        train.append(ResizeAndConToNumpyArray(img, 28, 28))
        train_label.append(int(element))
        
        train_lebal_array =np.array(train_label)



data_dum = pd.get_dummies(train_lebal_array)
pd.DataFrame(data_dum)

test =[]
test_lebal =[]
path_test ='C:\\cnn\\test-images'

for i in range(10):
    num_test = str(i)
    testpath = path_test+'\\'+'t'+num_test+'.jpg'
    test_lebal.append(i)
    
    img_test = Image.open( testpath)
    test.append(ResizeAndConToNumpyArray(img_test, 28, 28))
    
    


#mnist=input_data.read_data_sets("C:\\msint_data",one_hot=True)#引數一：檔案目錄。引數二：是否為one_hot向量

#one_hot is encoding format
#None means tensor 的第一維度可以是任意維度
#/255. 做均一化
input_x=tf.placeholder(tf.float32,[None,28*28])/255.
#輸出是一個one hot的向量
output_y=tf.placeholder(tf.int32,[None,10])

#輸入層 [28*28*1]
input_x_images=tf.reshape(input_x,[-1,28,28,1])
#從(Test)資料集中選取3000個手寫數字的圖片和對應標籤

#test_x=mnist.test.images[:10] #image
#test_y=mnist.test.labels[:10] #label



#隱藏層
#conv1 5*5*32
#layers.conv2d parameters
#inputs 輸入，是一個張量
#filters 卷積核個數，也就是卷積層的厚度
#kernel_size 卷積核的尺寸
#strides: 掃描步長
#padding: 邊邊補0 valid不需要補0，same需要補0，為了保證輸入輸出的尺寸一致,補多少不需要知道
#activation: 啟用函式
conv1=tf.layers.conv2d(
    inputs=input_x_images,
    filters=32,
    kernel_size=[5,5],
    strides=1,
    padding='same',
    activation=tf.nn.relu
)
print(conv1)

#輸出變成了 [28*28*32]

#pooling layer1 2*2
#tf.layers.max_pooling2d
#inputs 輸入，張量必須要有四個維度
#pool_size: 過濾器的尺寸

pool1=tf.layers.max_pooling2d(
    inputs=conv1,
    pool_size=[2,2],
    strides=2
)
print(pool1)
#輸出變成了[?,14,14,32]

#conv2 5*5*64
conv2=tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5,5],
    strides=1,
    padding='same',
    activation=tf.nn.relu
)

#輸出變成了  [?,14,14,64]

#pool2 2*2
pool2=tf.layers.max_pooling2d(
    inputs=conv2,
    pool_size=[2,2],
    strides=2
)

#輸出變成了[?,7,7,64]

#flat(平坦化)
flat=tf.reshape(pool2,[-1,7*7*64])


#形狀變成了[?,3136]

#densely-connected layers 全連線層 1024
#tf.layers.dense
#inputs: 張量
#units： 神經元的個數
#activation: 啟用函式
dense=tf.layers.dense(
    inputs=flat,
    units=1024,
    activation=tf.nn.relu
)

#輸出變成了[?,1024]
print(dense)

#dropout
#tf.layers.dropout
#inputs 張量
#rate 丟棄率
#training 是否是在訓練的時候丟棄
dropout=tf.layers.dropout(
    inputs=dense,
    rate=0.5,
)
print(dropout)

#輸出層，不用啟用函式（本質就是一個全連線層）
logits=tf.layers.dense(
    inputs=dropout,
    units=10
)
#輸出形狀[?,10]
print(logits)

#計算誤差 cross entropy（交叉熵），再用Softmax計算百分比的概率
#tf.losses.softmax_cross_entropy
#onehot_labels: 標籤值
#logits: 神經網路的輸出值
loss=tf.losses.softmax_cross_entropy(onehot_labels=output_y,
                                     logits=logits)
# 用Adam 優化器來最小化誤差,學習率0.001 類似梯度下降
print(loss)
train_op=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)


#精度。計算預測值和實際標籤的匹配程度
#tf.metrics.accuracy
#labels：真實標籤
#predictions: 預測值
#Return: (accuracy,update_op)accuracy 是一個張量準確率，update_op 是一個op可以求出精度。
#這兩個都是區域性變數
accuracy_op=tf.metrics.accuracy(
    labels=tf.argmax(output_y,axis=1),
    predictions=tf.argmax(logits,axis=1)
)[1] #為什麼是1 是因為，我們這裡不是要準確率這個數字。而是要得到一個op

#建立會話
sess=tf.Session()
#初始化變數
#group 把很多個操作弄成一個組
#初始化變數，全域性，和區域性
init=tf.group(tf.global_variables_initializer(),
              tf.local_variables_initializer())
sess.run(init)

for i in range(50000):
    #batch=mnist.train.next_batch(50)  #從Train（訓練）資料集中取‘下一個’樣本
    #train_loss,train_op_=sess.run([loss,train_op],{input_x:batch[0],output_y:batch[1]})
    x=random.randint(0,839)
   # a=train[0]
    train_loss,train_op_=sess.run([loss,train_op],{input_x:train[x],output_y:data_dum.loc[[x]]})
    if i%1000==0:
      print(i)
      #  test_accuracy=sess.run(accuracy_op,{input_x:test_x,output_y:test_y})

        # print("Step=%d, Train loss=%.4f,[Test accuracy=%.2f]"%(i,train_loss,test_accuracy))

#測試： 列印10個預測值和真實值 
for i in range(10):
    test_output=sess.run(logits,{input_x:test[i]})
    inferenced_y=np.argmax(test_output,1)
    print('預測值t',i,':',inferenced_y)
    
  
    
    

