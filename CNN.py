import tensorflow as tf
import numpy as np
import os

#配置神经网络参数
INPUT_NODE = 28*28
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

FC_SIZE = 512

BATH_SIZE = 100
#学习率设置
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 8000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'model.ckpt'

train_img = np.load('./data/train_img_float.npy')
train_img = train_img.reshape((train_img.shape[0], train_img.shape[1], train_img.shape[2], 1))
train_label = np.load('./data/train_label_one_hot.npy')
test_img = np.load('./data/test_img_float.npy')
test_img = test_img.reshape((test_img.shape[0], test_img.shape[1], test_img.shape[2], 1))
test_label = np.load('./data/test_label.npy')

#定义前向传播
#regulatizer, graph
def inference(img, regularizer, train, flag):
    with tf.variable_scope('layer1-conv1', reuse = flag):
        #[CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP]，注意对多维矩阵的理解
        conv1_weights = tf.get_variable('weight', [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('bias', [CONV1_DEEP], initializer=tf.constant_initializer(0.0))

        conv1 = tf.nn.conv2d(img, conv1_weights, strides=[1,1,1,1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    with tf.variable_scope('layer3-conv2', reuse = flag):
        conv2_weights = tf.get_variable('weight', [CONV1_SIZE, CONV1_SIZE, CONV1_DEEP, CONV2_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias', [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1,1,1,1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    with tf.variable_scope('layer5-fc1', reuse = flag):
        fc1_weights = tf.get_variable('weight', [nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:    
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias', [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights)+fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2', reuse = flag):
        fc2_weights = tf.get_variable('weight', [FC_SIZE, NUM_LABELS], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias', [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
    
    return logit
 
def train(train_img, train_label):
    img = tf.placeholder(tf.float32, [BATH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], name = 'x-input')
    label = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name = 'label')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    pre = inference(img, regularizer, True)
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = pre, labels = tf.argmax(label, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, train_img.shape[0]/BATH_SIZE, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'model/model.ckpt')
        # tf.initialize_all_variables().run()
        for i in range(TRAINING_STEPS):
            xs, ys = train_img[(i*BATH_SIZE)%train_img.shape[0]:((i+1)*BATH_SIZE-1)%train_img.shape[0]+1], train_label[(i*BATH_SIZE)%train_img.shape[0]:((i+1)*BATH_SIZE-1)%train_img.shape[0]+1]
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={img:xs, label:ys})
            if i % 100 == 0:
                print('After %d training step(s), loss on training batch is %g.'%(step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))

flag = None
def prediction(test_img):
    global flag
    img = tf.placeholder(tf.float32, [test_img.shape[0], IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], name = 'x-input')
    label = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name = 'label')

    pre = inference(img, None, False, flag)
    flag = True
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'model/model.ckpt')    #存在就从模型中恢复变量
        pre = tf.argmax(pre, 1)
        pre = sess.run(pre,feed_dict={img:test_img})
    return pre

def correct_rate(pre, label):
    print(pre)
    print(label)
    count = 0
    for a,b in zip(pre, label):
        if a == b:
            count += 1
    return count/pre.shape[0]

if __name__ == '__main__':
    #训练完后自动存储模型，然后注释掉train，运行pre和correct
    train(train_img, train_label)
    # pre = prediction(test_img)
    # print(correct_rate(pre, test_label))
