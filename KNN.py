import numpy as np
import time

train_feature = np.load('./data/train_img_float.npy')
train_label = np.load('./data/train_label.npy')
test_feature = np.load('./data/test_img_float.npy')
test_label = np.load('./data/test_label.npy')

# 计算正确率
def correct_rate(a, b):
    correct_num = 0
    each_correct = np.zeros((10))
    for pre,label in zip(a,b):
        if pre == label:
            correct_num += 1
    return correct_num/a.shape[0]

# 特征即为每一点的灰度值，采用欧式距离，采用K近邻
# 返回对test_feature的预测
def GrayValue_EuclideanDistance_KNear(train_feature, train_label, test_feature, K):
    train_feature = train_feature.reshape([train_feature.shape[0], -1])[0:100]
    train_label = train_label[0:100]
    test_feature = test_feature.reshape([test_feature.shape[0], -1])
    pre = []
    for count,i in enumerate(test_feature):
        #计算距离
        dis = ((train_feature - i)**2).sum(axis = 1)
        #排序
        sorted_idx = dis.argsort()
        #看前K个中每类的个数
        class_num = np.array([0]*10)
        for i in range(K):
            class_num[train_label[sorted_idx[i]]] += 1
        #得到预测
        pre.append(class_num.argmax())
        if count % 10 == 0:
            print('Predicting...', count/test_feature.shape[0]*100, '% is done.')
    print('Predictation is done.')
    return np.array(pre)

if __name__ == '__main__':
    begin = time.time()
    pre = GrayValue_EuclideanDistance_KNear(train_feature, train_label, test_feature, 3)
    end = time.time()
    print('Correct rate:', correct_rate(pre, test_label))
    print('Time used:', end - begin, 's')
