from sklearn.svm import SVC
import numpy as np
import time

if __name__ == '__main__':
    im = np.load('./data/train_img_float.npy')
    im_label = np.load('./data/train_label.npy')

    test = np.load('./data/test_img_float.npy')
    test_label = np.load('./data/test_label.npy')

    im = im.reshape([im.shape[0],-1])
    
    test = test.reshape([test.shape[0], -1])

    print('Model is building......')
    begin = time.time()
    #高斯核
    #clf = SVC(C=5.0, kernel='rbf', gamma=0.05)
    #多项式核
    clf = SVC()
    clf.fit(im, im_label)
    end = time.time() 
    print('Time used:', end-begin)
    print('Model is done.')

    score = clf.score(test, test_label)
    print(" score: {:.6f}".format(score))

