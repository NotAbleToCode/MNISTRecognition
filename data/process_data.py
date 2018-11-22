import numpy as np

def read_label_file(path, savepath):
    f = open(path, 'rb')
    _ = int.from_bytes(f.read(4), byteorder = 'big')
    num = int.from_bytes(f.read(4), byteorder = 'big')
    label = []
    for _ in range(num):
        label.append(int.from_bytes(f.read(1), byteorder = 'big'))
    f.close()
    label = np.array(label)
    np.save(savepath, label)

def read_img_file(path, savepath):
    f = open(path, 'rb')
    _ = f.read(4)
    num = int.from_bytes(f.read(4), byteorder = 'big')
    row = int.from_bytes(f.read(4), byteorder = 'big')
    col = int.from_bytes(f.read(4), byteorder = 'big')
    img = np.empty((num, row, col), dtype = np.uint8)
    for count in range(num):
        for i in range(row):
            for q in range(col):
                img[count][i][q] = int.from_bytes(f.read(1), byteorder = 'big')
    np.save(savepath, img)
    f.close()

def binary_img(filepath, savepath):
    img = np.load(filepath)
    print(img.shape)
    img = (img > 128)*1
    np.save(savepath, img)

def float_img(filepath, savepath):
    img = np.load(filepath)
    print(img.shape)
    img = np.float32(img)
    img = img / 255
    np.save(savepath, img)

def one_hot_label(filepath, savepath):
    labels = np.load(filepath)
    one_hot = np.zeros((labels.shape[0],10))
    for i, label in enumerate(labels):
        one_hot[i][label] += 1
    np.save(savepath, one_hot)

#读取并以npy文件存储
read_label_file('./train-labels.idx1-ubyte', 'train_label.npy')
read_label_file('./t10k-labels.idx1-ubyte', 'test_label.npy')
read_img_file('./train-images.idx3-ubyte', 'train_img.npy')
read_img_file('./t10k-images.idx3-ubyte', 'test_img.npy')

#将train_img和test_img二值化，以128为阈值
binary_img('./train_img.npy', './train_img_binary.npy')
binary_img('./test_img.npy', './test_img_binary.npy')

#将图片归一化为0到1区间内
float_img('./train_img.npy', './train_img_float.npy')
float_img('./test_img.npy', './test_img_float.npy')

#将label用one_hot形式给出，CNN用
one_hot_label('train_label.npy', 'train_label_one_hot.npy')
one_hot_label('test_label.npy', 'test_label_one_hot.npy')

