import matplotlib.pyplot as plt
import cv2
import CNN
import numpy as np

pre_x = None
pre_y = None


def mouse_kick(event):
    #print("you moved" ,event.button, event.xdata, event.ydata)
    global pre_x
    global pre_y
    if event.button == 1:
        if pre_x == None:
            pre_x = event.xdata
            pre_y = event.ydata
        else:
            ax.plot([pre_x, event.xdata], [pre_y, event.ydata], color='#000000', linewidth = 15)
            pre_x = event.xdata
            pre_y = event.ydata
            fig.canvas.draw()

def mouse_free(event):
    global pre_x
    global pre_y
    pre_x = None
    pre_y = None

def key_kick(event):
    if event.key == ' ':
        plt.savefig('temp.jpg')
        plt.cla()
        plt.axis('off')
        ax.scatter([0, 0, 1, 1], [0,1,1,0], c = 'w')
        img = cv2.imread('temp.jpg', 0)
        img = 255 - img
        img = cv2.resize(img, (28, 28))
        cv2.imwrite('temp.jpg', img)
        img = np.float32(img)
        img = img/255
        a = CNN.prediction(img.reshape([1,img.shape[0],img.shape[1],1]))[0]
        plt.text(0.5, 0.5, a, fontsize=50)
        plt.show()
    elif event.key == 'a':
        plt.cla()
        plt.axis('off')
        ax.scatter([0, 0, 1, 1], [0,1,1,0], c = 'w')
        plt.show()

fig, ax = plt.subplots(figsize=(4, 4))
plt.axis('off')
ax.scatter([0, 0, 1, 1], [0,1,1,0], c = 'w')

fig.canvas.mpl_connect('motion_notify_event', mouse_kick)
fig.canvas.mpl_connect('button_release_event', mouse_free)
fig.canvas.mpl_connect('key_press_event', key_kick)
plt.show()
