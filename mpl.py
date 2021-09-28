import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np


def showimages(images, titles=None, num_cols=2, axis='off', scale=3, normalize=False):
    """一个窗口显示多张图片
        目前只有简单的显示
        args:
            images:图片
            titles:每张图片标题
            num_cols:每行最多的图片数
            scale:图片尺寸
    """
    plt.figure()
    num = len(titles)
    if num == 1:
        if np.ndim(images) == 2:
            plt.imshow(images, cmap='gray')
        else:
            B, G, R = cv.split(images)
            images = cv.merge([R, G, B])
            plt.imshow(images)
        plt.title(titles)
        plt.axis(axis)
    else:
        for i in range(num):
            img = images[i]
            plt.subplot(num // num_cols + 1, num_cols, i + 1)
            if np.ndim(img) == 2:
                plt.imshow(img, cmap='gray')
            else:
                B, G, R = cv.split(img)
                img = cv.merge([R, G, B])
                plt.imshow(img)
            plt.title(titles[i])
            plt.axis(axis)
    plt.show()


def drawhistogram(data, bins):
    plt.figure()
    n, bins, patches = plt.hist(data, bins=bins, alpha=0.5, histtype='bar')
    plt.show()
