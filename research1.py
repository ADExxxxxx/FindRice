'''
找米粒实验
'''

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# 打印灰度直方图
def grayHist(grayImage):
    plt.hist(grayImage.ravel(), 256)
    plt.show()


if __name__ == "__main__":
    # 原始图像
    img = cv.imread("rice.png")
    kernels = np.ones((5, 5), np.uint8)
    # 腐蚀
    img_erode = cv.erode(img, kernels, iterations=5)
    # 膨胀
    img_dilation = cv.dilate(img_erode, kernels, iterations=5)

    # 通过开运算获取到图像背景，然后使用原始图像减去背景图像
    img_interface = img - img_dilation

    # 转换颜色空间
    gray_img = cv.cvtColor(img_interface, cv.COLOR_BGR2GRAY)

    # 图像二值化
    ret, binary = cv.threshold(gray_img, 60, 255, cv.THRESH_BINARY)

    # 寻找米粒轮廓
    binary, contours, hierarchy = cv.findContours(binary,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, (0, 255, 255), 1)

    list_area = [cv.contourArea(i) for i in contours]
    list_arcLength = [cv.arcLength(i, closed=True) for i in contours]

    list_area = np.array(list_area)
    list_arcLength = np.array(list_arcLength)

    print("最大面积:", np.max(list_area))
    print("最大周长:", np.max(list_arcLength))
    print("最大面积的米粒位于:", np.argmax(list_area))
    print("最大周长的米粒位于:", np.argmax(list_arcLength))

    cv.drawContours(img, contours[np.argmax(list_arcLength)], -1, (0, 0, 255), 3)
    cv.imshow("contours", img)
    cv.waitKey(0)

