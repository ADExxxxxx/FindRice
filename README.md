### 实验目的
##### 1,通过阈值分割将原图像转变为二值图像
##### 2,找出米粒的连通域，数出米粒的数目
##### 3,找出米粒中最大的面积和周长是多少，并给出在图片的位置
![原始图片](https://img-blog.csdnimg.cn/20200404202302636.png#pic_center)


### 实验过程
openCV提供了非常好用的简单全局阈值分割的函数

> ***cv2.threshold(src, thresh, maxval, type, dst=None)***
>[关于threshold函数详解](https://blog.csdn.net/weixin_42296411/article/details/80901080?depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-1&utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-1)

对原始灰度图像进行灰度直方图分析，可以明显看出灰度值分布区间较大，并且有三座峰(这里暂时不考虑多阈值分割问题)，因此难以通过单一阈值进行有效分割，所以应使用 ***OTSU*** 或者 ***TRIANGLE*** 的优化方法。

![原始灰度直方图](https://img-blog.csdnimg.cn/20200404202410231.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyMzA4Mjg3,size_16,color_FFFFFF,t_70#pic_center)

但是如果直接对图像进行二值化处理，得到的效果可能不尽人意(Tips:在二值化之前需要转化为灰度图像，因要使用 ***OTSU*** 或者 ***THRESH*** 进行算法优化，两种优化策略详情见方法链接。而且该实验场景下灰度图像会大大简化实验操作)。

```python
import cv2 as cv
#原始图像
img = cv.imread("rice.png")
#色彩空间转换
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret, otsu = cv.threshold(gray_img, 0, 255, 
cv.THRESH_BINARY | cv.THRESH_OTSU)
ret2, triangle = cv.threshold(gray_img, 0, 255, 
cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
cv.imshow("otsu", otsu)
cv.imshow("triangle", triangle)
cv.waitKey(0)
```
![直接进行阈值分割效果](https://img-blog.csdnimg.cn/20200404202741603.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyMzA4Mjg3,size_16,color_FFFFFF,t_70#pic_center)
受到噪点影响，对后续处理工作将带来极大的不便利，因此首要目标是去除背景噪点。常见的去除噪点的方式有各种滤波器，本次实验中我尝试利用了中值滤波和高斯滤波，虽然能有效去除背景噪点，但是不可避免的会对整个米粒形态造成一定的影响，因此在这里我介绍另外一种方法，通过使用形态学的开操作对图像进行预处理。(关于图像处理中的形态学操作原理有兴趣的可以去查阅 __冈萨雷斯版本的数字图像处理__ )。

>形态学操作其实就是改变物体的形状，比如腐蚀就是”变瘦”，膨胀就是”变胖”

在实验中我将使用开操作(先对图像进行腐蚀再对图像进行膨胀)通过这种方式来获取到理想的图像。
```python
kernels = np.ones((5, 5), np.uint8)
# 腐蚀
img_erode = cv.erode(img, kernels, iterations=5)
cv.imshow("erode_image",img_erode)
# 膨胀
img_dilation = cv.dilate(img_erode, kernels, iterations=5)
cv.imshow("dilation_image", img_dilation)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020040420293550.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyMzA4Mjg3,size_16,color_FFFFFF,t_70#pic_center)
通过上述操作可以得到原始图片背景(Tips:这里用的是原始图片,循环操作次数为5是因为循环次数较少则无法将米粒完全腐蚀，得不到期望结果，这些都可以通过试验得到)。
之后我们就可以利用原始图像直接减去该背景得到较为理想的图像
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200404203147630.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyMzA4Mjg3,size_16,color_FFFFFF,t_70#pic_center)![在这里插入图片描述](https://img-blog.csdnimg.cn/20200404203242245.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyMzA4Mjg3,size_16,color_FFFFFF,t_70#pic_center)
将当前图像转化为灰度图像后对其灰度直方图进行分析，可明显看到阈值处于40-70之间，因此很容易选择一个全局阈值对整张图片进行分割处理。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200404203329378.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyMzA4Mjg3,size_16,color_FFFFFF,t_70#pic_center)
因此直接使用简单的二值化处理，就能得到理想的二值化图像

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200404203831178.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyMzA4Mjg3,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200404203435413.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyMzA4Mjg3,size_16,color_FFFFFF,t_70#pic_center)
利用该二值化图像就可以通过opencv提供的 ***findContours*** 函数找到轮廓位置，通过 ***drawContours*** 画出轮廓
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200404203529623.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyMzA4Mjg3,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200404203601413.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyMzA4Mjg3,size_16,color_FFFFFF,t_70#pic_center)
其中 ***findContours*** 的返回值contours是List类型，保存了每个连通域的轮廓信息，直接通过 ***len()*** 函数就可以得知轮廓个数也就是米粒个数。

而对于面积和周长的求取，opencv提供了 ***contourArea*** 和 ***arcLength*** 两种方法，非常容易得到。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200404203709809.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyMzA4Mjg3,size_16,color_FFFFFF,t_70#pic_center)
而之后就只剩下简单的从列表中找出最大值和其所在的位置，后通过 ***drawContours*** 绘制出最大米粒位置。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200404203729555.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyMzA4Mjg3,size_16,color_FFFFFF,t_70#pic_center)![在这里插入图片描述](https://img-blog.csdnimg.cn/20200404203755569.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyMzA4Mjg3,size_16,color_FFFFFF,t_70#pic_center)
但从图中发现最大的米粒是通过两个米粒所共同构成的连通区域，针对这个问题可以利用对所有米粒的面积进行分析，去除奇异值或者其他处理手段解决。

> 图像处理只有多动手实验才能提升自己的理解，空想算法只是空中楼阁，我们每个人都是调参师！

[项目源码github地址](https://github.com/ADExxxxxx/FindRice)
