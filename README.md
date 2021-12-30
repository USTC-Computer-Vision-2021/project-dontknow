[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-f059dc9a6f8d3a56e377f745f24479a46679e63a5d9fe6f495e02850cd0d8118.svg)](https://classroom.github.com/online_ide?assignment_repo_id=6627215&assignment_repo_type=AssignmentRepo)
# 题目：基于增强现实实现虚拟时钟和动画播放
成员及分工
+ 徐晋 PB18061339：寻找素材 obj建模 数学原理 代码调试 readme写作 
+ 马龙 PB18061359：寻找素材 python代码写作与调试 实验设计 readme写作 
## 问题描述
当你正在考试却发现你所在的教室没有钟表怎么办，这个时候只要打开你的手机就可以获得一个钟表，但是为什么打开手机不看时间和为什么考试途中有手机这种事，应该是医生和监考老师关心的。

你还可以观看飞机和太空站的运行，但是飞机太大了，不建议观看（绝对不是模型太大导致过于卡顿），为了避免观看不佳，所以我们简化了太空站的模型。
## 原理分析
### 视觉标定
### 姿态估计
除了marker的姿态，同时还考虑了摄像头的姿态，利用标定程序获得摄像头的姿态矩阵A。但实际上我们使用的是笔记本摄像头，姿态并不总是固定，同时笔记本摄像头的两个轴又是固定的，只有一个自由度，所以这一部分影响并不大。
### 模型旋转
为了实现钟表指针的旋转，我们使用blender将时针、分针、秒针建模为大小不一的长方体，在使用代码读入模型数据后打印其三维坐标，根据其坐标找到对应的旋转轴。在本实验中旋转轴垂直于基准面，位置位于靠近画面中心的边的中点。所以我们的任务就是把模型的每个顶点绕着一个平行于z轴的直线做旋转。数学原理如下：

绕z轴旋转的旋转矩阵如下：

若绕直线旋转，该直线在xoy平面上的投影坐标为（x,y），则坐标变换公式如下：

z坐标不变。

其中旋转角度由当前读取到的系统时间确定，根据时、分、秒算出对应指针的旋转角度：



## 代码实现
代码逻辑总结如下：

①读取marker（图片文件），模型（.obj），模型贴图（图片文件）

②对marker进行旋转 保持识别时的旋转不变性

③以下循环：

    读取摄像头当前画面
    
    使用aruco算法识别画面中的marker 若识别到则返回对应的变换矩阵
    
    根据变换矩阵把模型贴在当前画面上输出
    
## 效果展示
见result文件夹。

其中result1.文件内容是在marker上显示指针钟表；result2.文件内容是飞机飞行动画，由于使用的模型面数较多，导致计算时间较长，最后生成的动画帧数很低；result2.文件内容是卫星航天器飞行动画，由于使用自制模型较为简单，动画较为流畅。
## 运行说明
需要修改的地方为读取模型、贴图、marker的文件夹路径，也可根据需求修改或更换文件夹里的模型、贴图、marker内容。我们在input文件夹的exture里存放了一些贴图，对于arclock具体代码里读入贴图和obj文件采取了文件读入的方式，你可以将贴图放在一个文件夹下并且修改arclock.py中238行的文件夹路径，clock模型存储在input的clock文件夹里，你应该在arclock.py中的240行修改文件夹路径。 对于plane.py采取了直接的文件读入你可以通过修改模型路径，实现太空站模型s.obj和飞机模型plane.obj的转换。


## 未来展望
实现视野中出现多个marker时同时识别，并在每个marker上以其对应角度展示动画；

不再自行设计轨迹路线和旋转角度，而是找到业界常用的3D动画文件格式的接口和打开方式，想办法把现有的3D动画投影到marker上，来代替目前往每帧图片上贴不同姿态的obj文件的方式。

## 引用参考
### 理论和代码引用
https://github.com/jayantjain100/Augmented-Reality

https://github.com/juangallostra/augmented-reality
### 模型来源
https://www.aigei.com/

部分自建
