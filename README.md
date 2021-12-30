[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-f059dc9a6f8d3a56e377f745f24479a46679e63a5d9fe6f495e02850cd0d8118.svg)](https://classroom.github.com/online_ide?assignment_repo_id=6627215&assignment_repo_type=AssignmentRepo)
# 题目
成员及分工
+ 徐晋 PB18061339：寻找素材 obj建模 数学原理 代码调试 readme写作 
+ 马龙 PB18061359：寻找素材 python代码写作与调试 实验设计 readme写作 
## 问题描述

## 原理分析

为了实现钟表指针的旋转，我们使用blender将时针、分针、秒针建模为大小不一的长方体，在使用代码读入模型数据后打印其三维坐标，根据其坐标找到对应的旋转轴。在本实验中旋转轴垂直于基准面，位置位于靠近画面中心的边的中点。所以我们的任务就是把模型的每个顶点绕着一个平行于z轴的直线做旋转。数学原理如下：

绕z轴旋转的旋转矩阵如下：

若绕直线旋转，该直线在xoy平面上的投影坐标为（x,y），则坐标变换公式如下：

z坐标不变。

其中旋转角度由当前读取到的系统时间确定，根据时、分、秒算出对应指针的旋转角度：



## 代码实现

## 效果展示
见result文件夹

其中result1.文件内容是在marker上显示指针钟表；result2.文件内容是飞机飞行动画，由于使用的模型面数较多，导致计算时间较长，最后生成的动画帧数很低；result2.文件内容是卫星航天器飞行动画，由于使用自制模型较为简单，动画较为流畅。
## 运行说明
需要修改的地方为读取模型、贴图、marker的文件夹路径，也可根据需求修改或更换文件夹里的模型、贴图、marker内容。

## 未来展望
实现视野中出现多个marker时同时识别，并在每个marker上以不同角度展示动画；

找到业界常用的3D动画文件格式的接口和打开方式，把它投影到marker上。

## 引用参考
### 理论和代码引用
https://github.com/jayantjain100/Augmented-Reality

https://github.com/juangallostra/augmented-reality
### 模型来源
https://www.aigei.com/

部分自建
