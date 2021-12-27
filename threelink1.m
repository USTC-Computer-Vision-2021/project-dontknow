%三连杆平面机器人
clear L
q1=sym('q1');%连杆参数化
l1=sym('l1');
q2=sym('q2');
l2=sym('l2');
q3=sym('q3');
l3=sym('l3');
T01=trotz(q1)*transl(l1,0,0);%相邻连杆的变换矩阵
T12=trotz(q2)*transl(l2,0,0);
T23=trotz(q3)*transl(l3,0,0);
T03=T01*T12*T23%运动学方程式即连杆变换矩阵




