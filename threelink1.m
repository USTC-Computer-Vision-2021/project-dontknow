%������ƽ�������
clear L
q1=sym('q1');%���˲�����
l1=sym('l1');
q2=sym('q2');
l2=sym('l2');
q3=sym('q3');
l3=sym('l3');
T01=trotz(q1)*transl(l1,0,0);%�������˵ı任����
T12=trotz(q2)*transl(l2,0,0);
T23=trotz(q3)*transl(l3,0,0);
T03=T01*T12*T23%�˶�ѧ����ʽ�����˱任����




