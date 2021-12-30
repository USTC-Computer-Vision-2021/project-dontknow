import aruco_module as aruco
import os
import cv2
import datetime
import numpy as np
import math

#摄像头姿态矩阵
A = [[1019.37187, 0, 618.709848], [0, 1024.2138, 327.280578], [0, 0, 1]] 
A = np.array(A)


def get_extended_RT(A, H):
	#将摄像头姿态估计和marker的变换矩阵进行综合 
	
	H = np.float64(H) #数据类型转换
	A = np.float64(A)
	R_12_T = np.linalg.inv(A).dot(H)

	r1 = np.float64(R_12_T[:, 0]) 
	r2 = np.float64(R_12_T[:, 1]) 
	T = R_12_T[:, 2] 
	
	norm = np.float64(math.sqrt(np.float64(np.linalg.norm(r1)) * np.float64(np.linalg.norm(r2))))
	
	r3 = np.cross(r1,r2)/(norm)
	R_T = np.zeros((3, 4))
	R_T[:, 0] = r1
	R_T[:, 1] = r2 
	R_T[:, 2] = r3 
	R_T[:, 3] = T
	return R_T



def augment(img, objs, projection, template, scale=4):
#该函数用来marker的角度将obj模型贴在摄像头捕获的实时图像上

    h, w = template.shape
    vertices = obj.vertices
    img = np.ascontiguousarray(img, dtype=np.uint8)


    a = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], np.float64)
    imgpts = np.int32(cv2.perspectiveTransform(a.reshape(-1, 1, 3), projection))
    cv2.fillConvexPoly(img, imgpts, (0, 0, 0))


     #用当前系统时间决定飞船运行轨迹，speed决定了速率
    t=time.time()
    speed=2
    angle = -(t )* 360 / 60 * math.pi / 180 *speed


    x = 0  # 旋转轴
    y = 0
    z = 5
    # 对于每一个面 遍历贴在图片上
    for face in obj.faces:
        # a face is a list [face_vertices, face_tex_coords, face_col]
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])  # -1 because of the shifted numbering
        points = scale * points
        points = np.array([[p[2] + w / 2, p[0] + h / 2, p[1]] for p in points])  # shifted to centre



        cosA = math.cos(angle)
        # print(cosA)
        sinA = math.sin(angle)
	#根据设计的轨迹旋转模型的面

        for i in range(len(points)):

            points[i][0] = (points[i][0] - x) * cosA - (points[i][1] - y) * sinA + x
            points[i][1] = (points[i][0] - x) * sinA + (points[i][1] - y) * cosA + y


        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)  # transforming to pixel coords
        imgpts = np.int32(dst)
        cv2.fillConvexPoly(img, imgpts, face[-1])#将face贴在摄像头图片上



    return img



class three_d_object:#读取.obj类型文件的类
    def __init__(self, filename_obj, filename_texture, color_fixed=False):
        self.texture = cv2.imread(filename_texture)
        self.vertices = []
        self.faces = []

        self.texcoords = []

        for line in open(filename_obj, "r"):#读取文件头等
            if line.startswith('#'):
                # it's a comment, ignore
                continue

            values = line.split()
            if not values:
                continue

            if values[0] == 'v':
                # vertex description (x, y, z)
                v = [float(a) for a in values[1:4]]
                self.vertices.append(v)

            elif values[0] == 'vt':
                # texture coordinate (u, v)
                self.texcoords.append([float(a) for a in values[1:3]])

            elif values[0] == 'f':
                # face description
                face_vertices = []
                face_texcoords = []
                for v in values[1:]:
                    w = v.split('/')
                    face_vertices.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        face_texcoords.append(int(w[1]))
                    else:
                        color_fixed = True
                        face_texcoords.append(0)
                self.faces.append([face_vertices, face_texcoords])

        for f in self.faces:
            if not color_fixed:#将选择的贴图添加到模型上
                1
                f.append(three_d_object.decide_face_color(f[-1], self.texture, self.texcoords))
            else:
                f.append((50, 50, 50))  # default color

        # cv2.imwrite('texture_marked.png', self.texture)

    def decide_face_color(hex_color, texture, textures):
        # doesnt use proper texture
        # takes the color at the mean of the texture coords

        h, w, _ = texture.shape
        col = np.zeros(3)
        coord = np.zeros(2)
        all_us = []
        all_vs = []

        for i in hex_color:
            t = textures[i - 1]
            coord = np.array([t[0], t[1]])
            u, v = int(w * (t[0]) - 0.0001), int(h * (1 - t[1]) - 0.0001)
            all_us.append(u)
            all_vs.append(v)

        u = int(sum(all_us) / len(all_us))
        v = int(sum(all_vs) / len(all_vs))

        # all_us.append(all_us[0])
        # all_vs.append(all_vs[0])
        # for i in range(len(all_us) - 1):
        #     texture = cv2.line(texture, (all_us[i], all_vs[i]), (all_us[i + 1], all_vs[i + 1]), (0,0,255), 2)
        #     pass

        col = np.uint8(texture[v, u])
        col = [int(a) for a in col]
        col = tuple(col)
        return (col)

if __name__ == '__main__':
    i = 0
    j=0
    obj = three_d_object('data/3d_objects/s.obj','data/3d_objects/111.jpg')


    marker_colored = cv2.imread('data/m1.png')#读取待识别marker文件
    #marker_colored2 = cv2.imread('data/m1.png')
    assert marker_colored is not None, "Could not find the aruco marker image file"

    marker_colored = cv2.flip(marker_colored, 1)
    marker_colored = cv2.resize(marker_colored, (480, 480), interpolation=cv2.INTER_CUBIC)
    marker = cv2.cvtColor(marker_colored, cv2.COLOR_BGR2GRAY)#marker预处理


    print("trying to access the webcam")
    cv2.namedWindow("webcam")
    vc = cv2.VideoCapture(0)#打开电脑摄像头
    assert vc.isOpened(), "couldn't access the webcam"

    h,w = marker.shape
    #considering all 4 rotations 保持识别marker时的旋转不变性
    marker_sig1 = aruco.get_bit_sig(marker, np.array([[0,0],[0,w], [h,w], [h,0]]).reshape(4,1,2))
    marker_sig2 = aruco.get_bit_sig(marker, np.array([[0,w], [h,w], [h,0], [0,0]]).reshape(4,1,2))
    marker_sig3 = aruco.get_bit_sig(marker, np.array([[h,w],[h,0], [0,0], [0,w]]).reshape(4,1,2))
    marker_sig4 = aruco.get_bit_sig(marker, np.array([[h,0],[0,0], [0,w], [h,w]]).reshape(4,1,2))

    sigs = [marker_sig1, marker_sig2, marker_sig3, marker_sig4]


    rval, frame = vc.read()#读取当前摄像头的一帧画面
    assert rval, "couldn't access the webcam"
    h2, w2,  _= frame.shape

    h_canvas = max(h, h2)
    w_canvas = w + w2

    #k = 0

    while rval:
        rval, frame = vc.read() #读取摄像头的一帧画面
        key = cv2.waitKey(20)
        if key == 27: 
            break

        canvas = np.zeros((h_canvas, w_canvas, 3), np.uint8) #最终展示的图片
        canvas[:h, :w, :] = marker_colored #把marker放在左边

        success1, H = aruco.find_homography_aruco(frame, marker, sigs)#匹配算法
        #success2, H2 = aruco.find_homography_aruco(frame, marker2, sigs2)
        # success = False

        if not success1:#如果没有匹配成功 还是显示当前摄像头画面
            # print('homograpy est failed')
            canvas[:h2 , w: , :] = np.flip(frame, axis = 1)
            cv2.imshow("webcam", canvas )

            continue


        R_T = get_extended_RT(A, H)
        transformation = A.dot(R_T)#计算marker与实际识别到的marker之间的变换矩阵




        augmented2 = np.flip(augment(frame, objs, transformation, marker), axis=1)#根据变换矩阵把模型贴在当前画面
        #k=k+1
        canvas[:h2 , w: , :] = augmented2
        cv2.imshow("webcam", canvas)#展示当前画面






        R_T = get_extended_RT(A, H)
        transformation = A.dot(R_T)




        augmented2 = np.flip(augment(frame, obj, transformation, marker), axis=1)
        k=k+1
        canvas[:h2 , w: , :] = augmented2
        cv2.imshow("webcam", canvas)


