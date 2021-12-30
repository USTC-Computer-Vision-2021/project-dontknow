
import aruco_module as aruco
from my_constants import *
from utils import get_extended_RT
import os

import cv2
import datetime
import numpy as np
import math





def augment(img, objs, projection, template, scale=4):


    hour = objs[1]
    min = objs[2]
    sec = objs[3]
    clock = objs[0]

    h, w = template.shape
    hvertices = hour.vertices
    mvertices = min.vertices
    svertices = sec.vertices
    cvertices = clock.vertices
    img = np.ascontiguousarray(img, dtype=np.uint8)

    # blacking out the aruco marker
    a = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], np.float64)
    imgpts = np.int32(cv2.perspectiveTransform(a.reshape(-1, 1, 3), projection))
    cv2.fillConvexPoly(img, imgpts, (0, 0, 0))

    now = datetime.datetime.now()
    print(now.hour % 12)
    secangle = now.second * 360 / 60 * math.pi / 180
    minangle = now.minute * 360 / 60 * math.pi / 180 + secangle / 60
    hangle = (now.hour % 12) * 360 / 12 * math.pi / 180 + minangle / 12
    #各个指针的角速度计算

    x = 239  # 旋转轴
    y = 240
    z = 5

    for face in clock.faces:

        face_vertices = face[0]
        points = np.array([cvertices[vertex - 1] for vertex in face_vertices])  # -1 because of the shifted numbering
        points = scale * points
        points = np.array([[p[2] + w / 2, p[0] + h / 2, p[1]] for p in points])  # shifted to centre

        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)  # transforming to pixel coords
        imgpts = np.int32(dst)
        cv2.fillConvexPoly(img, imgpts, face[-1])


    for face in hour.faces:

        face_vertices = face[0]
        points = np.array([hvertices[vertex - 1] for vertex in face_vertices])  # -1 because of the shifted numbering
        points = scale * points
        points = np.array([[p[2] + w / 2, p[0] + h / 2, p[1]] for p in points])  # shifted to centre


        cosA = math.cos(hangle)
        # print(cosA)
        sinA = math.sin(hangle)

        for i in range(4):
            xx = (points[i][0] - x) * cosA - (points[i][1] - y) * sinA + x
            yy = (points[i][0] - x) * sinA + (points[i][1] - y) * cosA + y
            points[i][0] = xx
            points[i][1] = yy



        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)  # transforming to pixel coords
        imgpts = np.int32(dst)
        cv2.fillConvexPoly(img, imgpts, face[-1])

    for face in min.faces:
        # a face is a list [face_vertices, face_tex_coords, face_col]
        face_vertices = face[0]
        points = np.array([mvertices[vertex - 1] for vertex in face_vertices])  # -1 because of the shifted numbering
        points = scale * points
        points = np.array([[p[2] + w / 2, p[0] + h / 2, p[1]] for p in points])  # shifted to centre

        cosA = math.cos(minangle)

        sinA = math.sin(minangle)

        for i in range(4):
            xx = (points[i][0] - x) * cosA - (points[i][1] - y) * sinA + x
            yy = (points[i][0] - x) * sinA + (points[i][1] - y) * cosA + y
            points[i][0] = xx
            points[i][1] = yy

        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)  # transforming to pixel coords
        imgpts = np.int32(dst)
        cv2.fillConvexPoly(img, imgpts, face[-1])

    for face in sec.faces:
        # a face is a list [face_vertices, face_tex_coords, face_col]
        face_vertices = face[0]
        points = np.array([svertices[vertex - 1] for vertex in face_vertices])  # -1 because of the shifted numbering
        points = scale * points
        points = np.array([[p[2] + w / 2, p[0] + h / 2, p[1]] for p in points])  # shifted to centre

        cosA = math.cos(secangle)
        # print(cosA)
        sinA = math.sin(secangle)

        for i in range(4):
            xx = (points[i][0] - x) * cosA - (points[i][1] - y) * sinA + x
            yy = (points[i][0] - x) * sinA + (points[i][1] - y) * cosA + y
            points[i][0] = xx
            points[i][1] = yy

        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)  # transforming to pixel coords
        imgpts = np.int32(dst)
        cv2.fillConvexPoly(img, imgpts, face[-1])

    return img


class three_d_object:
    def __init__(self, filename_obj, filename_texture, color_fixed=False):
        self.texture = cv2.imread(filename_texture)
        self.vertices = []
        self.faces = []

        self.texcoords = []

        for line in open(filename_obj, "r"):
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
            if not color_fixed:
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
    paths = {}
    picture={}
    objs={}
    path ='data/3d_objects/clock'
    flie_dir=os.listdir(path)
    picturepath='data/3d_objects/tex'
    picture_dir=os.listdir(picturepath)
    for file in picture_dir:
        picture[j]=os.path.join(picturepath,file)
        j=j+1;
    for file in flie_dir:
       paths[i]=os.path.join(path,file)

       print(paths[i])
       objs[i] = three_d_object(paths[i], picture[i])
       i = i + 1


    marker_colored = cv2.imread('data/m1.png')
    marker_colored2 = cv2.imread('data/m1.png')
    assert marker_colored is not None, "Could not find the aruco marker image file"

    marker_colored = cv2.flip(marker_colored, 1)
    marker_colored = cv2.resize(marker_colored, (480, 480), interpolation=cv2.INTER_CUBIC)
    marker = cv2.cvtColor(marker_colored, cv2.COLOR_BGR2GRAY)


    print("trying to access the webcam")
    cv2.namedWindow("webcam")
    vc = cv2.VideoCapture(0)
    assert vc.isOpened(), "couldn't access the webcam"

    h,w = marker.shape
    #considering all 4 rotations
    marker_sig1 = aruco.get_bit_sig(marker, np.array([[0,0],[0,w], [h,w], [h,0]]).reshape(4,1,2))
    marker_sig2 = aruco.get_bit_sig(marker, np.array([[0,w], [h,w], [h,0], [0,0]]).reshape(4,1,2))
    marker_sig3 = aruco.get_bit_sig(marker, np.array([[h,w],[h,0], [0,0], [0,w]]).reshape(4,1,2))
    marker_sig4 = aruco.get_bit_sig(marker, np.array([[h,0],[0,0], [0,w], [h,w]]).reshape(4,1,2))

    sigs = [marker_sig1, marker_sig2, marker_sig3, marker_sig4]


    rval, frame = vc.read()
    assert rval, "couldn't access the webcam"
    h2, w2,  _= frame.shape

    h_canvas = max(h, h2)
    w_canvas = w + w2

    k = 0

    while rval:
        rval, frame = vc.read() #fetch frame from webcam
        key = cv2.waitKey(20)
        if key == 27: # Escape key to exit the program
            break

        canvas = np.zeros((h_canvas, w_canvas, 3), np.uint8) #final display
        canvas[:h, :w, :] = marker_colored #marker for reference

        success1, H = aruco.find_homography_aruco(frame, marker, sigs)
        #success2, H2 = aruco.find_homography_aruco(frame, marker2, sigs2)
        # success = False

        if not success1:
            # print('homograpy est failed')
            canvas[:h2 , w: , :] = np.flip(frame, axis = 1)
            cv2.imshow("webcam", canvas )

            continue


        R_T = get_extended_RT(A, H)
        transformation = A.dot(R_T)




        augmented2 = np.flip(augment(frame, objs, transformation, marker), axis=1)
        k=k+1
        canvas[:h2 , w: , :] = augmented2
        cv2.imshow("webcam", canvas)


