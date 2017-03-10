# -*- coding:utf-8 -*-
'''
    Author : Xuefei Chen
    Email : chenxuefei_pp@163.com
    Created on : 2017/3/6 9:17
'''

from threading import Thread
import numpy as np
import time

from plot.Plot4Lk import plot_3d

try:
    import cv2.cv2 as cv2
except Exception:
    import cv2
import math

FEATURE_MAX_NUM = 600
PI = 3.141592653

window_threads = []

def output_windows_thread(windowname):
    cv2.namedWindow(windowname)
    cv2.waitKeyEx()
    pass

def create_thread_window(wintitle):
    t = Thread(target=output_windows_thread,args=(wintitle,))
    t.setDaemon(True)
    t.start()
    window_threads.append(t)
    pass

def segment_length(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def draw_path_on_baseimage(m1, frame1_features, frame2_features, found_futures, found_err):
    line_color = (0,0,255)
    for pos in range(0,FEATURE_MAX_NUM):
        if (found_futures[pos]== 0 or found_err[pos] > 1.500):
            continue
        p = frame1_features[pos][0]
        q = frame2_features[pos][0]

        px,py ,qx,qy = p[0],p[1],q[0],q[1]

        angle = math.atan2(py - qy , px - qx)
        hypotenuse = math.sqrt(np.square(py - qy) + np.square(px - qx))
        qx -= 5 * hypotenuse * math.cos(angle)
        qy -= 5 * hypotenuse * math.sin(angle)

        cv2.line(m1, (int(px),int(py)), (int(qx),int(qy)), line_color, 1, cv2.LINE_AA, 0)
        px = qx + 9 * math.cos(angle + PI / 4)
        py = qy + 9 * math.sin(angle + PI / 4)
        cv2.line(m1,  (int(px),int(py)), (int(qx),int(qy)),  line_color, 1, cv2.LINE_AA, 0)
        px = qx + 9 * math.cos(angle - PI / 4)
        py = qy + 9 * math.sin(angle - PI / 4)
        cv2.line(m1,  (int(px),int(py)), (int(qx),int(qy)),  line_color, 1, cv2.LINE_AA, 0)
    pass


def draw_features_in_base_image(base_out, frame1_features, frame2_features, found_futures, found_err):
    area_color = (0, 0, 255)
    for pos in range(0,FEATURE_MAX_NUM):
        if (
            found_futures[pos] == 0
            or
            found_err[pos] > 1.500
            ):
            continue
        start_p = frame1_features[pos][0]
        end_p = frame2_features[pos][0]
        if(segment_length(start_p,end_p) > 2.0):
            continue
        cv2.line(base_out, (int(start_p[0]),int(start_p[1])), (int(end_p[0]),int(end_p[1])), area_color, 1,cv2.LINE_AA)
    pass

def calc_features_by_lk(base_out, input_video_filename, base_output_filename):
    capture = cv2.VideoCapture(input_video_filename)
    while(True):
        ret1,m1 = capture.read()
        ret2,m2 = capture.read()
        if(not ret1 or not ret2):
            break

        frame1 = m1.copy()
        frame2 = m2.copy()

        frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        sc = frame1.mean()

        if (sc < 0.1):
            continue

        frame1_features = cv2.goodFeaturesToTrack(frame1,FEATURE_MAX_NUM,0.01,0.01)

        frame2_features,found_futures,found_err = cv2.calcOpticalFlowPyrLK(
            frame1,frame2,
            frame1_features,None,winSize=(5, 5),maxLevel=5,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        draw_path_on_baseimage(m1, frame1_features, frame2_features, found_futures, found_err)
        draw_features_in_base_image(base_out, frame1_features, frame2_features, found_futures, found_err)

        cv2.imshow("output_window", m1)
        cv2.imshow("superposition_window", base_out)

    capture.release()
    pass

def ctfLKOpticalFlow():
    input_video_filename = 'videos/0.avi'
    base_image_filename = ""
    base_output_filename = "VideoOut.avi"
    base_outimg_filename = 'images/5.png'

    create_thread_window('output_window')
    create_thread_window('superposition_window')
    #create_thread_window('freq_window')


    if (base_image_filename == ''):
        capture = cv2.VideoCapture(input_video_filename)
        ret,base_image = capture.read()
        capture.release()
    else:
        base_image = cv2.imread(base_image_filename)

    if(base_image.size == 0):
        return

    base_out = np.zeros(base_image.shape, base_image.dtype)
    calc_features_by_lk(base_out, input_video_filename, base_output_filename)
    cv2.imwrite(base_outimg_filename,base_out)

def process_direct():
    create_thread_window('main_window')
    time.sleep(1)
    src = cv2.imread('../images/5.png')
    second_src = src.copy()
    base_out = src.copy()
    gray = cv2.cvtColor(base_out,cv2.COLOR_BGR2GRAY)
    base_out = cv2.adaptiveThreshold(gray,255.0,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY,11,0.0)
    #cv2.imshow('main_window', base_out)
    #erode_kernel = cv2.getStructuringElement(cv2.MORPH_ERODE,(3,3))
    #base_out = cv2.erode(base_out,erode_kernel)

    contours = cv2.findContours(base_out,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_TC89_KCOS)
    #hulls = []
    for idx in range(0,len(contours[1])):
        area = cv2.contourArea(contours[1][idx])
        if area > 50.0:
            cv2.drawContours(src,contours[1],idx,(0,255,0),cv2.FILLED,cv2.LINE_AA)
        pass
    #cv2.drawContours(src, hulls, -1, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)

    throd = cv2.threshold(cv2.cvtColor(src,cv2.COLOR_BGR2GRAY),120,255.0,cv2.THRESH_BINARY)[1]

    #cv2.imwrite('images/throd.png',throd[1])
    cv2.imshow('main_window',throd)
    for t in window_threads:
        t.join()

if  __name__ == '__main__':
    process_direct()
