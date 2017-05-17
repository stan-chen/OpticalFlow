# -*- coding:utf-8 -*-
'''
    Author : Xuefei Chen
    Email : chenxuefei_pp@163.com
    Created on : 2017/3/6 9:17
'''

from threading import Thread
import numpy as np
import math

try:
    import cv2.cv2 as cv2
except Exception:
    import cv2

# TODO 判断黑暗参数，如果小于这个值，说明太暗排除图片，可调 ！
MEAN_THRESHOLD = 1.6

FEATURE_MAX_NUM = 600
PI = 3.141592653
# 窗口线程列表
window_threads = []


def output_windows_thread(windowname):
    cv2.namedWindow(windowname)
    cv2.waitKeyEx()
    pass


def create_thread_window(wintitle):
    t = Thread(target=output_windows_thread, args=(wintitle,))
    t.setDaemon(True)
    t.start()
    window_threads.append(t)
    pass


def segment_length(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def draw_path_on_baseimage(m1, frame1_features, frame2_features, found_futures, found_err):
    line_color = (0, 0, 255)
    for pos in range(0, len(found_futures)):
        if (found_futures[pos] == 0 or found_err[pos] > 1.500):
            continue
        p = frame1_features[pos][0]
        q = frame2_features[pos][0]

        px, py, qx, qy = p[0], p[1], q[0], q[1]

        angle = math.atan2(py - qy, px - qx)
        hypotenuse = math.sqrt(np.square(py - qy) + np.square(px - qx))
        qx -= 5 * hypotenuse * math.cos(angle)
        qy -= 5 * hypotenuse * math.sin(angle)

        cv2.line(m1, (int(px), int(py)), (int(qx), int(qy)), line_color, 1, cv2.LINE_AA, 0)
        px = qx + 9 * math.cos(angle + PI / 4)
        py = qy + 9 * math.sin(angle + PI / 4)
        cv2.line(m1, (int(px), int(py)), (int(qx), int(qy)), line_color, 1, cv2.LINE_AA, 0)
        px = qx + 9 * math.cos(angle - PI / 4)
        py = qy + 9 * math.sin(angle - PI / 4)
        cv2.line(m1, (int(px), int(py)), (int(qx), int(qy)), line_color, 1, cv2.LINE_AA, 0)
    pass

def draw_features_in_base_image(base_out, frame1_features, frame2_features, found_futures, found_err):
    area_color = (0, 0, 255)
    for pos in range(0, len(found_futures)):
        if (
                        found_futures[pos] == 0
                or
                        found_err[pos] > 1.500
        ):
            continue
        start_p = frame1_features[pos][0]
        end_p = frame2_features[pos][0]
        if (segment_length(start_p, end_p) > 2.0):
            continue
        cv2.line(base_out, (int(start_p[0]), int(start_p[1])), (int(end_p[0]), int(end_p[1])), area_color, 1,
                 cv2.LINE_AA)
    pass


def calc_features_by_lk(base_out, input_video_filename, base_output_filename):
    capture = cv2.VideoCapture(input_video_filename)
    while (True):
        ret1, m1 = capture.read()
        ret2, m2 = capture.read()
        if (not ret1 or not ret2):
            break

        frame1 = m1.copy()
        frame2 = m2.copy()

        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        sc = frame1.mean()

        if (sc < 1.6):
            continue

        frame1_features = cv2.goodFeaturesToTrack(frame1, FEATURE_MAX_NUM, 0.01, 0.01)

        frame2_features, found_futures, found_err = cv2.calcOpticalFlowPyrLK(
            frame1, frame2,
            frame1_features, None, winSize=(5, 5), maxLevel=5,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        draw_path_on_baseimage(m1, frame1_features, frame2_features, found_futures, found_err)
        draw_features_in_base_image(base_out, frame1_features, frame2_features, found_futures, found_err)

        cv2.imshow("output_window", m1)
        cv2.imshow("superposition_window", base_out)

    capture.release()
    pass


def find_second_contours(base_out, output):
    contours = cv2.findContours(base_out, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    contours_map = {}
    for idx in range(0, len(contours[1])):
        area = cv2.contourArea(contours[1][idx])
        contours_map[idx] = {'area': area, 'contour': contours[1][idx]}
        pass

    contours_map = sorted(contours_map.items(), key=lambda d: d[1]['area'])
    for idx in range(1, 3):
        id = - idx
        max_contours = contours_map[id][1]['contour']
        hulls = cv2.convexHull(max_contours, returnPoints=False)
        pts_index = cv2.convexityDefects(contour=max_contours, convexhull=hulls)
        pts = []
        for v in pts_index:
            pts.append(max_contours[v[0][0]])
            pts.append(max_contours[v[0][1]])

        ndpts = np.zeros((len(pts), 1, 2), np.int32)
        for idx in range(0, len(pts)):
            ndpts[idx] = pts[idx]

        cv2.drawContours(output, ndpts, -1, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)

    return output


# 金字塔算法主程序
def ctfLKOpticalFlow(*args, **kwargs):
    """
    :param args:
    :param kwargs:
        src_video : 基础待检测的视频文件名
        base_image : 基础待检测视频的背景图片，如为空则为视频第一帧
        output_video : 检测过程输出视频文件名
        output_image : 检测结果输出图片文件名
    """
    input_video_filename = kwargs.get('src_video', 'videos/0.avi')
    base_image_filename = kwargs.get('base_image', None)
    output_video_filename = kwargs.get('output_video', None)
    output_image_filename = kwargs.get('output_image', 'images/0.png')

    create_thread_window('output_window')
    create_thread_window('superposition_window')

    try:
        if (base_image_filename is None):
            capture = cv2.VideoCapture(input_video_filename)
            ret, base_image = capture.read()
            capture.release()
        else:
            base_image = cv2.imread(base_image_filename)
    except Exception as e:
        print(e)
        exit(-1)

    # 创建空白图层
    empty_image = np.zeros(base_image.shape, base_image.dtype)
    # 从原始图层复制结果图层
    result_image = base_image.copy()
    # 开始主扫描算法
    calc_features_by_lk(empty_image, input_video_filename, output_video_filename)

    # 复制扫描结果图层
    scan_result_image = empty_image.copy()

    # 自适应二值化扫描结果图层方便寻找边缘
    empty_image = cv2.adaptiveThreshold(cv2.cvtColor(empty_image, cv2.COLOR_BGR2GRAY), 255.0,
                                        cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 11, 0.0)
    # 寻找边缘
    contours = cv2.findContours(empty_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

    # 绘出边缘
    for idx in range(0, len(contours[1])):
        area = cv2.contourArea(contours[1][idx])
        if area > 50.0:
            # 绘制到扫描结果图层
            cv2.drawContours(scan_result_image, contours[1], idx, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)
            cv2.drawContours(result_image, contours[1], idx, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)
        pass

    # 阈值化扫描结果
    throd_image = cv2.threshold(cv2.cvtColor(scan_result_image, cv2.COLOR_BGR2GRAY), 120, 255.0, cv2.THRESH_BINARY)[1]

    cv2.imshow('superposition_window', throd_image)

    cv2.imshow('output_window', result_image)

    cv2.imwrite(output_image_filename, result_image)

    for t in window_threads:
        t.join()


if __name__ == '__main__':
    ctfLKOpticalFlow()
