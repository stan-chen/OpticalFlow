#pragma once

#include <opencv2/opencv.hpp>
#include <boost/thread.hpp>


//基本类型定义
//PI定义
constexpr double  PI = 3.14159265358979323846;


/**
* \brief 打开视频文件并返回帧数
* \param capture 视频句柄
* \param filename 文件名
* \return 返回帧数
*/
inline int open_video(cv::VideoCapture & capture, const std::string &filename)
{
    //打开视频文件
    capture.open(filename);
    auto  frame_count = 0; //总帧长

    assert(capture.isOpened());//检查是否打开

                               //获取总帧数
    frame_count = static_cast<int>(capture.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_COUNT));

    //进度条，可以查看处理了多少帧
    return frame_count;
}

/**
* \brief 创建窗口线程函数，将窗口独立到线程中
*/
inline void    output_windows_thread(const std::string &windowname)
{
    cv::namedWindow(windowname);
    cv::waitKeyEx();
}


inline  void    create_thread_window(const std::string &windowname)
{
    boost::thread(output_windows_thread, windowname);

    
}




