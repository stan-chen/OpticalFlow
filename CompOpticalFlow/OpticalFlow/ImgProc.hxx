#pragma once

//图像二值化阈值
constexpr double  IMAGE_THRESHOLD = 127.00f;

/**
* \brief 定义平方函数
* \tparam _Ty 平方类型
* \param a 平方因子
* \return
*/
template<typename _Ty>
constexpr double square(_Ty a)
{
    return a * a;
}

/**
* \brief 两点之间线段长度
* \param p1 点向量1
* \param p2 点向量2
* \return 计算出的线段长度
*/
inline double  segment_length(const cv::Point & p1, const cv::Point &p2)
{
    return std::sqrt(square(p1.x - p2.x) + square(p1.y - p2.y));
}

/**
* \brief 定义图像处理函数
* \param name 处理方法名
*/
#define     DEF_IMG_HANDLER(name)   \
    inline void  name(cv::Mat &in,cv::Mat &out)

/**
* \brief 定义处理操作符
* \tparam Functor 处理方法
* \param img 处理图像
* \param f 方法
* \return 处理后的图像
*/
template <typename Functor>
cv::Mat  &operator << (cv::Mat &img, Functor f)
{
    f(img, img);
    return img;
}

/**
* \brief 灰度化处理方法
* \param in 原图像三通道
* \param out 输出图像，灰度图像
*/
DEF_IMG_HANDLER(cvtgray)
{
    assert(in.channels() == 3);
    cv::cvtColor(in, out, cv::COLOR_BGR2GRAY);
}

/**
* \brief 灰度图像转彩色图像
* \param in 灰度图像，单通道
* \param out 彩色图像，三通道
*/
DEF_IMG_HANDLER(cvtbgr)
{
    assert(in.channels() == 1);
    cv::cvtColor(in, out, cv::COLOR_GRAY2BGR);
}

/**
* \brief 图像转HSV格式
* \param in 源图像
* \param out hsv格式图像
*/
DEF_IMG_HANDLER(cvthsv)
{
    cv::cvtColor(in, out, cv::COLOR_BGR2HSV);
}

/**
* \brief 二值化图像
* \param in 传入图像
* \param out 二值化图像，单通道
*/
DEF_IMG_HANDLER(binary)
{
    cv::threshold(in, out, IMAGE_THRESHOLD, 255.0f, CV_THRESH_BINARY);

//    cv::adaptiveThreshold(in, out, 255.0f,
//        cv::AdaptiveThresholdTypes::ADAPTIVE_THRESH_GAUSSIAN_C,
//        cv::ThresholdTypes::THRESH_BINARY, 7, 0.0);
}

/**
* \brief 白平衡处理，均衡化三通道
* \param in 输入图像三通道
* \param out 输出图像三通道
*/
DEF_IMG_HANDLER(whiteblace)
{
    assert(in.channels() == 3);
    using namespace cv;
    Mat &g_srcImage = in, &dstImage = out;
    std::vector<Mat> g_vChannels;

    split(g_srcImage, g_vChannels);
    Mat imageBlueChannel = g_vChannels.at(0);
    Mat imageGreenChannel = g_vChannels.at(1);
    Mat imageRedChannel = g_vChannels.at(2);

    double imageBlueChannelAvg = 0;
    double imageGreenChannelAvg = 0;
    double imageRedChannelAvg = 0;

    //求各通道的平均值
    imageBlueChannelAvg = mean(imageBlueChannel)[0];
    imageGreenChannelAvg = mean(imageGreenChannel)[0];
    imageRedChannelAvg = mean(imageRedChannel)[0];

    //求出个通道所占增益
    double K = (imageRedChannelAvg + imageGreenChannelAvg + imageRedChannelAvg) / 3;
    double Kb = K / imageBlueChannelAvg;
    double Kg = K / imageGreenChannelAvg;
    double Kr = K / imageRedChannelAvg;

    //更新白平衡后的各通道BGR值
    addWeighted(imageBlueChannel, Kb, 0, 0, 0, imageBlueChannel);
    addWeighted(imageGreenChannel, Kg, 0, 0, 0, imageGreenChannel);
    addWeighted(imageRedChannel, Kr, 0, 0, 0, imageRedChannel);

    merge(g_vChannels, dstImage);//图像各通道合并 
}

/**
* \brief 灰度直方图均衡化
* \param in 输入灰度图
* \param out 输出灰度图
*/
DEF_IMG_HANDLER(equalizeHist)
{
    assert(in.channels() == 1);
    cv::equalizeHist(in, out);
}

/**
* \brief 图像锐化，拉普拉斯算子
* \param in 输入图像
* \param out 输出图像
*/
DEF_IMG_HANDLER(sharpen)
{
    cv::Mat kernela(3, 3, CV_32F, cv::Scalar(0));
    kernela.at<float>(1, 1) = 5.0;
    kernela.at<float>(0, 1) = -1.0;
    kernela.at<float>(2, 1) = -1.0;
    kernela.at<float>(1, 0) = -1.0;
    kernela.at<float>(1, 2) = -1.0;
    filter2D(in, out, in.depth(), kernela);
}

