#include "stdafx.h"
#include <stdio.h>  
#include <windows.h>
#include <opencv2\opencv.hpp>  
using namespace cv;

static const double pi = 3.14159265358979323846;
inline static double square(int a)
{
    return a * a;
}


/*该函数目的：给img分配内存空间，并设定format，如位深以及channel数*/
inline static void allocateOnDemand(IplImage **img, CvSize size, int depth, int channels)
{
    if (*img != NULL) return;
    *img = cvCreateImage(size, depth, channels);
    if (*img == NULL)
    {
        fprintf(stderr, "Error: Couldn't allocate image.  Out of memory?\n");
        exit(-1);
    }
}
/*主函数，原程序是读取avi视频文件，然后处理，我简单改成从摄像头直接读取数据*/
int main(int argc, char *argv[])
{

    //读取摄像头
    VideoCapture cap{};

    cap.open("0.1.avi");

    //读取视频文件

    //VideoCapture cap; cap.open("optical_flow_input.avi");
    if (!cap.isOpened())
    {
        return -1;
    }
    Mat frame;

    /*
    bool stop = false;
    while (!stop)
    {
    cap >> frame;
    //	cvtColor(frame, edges, CV_RGB2GRAY);
    //	GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
    //	Canny(edges, edges, 0, 30, 3);
    //	imshow("当前视频", edges);
    imshow("当前视频", frame);
    if (waitKey(30) >= 0)
    stop = true;
    }
    */

    //CvCapture *input_video = cvCaptureFromFile(	"optical_flow_input.avi"	);
    //cv::VideoCapture cap = *(cv::VideoCapture *) userdata;


    //if (input_video == NULL)
    //	{
    //	fprintf(stderr, "Error: Can't open video device.\n");
    //	return -1;
    //	}

    /*先读取一帧，以便得到帧的属性，如长、宽等*/
    //cvQueryFrame(input_video);

    /*读取帧的属性*/
    CvSize frame_size;
    frame_size.height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    frame_size.width = cap.get(CV_CAP_PROP_FRAME_WIDTH);

    /*********************************************************/

    /*用于把结果写到文件中去,非必要
    int frameW = frame_size.height; // 744 for firewire cameras
    int frameH = frame_size.width; // 480 for firewire cameras
    VideoWriter writer("VideoTest.avi", -1, 25.0, cvSize(frameW, frameH), true);

    /*开始光流法*/
    //VideoWriter writer("VideoTest.avi", CV_FOURCC('D', 'I', 'V', 'X'), 25.0, Size(640, 480), true);

    while (true)
    {
        static IplImage *frame = NULL, *frame1 = NULL, *frame1_1C = NULL,
            *frame2_1C = NULL, *eig_image = NULL, *temp_image = NULL,
            *pyramid1 = NULL, *pyramid2 = NULL;

        Mat framet;
        /*获取第一帧*/
        //	cap >> framet;
        cap.read(framet);
        Mat edges;
        //黑白抽象滤镜模式
        // cvtColor(framet, edges, CV_RGB2GRAY);
        //	GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
        //	Canny(edges, edges, 0, 30, 3);

        //转换mat格式到lpiimage格式
        frame = &IplImage(framet);
        if (frame == NULL)
        {
            fprintf(stderr, "Error: Hmm. The end came sooner than we thought.\n");
            return -1;
        }

        /*由于opencv的光流函数处理的是8位的灰度图，所以需要创建一个同样格式的
        IplImage的对象*/
        allocateOnDemand(&frame1_1C, frame_size, IPL_DEPTH_8U, 1);

        /* 把摄像头图像格式转换成OpenCV惯常处理的图像格式*/
        cvConvertImage(frame, frame1_1C, 0);

        /* 我们需要把具有全部颜色信息的原帧保存，以备最后在屏幕上显示用*/
        allocateOnDemand(&frame1, frame_size, IPL_DEPTH_8U, 3);
        cvConvertImage(frame, frame1, 0);

        /* 获取第二帧 */
        //cap >> framet;
        cap.read(framet);
        //	cvtColor(framet, edges, CV_RGB2GRAY);
        //	GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
        //	Canny(edges, edges, 0, 30, 3);
        frame = &IplImage(framet);
        if (frame == NULL)
        {
            fprintf(stderr, "Error: Hmm. The end came sooner than we thought.\n");
            return -1;
        }

        /*原理同上*/
        allocateOnDemand(&frame2_1C, frame_size, IPL_DEPTH_8U, 1);
        cvConvertImage(frame, frame2_1C, 0);

        /*********************************************************
        开始shi-Tomasi算法，该算法主要用于feature selection,即一张图中哪些是我
        们感兴趣需要跟踪的点(interest point)
        input:
        * "frame1_1C" 输入图像.
        * "eig_image" and "temp_image" 只是给该算法提供可操作的内存区域.
        * 第一个".01" 规定了特征值的最小质量，因为该算法要得到好的特征点，哪就
        需要一个选择的阈值
        * 第二个".01" 规定了像素之间最小的距离，用于减少运算复杂度，当然也一定
        程度降低了跟踪精度
        * "NULL" 意味着处理整张图片，当然你也可以指定一块区域
        output:
        * "frame1_features" 将会包含fram1的特征值
        * "number_of_features" 将在该函数中自动填充上所找到特征值的真实数目,
        该值<= 400
        **********************************************************/

        /*开始准备该算法需要的输入*/

        /* 给eig_image,temp_image分配空间*/
        allocateOnDemand(&eig_image, frame_size, IPL_DEPTH_32F, 1);
        allocateOnDemand(&temp_image, frame_size, IPL_DEPTH_32F, 1);

        /* 定义存放frame1特征值的数组，400只是定义一个上限 */
        CvPoint2D32f frame1_features[400];
        int    number_of_features = 400;

        /*开始跑shi-tomasi函数*/
        cvGoodFeaturesToTrack(frame1_1C, eig_image, temp_image,
            frame1_features, &number_of_features, .01, .01, NULL);

        /**********************************************************
        开始金字塔Lucas Kanade光流法，该算法主要用于feature tracking,即是算出
        光流，并跟踪目标。
        input:
        * "frame1_1C" 输入图像，即8位灰色的第一帧
        * "frame2_1C" 第二帧，我们要在其上找出第一帧我们发现的特征点在第二帧
        的什么位置
        * "pyramid1" and "pyramid2" 是提供给该算法可操作的内存区域，计算中间
        数据
        * "frame1_features" 由shi-tomasi算法得到的第一帧的特征点.
        * "number_of_features" 第一帧特征点的数目
        * "optical_flow_termination_criteria" 该算法中迭代终止的判别，这里是
        epsilon<0.3，epsilon是两帧中对应特征窗口的光度之差的平方，这个以后的文
        章会讲
        * "0" 这个我不知道啥意思，反正改成1就出不来光流了，就用作者原话解释把
        means disable enhancements.  (For example, the second array isn't
        pre-initialized with guesses.)
        output:
        * "frame2_features" 根据第一帧的特征点，在第二帧上所找到的对应点
        * "optical_flow_window" lucas-kanade光流算法的运算窗口,具体lucas-kanade
        会在下一篇详述
        * "5" 指示最大的金字塔层数，0表示只有一层，那就是没用金字塔算法
        * "optical_flow_found_feature" 用于指示在第二帧中是否找到对应特征值，
        若找到，其值为非零
        * "optical_flow_feature_error" 用于存放光流误差
        **********************************************************/

        /*开始为pyramid lucas kanade光流算法输入做准备*/
        CvPoint2D32f frame2_features[400];

        /* 该数组相应位置的值为非零，如果frame1中的特征值在frame2中找到 */
        char optical_flow_found_feature[400];

        /* 数组第i个元素表对应点光流误差*/
        float optical_flow_feature_error[400];

        /*lucas-kanade光流法运算窗口,这里取3*3的窗口,可以尝试下5*5,区别就是5*5
        出现aperture problem的几率较小,3*3运算量小，对于feature selection即shi-tomasi算法来说足够了*/
        CvSize optical_flow_window = cvSize(5, 5);
        // CvSize optical_flow_window = cvSize(5, 5);
        /* 终止规则，当完成20次迭代或者当epsilon<=0.3，迭代终止，可以尝试下别的值*/
        CvTermCriteria optical_flow_termination_criteria = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, .3);

        /*分配工作区域*/
        allocateOnDemand(&pyramid1, frame_size, IPL_DEPTH_8U, 1);
        allocateOnDemand(&pyramid2, frame_size, IPL_DEPTH_8U, 1);

        /*开始跑该算法*/
        cvCalcOpticalFlowPyrLK(frame1_1C, frame2_1C, pyramid1, pyramid2, frame1_features, frame2_features, number_of_features,
            optical_flow_window, 5, optical_flow_found_feature, optical_flow_feature_error, optical_flow_termination_criteria, 0);

        /*画光流场，画图是依据两帧对应的特征值，
        这个特征值就是图像上我们感兴趣的点，如边缘上的点P(x,y)*/
        for (int i = 0; i< number_of_features; i++)
        {
            /* 如果没找到对应特征点 */
            if (optical_flow_found_feature[i] == 0)
                continue;
            int line_thickness;
            line_thickness = 1;

            /* CV_RGB(red, green, blue) is the red, green, and blue components
            * of the color you want, each out of 255.
            */
            CvScalar line_color;
            line_color = CV_RGB(255, 0, 0);

            /*画箭头,因为帧间的运动很小，所以需要缩放，不然看不见箭头，缩放因子为3*/
            CvPoint p, q;
            p.x = (int)frame1_features[i].x;
            p.y = (int)frame1_features[i].y;
            q.x = (int)frame2_features[i].x;
            q.y = (int)frame2_features[i].y;

            double angle;
            angle = atan2((double)p.y - q.y, (double)p.x - q.x);
            double hypotenuse;
            hypotenuse = sqrt(square(p.y - q.y) + square(p.x - q.x));

            /*执行缩放*/
            q.x = (int)(p.x - 5 * hypotenuse * cos(angle));
            q.y = (int)(p.y - 5 * hypotenuse * sin(angle));

            /*画箭头主线*/
            /* "frame1"要在frame1上作画.
            * "p" 线的开始点.
            * "q" 线的终止点.
            * "CV_AA" 反锯齿.
            * "0" 没有小数位.
            */
            cvLine(frame1, p, q, line_color, line_thickness, CV_AA, 0);

            /* 画箭的头部*/
            p.x = (int)(q.x + 9 * cos(angle + pi / 4));
            p.y = (int)(q.y + 9 * sin(angle + pi / 4));
            cvLine(frame1, p, q, line_color, line_thickness, CV_AA, 0);
            p.x = (int)(q.x + 9 * cos(angle - pi / 4));
            p.y = (int)(q.y + 9 * sin(angle - pi / 4));
            cvLine(frame1, p, q, line_color, line_thickness, CV_AA, 0);
        }


        /*显示图像*/


        /*创建一个名为optical flow的窗口，大小自动改变*/
        cvNamedWindow("Optical Flow", CV_WINDOW_NORMAL);
        cvFlip(frame1, NULL, 2);
        cvShowImage("Optical Flow", frame1);

        /*延时，要不放不了*/
        cvWaitKey(33);

        /*写入到文件中去*/


        //	 cv::Mat m = cv::cvarrToMat(frame1);//转换lpimgae到mat格式
        //	 writer << m;//opencv3.0 version writer

    }
    cap.release();
    cvWaitKey(33);
    system("pause");
}