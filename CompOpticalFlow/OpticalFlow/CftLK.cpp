

#include "../Interface.hxx"
#include "ImgProc.hxx"

//特征点个数定义
constexpr uint    FEATURE_MAX_NUM = 600;

/**
* \brief 对图像进行预处理
* \param base_image 待预处理的图像
*/
void handle_base_out_pre(cv::Mat& base_image)
{
    std::vector<cv::Mat> mv;

    //分离红色通道分量
    cv::split(base_image, mv);
    auto &red = mv.at(2);

    /// 膨胀操作
    int dilation_size = 1;
    auto element = getStructuringElement(cv::MORPH_RECT,
        cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
        cv::Point(dilation_size, dilation_size));
    dilate(red, red, element);
    dilate(red, red, element);
    dilate(red, red, element);
    //将分量组合
    cv::merge(mv, base_image);

    cv::imshow("superposition_window", base_image);

    cv::imwrite("叠加膨胀结果.png", base_image);
}
/**
* \brief 查找最大连通域
* \param base_image
*/
void handle_base_out_last(const cv::Mat& base_image, cv::Mat & backend_image)
{
    //最大面积区域需要显示几个
    constexpr   auto area_number = 3;

    cv::Mat		new_image = backend_image;

    assert(!new_image.empty());//检查是否为空


    std::vector<cv::Vec4i>          hierarchy;
    std::vector<cv::Mat>            contours;

    cv::Mat input;
    base_image.copyTo(input);

    cv::cvtColor(input, input, cv::COLOR_BGR2GRAY);

    input << binary;//二值化
    cv::imwrite("二值结果.png", input);

    cv::findContours(input, contours, hierarchy,
        cv::RetrievalModes::RETR_CCOMP,
        cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE);

    //输出查找到的所有轮廓
    {
        cv::Mat all_outline;
        new_image.copyTo(all_outline);
        cv::drawContours(all_outline, contours, -1, CV_RGB(0, 255, 255), 1, cv::LINE_AA);
        cv::imwrite("查找到的所有轮廓.png", all_outline);
    }

    /// Find the convex hull object for each contour  
    std::vector<std::vector<cv::Point2i> >	hull(contours.size()), dest_hull;
    // Int type hull  
    std::vector<std::vector<int>>	hullsI(contours.size());
    // Convexity defects  
    std::vector<cv::Mat>			defects(contours.size());

    //凹点还原
    for (size_t i = 0; i < contours.size(); i++)
    {
        cv::convexHull(contours[i], hull[i], false);
        // find int type hull  
        cv::convexHull(contours[i], hullsI[i], false);
        // get convexity defects  
        convexityDefects(contours[i], hullsI[i], defects[i]);
    }

    //查找最大轮廓
    for (auto i = 0; i < area_number; ++i)
    {
        auto max_contours = std::max_element(hull.begin(), hull.end(), [](const std::vector<cv::Point> &l, const std::vector<cv::Point> &r)->bool
        {
            auto area1 = fabs(cv::contourArea(l));
            auto area2 = fabs(cv::contourArea(r));
            return area1 < area2;
        });
        dest_hull.push_back(*max_contours);
        hull.erase(max_contours);
    }

    cv::drawContours(new_image, dest_hull, -1, CV_RGB(0, 255, 255), 1, cv::LINE_AA);

    //cv::drawContours(base_image, contours ,-1, CV_RGB(255, 0, 0),1,cv::LINE_AA);
    //根据凹点还原在原图连线，表示出最大连通点
    //for (size_t i = 0; i< contours.size(); i++)
    //{
    //	cv::Scalar color = CV_RGB(0,255,0);
    //	drawContours(new_image, hull, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point2i());

    //	//auto d = defects[i].begin<cv::Vec4i>();
    //	//while (d != defects[i].end<cv::Vec4i>()) {
    //	//	cv::Vec4i& v = (*d);
    //	//	//if(IndexOfBiggestContour == i)  
    //	//	{

    //	//		int startidx = v[0];
    //	//		cv::Point2i ptStart(contours[i].at<cv::Point2i>(startidx)); // point of the contour where the defect begins  
    //	//		int endidx = v[1];
    //	//		cv::Point2i ptEnd(contours[i].at<cv::Point2i>(endidx)); // point of the contour where the defect ends  
    //	//		int faridx = v[2];
    //	//		cv::Point2i ptFar(contours[i].at<cv::Point2i>(faridx));// the farthest from the convex hull point within the defect  
    //	//		int depth = v[3] / 256; // distance between the farthest point and the convex hull  

    //	//		if (depth > 20 && depth < 80)
    //	//		{
    //	//			//line(new_image, ptStart, ptFar, CV_RGB(0, 255, 0), 2);
    //	//			//line(new_image, ptEnd, ptFar, CV_RGB(0, 255, 0), 2);
    //	//			/*circle(base_image, ptStart, 4, cv::Scalar(255, 0, 100), 2);
    //	//			circle(base_image, ptEnd, 4, cv::Scalar(255, 0, 100), 2);
    //	//			circle(base_image, ptFar, 4, cv::Scalar(100, 0, 255), 2);*/
    //	//		}
    //	//	}
    //	//	++d;
    //	//}
    //}
    //cv::drawContours(base_image, defects, -1, CV_RGB(0, 255, 0), 1, cv::LINE_AA);
    cv::imshow("output_window", new_image);
    cv::imwrite("最终结果.png", new_image);
}
/**
* \brief 开始绘制运动轨迹 如需要
* \param m1 基本矩阵
* \param frame1_features 查找到的第一帧特征点
* \param frame2_features 查找到的第二帧特征点
* \param found_features 特征点错误矩阵
*/
void    draw_path_on_baseimage(
    cv::Mat &m1, 
    const cv::Mat &frame1_features,        
    const cv::Mat &frame2_features, 
    const cv::Mat &found_features,
    const cv::Mat &found_err)
{
    const static cv::Scalar line_color = CV_RGB(255, 0, 0);

    for (auto pos = 0; pos < FEATURE_MAX_NUM; ++pos)
    {
        /* 如果没找到对应特征点 */
        if (found_features.at<uchar>(pos) == 0 || found_err.at<float>(pos) > 1.500f)
            continue;

        auto p = frame1_features.at<cv::Point2f>(pos);
        auto q = frame2_features.at<cv::Point2f>(pos);

        if (segment_length(p, q) > 10.0f)
            continue;

        double angle;
        angle = atan2(static_cast<double>(p.y) - q.y, static_cast<double>(p.x) - q.x);
        double hypotenuse;
        hypotenuse = sqrt(square(p.y - q.y) + square(p.x - q.x));

        /*执行缩放*/
        q.x -= 5 * hypotenuse * cos(angle);
        q.y -= 5 * hypotenuse * sin(angle);

        /*画箭头主线*/
        cv::line(m1, p, q, line_color, 1, cv::LineTypes::LINE_AA, 0);

        /* 画箭的头部*/
        p.x = q.x + 9 * cos(angle + PI / 4);
        p.y = q.y + 9 * sin(angle + PI / 4);
        cv::line(m1, p, q, line_color, 1, cv::LineTypes::LINE_AA, 0);
        p.x = q.x + 9 * cos(angle - PI / 4);
        p.y = q.y + 9 * sin(angle - PI / 4);
        cv::line(m1, p, q, line_color, 1, cv::LineTypes::LINE_AA, 0);
    }
}
/**
* \brief 将特征点信息绘制到基础背景图像中去
* \param base_image 要绘制的图像
* \param frame1_features 第一帧特征点
* \param frame2_features 第二帧特征点
* \param found_features 特征点信息
*/
void draw_features_in_base_image(
    cv::Mat& base_image,    
    const cv::Mat& frame1_features,
    const cv::Mat& frame2_features, 
    const cv::Mat& found_features,
    const cv::Mat& found_err)
{
    const static cv::Scalar area_color = CV_RGB(255, 0, 0);

    for (auto pos = 0; pos < FEATURE_MAX_NUM; ++pos)
    {
        /* 如果没找到对应特征点 */
        if (found_features.at<uchar>(pos) == 0 || found_err.at<float>(pos) > 1.500f)
            continue;

        auto start_p = frame1_features.at<cv::Point2f>(pos);
        auto end_p = frame2_features.at<cv::Point2f>(pos);
        //如果线段长度大于某个值，说明不是细微运动则排除该线段
        if (segment_length(start_p, end_p) > 10.0f)
            return;

        cv::line(base_image, start_p, end_p, area_color, 3, cv::LINE_AA);
    }
}



void count_features_freq(
    cv::Mat& base_count,
    const cv::Mat& frame1_features,
    const cv::Mat& frame2_features,
    const cv::Mat& found_features,
    const cv::Mat& found_err)
{
    
    for (auto pos = 0; pos < FEATURE_MAX_NUM; ++pos)
    {
        /* 如果没找到对应特征点 */
        if (found_features.at<uchar>(pos) == 0 || found_err.at<float>(pos) > 1.500f)
            continue;

        auto start_p = frame1_features.at<cv::Point2f>(pos);
        auto end_p = frame2_features.at<cv::Point2f>(pos);
        //如果线段长度大于某个值，说明不是细微运动则排除该线段
        if (segment_length(start_p, end_p) > 10.0f)
            return;      

        base_count.at<double>(start_p) ++;

        base_count.at<double>(end_p)++;
    }
}

/**
* \brief 开始LK光流法算法
* \param base_out 叠加输出基本源图像
*/
void draw_features_by_lk(cv::Mat & base_out, cv::Mat &base_count,const std::string & in_video_name, const std::string & out_video_name)
{
    //打开视频文件并且创建进度条
    cv::VideoCapture capture{};
    auto frame_count = open_video(capture, in_video_name);
    //boost::progress_display progress_bar(frame_count, std::cout, "当前处理进度\n\t\t", "总进度\t\t", "当前进度\t");
    //打开输出视频文件 如需要
    cv::VideoWriter writer(out_video_name, CV_FOURCC('X', 'V', 'I', 'D'), 29.0, cv::Size{ 656,488 });

    //如果还有新帧则继续
    while (capture.grab())
    {
        cv::Mat     m1, m2, frame1, frame2;
        //取出视频两帧
        capture >> m1;
        capture >> m2;
        //如果返回不足两帧则退出
        if (m1.empty() || m2.empty())
            break;
        //复制到新空间
        frame1 = m1;
        frame2 = m2;
        //两帧灰度化
        frame1 << cvtgray;
        frame2 << cvtgray;

        auto sc = cv::mean(frame1);

        if (sc[0] < 1.0f)
            continue;

        //创建两个特征值存储矩阵
        cv::Mat frame1_features, frame2_features;
        //开始进行特征值检测，先检测出第一帧特征值
        cv::goodFeaturesToTrack(frame1, frame1_features, FEATURE_MAX_NUM, .01, .01);
        //创建查找错误矩阵
        cv::Mat found_futures, found_err;
        //创建迭代算子
        cv::TermCriteria optical_flow_termination_criteria(cv::TermCriteria::Type::MAX_ITER | cv::TermCriteria::Type::EPS, 20, .3);
        //开始进行LK光流法计算，查找第二个特征值


        cv::calcOpticalFlowPyrLK(
            frame1, frame2,
            frame1_features, frame2_features,
            found_futures, found_err,
            cv::Size{ 5,5 }, 5, optical_flow_termination_criteria);

        //绘制移动轨迹 ， 如需要
        draw_path_on_baseimage(m1, frame1_features, frame2_features, found_futures,found_err);
        //绘制运动轨迹线到背景矩阵
        draw_features_in_base_image(base_out, frame1_features, frame2_features, found_futures,found_err);
        //计算频率
        count_features_freq(base_count, frame1_features, frame2_features, found_futures, found_err);

        //进度条更新
        //progress_bar += 2;

        //将运动轨迹线展示到窗口，如需要
        cv::imshow("output_window", m1);
        cv::imshow("superposition_window", base_out);
        cv::imshow("freq_window", base_count);

        //将运动轨迹线输出到视频文件 如需要
        writer << m1;
        //writer << (out << cvtbgr);
    }

    writer.release();
    capture.release();
}



void       draw_freq_count(const cv::Mat & base_count)
{

    cv::Scalar a = CV_RGB(0, 0, 0);
    cv::Scalar b = CV_RGB(0, 0, 255);

    cv::Mat         freq_count_mat = cv::Mat::zeros(base_count.rows, base_count.cols, CV_8UC3 );

    double min_elm;
    double max_elm;
    cv::minMaxIdx(base_count, &min_elm, &max_elm);

    std::cout << "Max:" << max_elm << "   Min:" << min_elm << std::endl;

    cv::Scalar weight;
    weight[0] = (b.val[0] - a.val[0]) / max_elm;
    weight[1] = (b.val[1] - a.val[1]) / max_elm;
    weight[2] = (b.val[2] - a.val[2]) / max_elm;

    std::cout << "B:" << weight[0] << "   G:" << weight[1] << " R:"<< weight[2] <<std::endl;

//    if (max_elm != 0.0f)
//    {
    for(auto bg = base_count.begin<double>();bg!=base_count.end<double>();++bg)
    {
        freq_count_mat.at<cv::Vec3b>(bg.pos())[0] += cvRound(*bg * weight[0]);
        freq_count_mat.at<cv::Vec3b>(bg.pos())[1] += cvRound(*bg * weight[1]);
        freq_count_mat.at<cv::Vec3b>(bg.pos())[2] += cvRound(*bg * weight[2]);       
    }
    //}
    cv::imshow("freq_window", freq_count_mat);
}


int         ctfLKOpticalFlow(int argc, char **args)
{
    std::string input_video_filename = "videos/1.avi";
    std::string base_image_filename = "";
    std::string base_output_filename = "VideoOut.avi";
    if (argc == 4)
        base_output_filename = args[3];

    //创建窗口  
    create_thread_window( "output_window");
    create_thread_window( "superposition_window");
    create_thread_window("freq_window");

    //boost::thread(output_windows_thread).detach();    
    //打开背景图像
    cv::Mat base_image;
    if (base_image_filename.empty())
    {
        cv::VideoCapture capture;
        open_video(capture, input_video_filename);
        capture >> base_image;
        capture.release();
    }
    else
        base_image = cv::imread(base_image_filename);

    //打开是否失败
    assert(!base_image.empty());

    //创建空白背景图像
    //cv::Mat         base_out{ base_image.rows,base_image.cols, base_image.type() };

    cv::Mat         base_out = cv::Mat::zeros( base_image.rows , base_image.cols, base_image.type() );

    //cv::Mat         base_count = cv::Mat::zeros( base_image.rows,base_image.cols, CV_64FC3 );

    cv::Mat         base_count = cv::Mat::zeros( base_image.rows,base_image.cols,CV_64FC1 ) ;

    //cv::Mat          base_out = cv::imread("post.png");
    //开始进行LK光流算法，将结果图像输出到base out
    draw_features_by_lk(base_out,base_count, input_video_filename, base_output_filename);
    /*
    * ！！！！此处开始对LK算法处理后的图像进行处理！！！！
    */
    //对叠加输出图片进行预处理
    //handle_base_out_pre(base_out);
    //对预处理图片进行整合处理，边缘过滤，凹点过滤
    //handle_base_out_last(base_out, base_image);

    double max_elm;
    double min_elm;
    cv::minMaxIdx(base_count, &min_elm, &max_elm);
    base_count.forEach<double>([&max_elm](double &v,const int *)
    {
        v = (v / max_elm) * 100.0f;
    });

    cv::imwrite("freq_mat.png", base_count);

    draw_freq_count(base_count);
    
    boost::this_thread::sleep(boost::posix_time::seconds(10000));
    return 0;
}
