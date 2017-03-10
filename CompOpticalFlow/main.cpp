
#include "interface.hxx"

int main(int argc , char **args)
{

    cv::Mat base_count = cv::imread("freq_mat.png");

    cv::cvtColor(base_count, base_count, CV_BGR2GRAY);

    double max_elm;
    double min_elm;
    cv::Point max_elm_idx, min_elm_idx;
    cv::minMaxLoc(base_count, &min_elm, &max_elm,&min_elm_idx,&max_elm_idx);

    std::cout << "Channel " << base_count.channels() << " Size:" << base_count.elemSize() << std::endl;

    cv::Point3i coor{ base_count.rows,base_count.cols, 100 };
    create_gl_gird(argc, args , coor, base_count);

    return 0;
    //return ctfLKOpticalFlow(argc, args);
}

