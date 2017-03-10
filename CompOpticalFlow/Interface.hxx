#pragma once


#include "Common.hxx"

#define ATD at<double>
#define elif else if

#ifndef bool
#define bool int
#define false ((bool)0)
#define true  ((bool)1)
#endif

int         HSOpticalFlow(int argc, char **args);
int         LKOpticalFlow(int argc, char **args);
int         ctfLKOpticalFlow(int argc, char **args);
int         opengl_demo(int argc, char* argv[]);
int         Gl_Gird(int argc, char** argv);
int         create_gl_gird(int argc, char **args,const cv::Point3i &coor, const cv::Mat &count);



