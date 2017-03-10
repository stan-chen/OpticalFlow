

#include "../Interface.hxx"
#include <GL/glew.h>
#include <GL/glut.h>


namespace {

    //设置程序的窗口大小
    GLint windowWidth = 500;
    GLint windowHeight = 500;
    //绕x轴旋转角度
    GLfloat xRotAngle = 0.0f;
    //绕y轴旋转角度
    GLfloat yRotAngle = 0.0f;
    //受支持的点大小范围
    GLfloat sizes[2];
    //受支持的点大小增量
    GLfloat step;


    cv::Point3f     AXES_COOR{100.0,200.0,100.0};

    cv::Mat         AXES_DATA;

    constexpr GLint AXES_LEN = 100;

    struct coordpoint
    {
        GLint x, y, z;
    };


    void    init()
    {
        //设置清理颜色为黑色
        glClearColor(0.0f, 0.0, 1.0, 1.0f);
        //设置绘画颜色为绿色
        glColor3f(1.0f, 1.0f, 0.0f);
        //使能深度测试
        glEnable(GL_DEPTH_TEST);
        //获取受支持的点大小范围
        glGetFloatv(GL_POINT_SIZE_RANGE, sizes);
        //获取受支持的点大小增量
        glGetFloatv(GL_POINT_SIZE_GRANULARITY, &step);
        printf("point size range:%f-%f\n", sizes[0], sizes[1]);
        printf("point step:%f\n", step);
    }

    //窗口大小变化回调函数
    void reshape(GLint w, GLint h) {
        //横宽比率
        GLfloat ratio;
        //设置世界坐标系为200.0f
        GLfloat coordinatesize = 100.0f;

        GLfloat coor_left = AXES_COOR.x;
        GLfloat coor_right = AXES_COOR.y;
        GLfloat coor_top = 150.0f;
        GLfloat coor_bottom = 50.0f;

        GLfloat coor_near = sqrt((AXES_COOR.y*AXES_COOR.y) + (AXES_COOR.x*AXES_COOR.x));
        GLfloat coor_far = coor_near;


        //窗口宽高为零直接返回
        if ((w == 0) || (h == 0))
            return;
        //设置视口和窗口大小一致
        glViewport(0, 0, w, h);
        //对投影矩阵应用随后的矩阵操作
        glMatrixMode(GL_PROJECTION);
        //重置当前指定的矩阵为单位矩阵　
        glLoadIdentity();

        ratio = (GLfloat)w / (GLfloat)h;
        //正交投影
        if (w < h)
            glOrtho(
                -coor_left,
                coor_right,
                -coor_bottom / ratio,
                coor_top / ratio,
                -coor_near*2,
                coor_far*2);
        else
            glOrtho(
                -coor_left*ratio,
                coor_right*ratio,
                -coor_bottom,
                coor_top,
                -coor_near*2,
                coor_far*2);


        //对模型视图矩阵堆栈应用随后的矩阵操作
        glMatrixMode(GL_MODELVIEW);
        //重置当前指定的矩阵为单位矩阵　
        glLoadIdentity();
    }

#if 0
    void    GLGrid(coordpoint& pt1, coordpoint& pt2, int num)
    {
        const float _xLen = (pt2.x - pt1.x) / num;
        const float _yLen = (pt2.y - pt1.y) / num;
        const float _zLen = (pt2.z - pt1.z) / num;

        glLineWidth(1.0f);
        glLineStipple(1, 0x0303);//线条样式
        glBegin(GL_LINES);
        glEnable(GL_LINE_SMOOTH);
        //glColor3f(0.0f,0.0f, 1.0f); //白色线条
        int xi = 0;
        int yi = 0;
        int zi = 0;

        //绘制平行于X的直线
        for (zi = 0; zi <= num; zi++)
        {
            float z = _zLen * zi + pt1.z;
            for (yi = 0; yi <= num; yi++)
            {
                float y = _yLen * yi + pt1.y;
                glVertex3f(pt1.x, y, z);
                glVertex3f(pt2.x, y, z);
            }
        }
        //绘制平行于Y的直线
        for (zi = 0; zi <= num; zi++)
        {
            float z = _zLen * zi + pt1.z;
            for (xi = 0; xi <= num; xi++)
            {
                float x = _xLen * xi + pt1.x;
                glVertex3f(x, pt1.y, z);
                glVertex3f(x, pt2.y, z);
            }
        }
        //绘制平行于Z的直线
        for (yi = 0; yi <= num; yi++)
        {
            float y = _yLen * yi + pt1.y;
            for (xi = 0; xi <= num; xi++)
            {
                float x = _xLen * xi + pt1.x;
                glVertex3f(x, y, pt1.z);
                glVertex3f(x, y, pt2.z);
            }
        }
        glEnd();
    }


    void    draw_grid()
    {
        //画X网格线
        coordpoint cpoint1 = { -50,0,-50 };
        coordpoint cpoint2 = { 50,0,50 };
        glColor3f(0.9f, 0.9f, 0.9f);
        GLGrid(cpoint1, cpoint2, AXES_LEN);

        //画Y网格线
        glPushMatrix();
        {
            glRotatef(90, 1.0, 0.0, 0.0);
            glTranslatef(0.0f, -5, -5);
            coordpoint cpoint3 = { -5,00,-5 };
            coordpoint cpoint4 = { 5,00,5 };
            glColor3f(0.9f, 0.9f, 0.0f);
            GLGrid(cpoint3, cpoint4, AXES_LEN);
        }
        glPopMatrix();

        //画Z网格线
        glPushMatrix();
        glRotatef(90, 0.0, 0.0, 1.0);
        glTranslatef(5, 5, -0);
        coordpoint cpoint5 = { -5,0,-5 };
        coordpoint cpoint6 = { 5,0,5 };
        glColor3f(0.0f, 0.9f, 0.0f);
        GLGrid(cpoint5, cpoint6, AXES_LEN);
        glPopMatrix();
    }
#endif

    void    draw_test()
    {
        std::cout << "X :" << AXES_COOR.x << " Y:" << AXES_COOR.y << " Z:" << AXES_COOR.z << std::endl;
        //绘制立体坐标系
        GLUquadricObj *objCylinder = gluNewQuadric();
        glRotatef(-45, 0.0, 1.0, 0.0);

        //画坐标系原点
        glPushMatrix();
        glColor3f(1.0f, 1.0f, 1.0f);
        glTranslatef(-AXES_COOR.x, 0, -AXES_COOR.y);
        glutSolidSphere(1.0f, 20, 20);
        //glutSolidTorus(0.2,1,10,10);圆环
        glPopMatrix();

        //画坐标轴
        //X
        glPushMatrix();
        glColor3f(0.0f, 1.0f, 0.0f);
        glTranslatef(-AXES_COOR.x, 0, -AXES_COOR.y);  //从原点开始
        gluCylinder(objCylinder, 0.5, 0.5, AXES_COOR.x, 10, 5);           //X
        glTranslatef(0, 0, AXES_COOR.x);       //移动x距离画尖尖
        gluCylinder(objCylinder, 0.5, 0.0, 0.5, 10, 5); //画尖尖
        glRasterPos2i(0, 0);
        glutBitmapCharacter(GLUT_BITMAP_8_BY_13, int('X'));
        glPopMatrix();
     
        //Y
        glPushMatrix();
        glColor3f(1, 0, 0.0f);
        glTranslatef(-AXES_COOR.x, 0, -AXES_COOR.y);
        glRotatef(90, 0.0, 1.0, 0.0);   
        gluCylinder(objCylinder, 0.5, 0.5, AXES_COOR.y, 10, 5);           //Y
        glTranslatef(0, 0, AXES_COOR.y);
        gluCylinder(objCylinder, 0.5, 0.0, 0.5, 10, 5);                 //Y
        glRasterPos2i(0, 0);
        glutBitmapCharacter(GLUT_BITMAP_8_BY_13, int('Y'));
        glPopMatrix();

        //Z
        glPushMatrix();
        glColor3f(1, 1, 0.0f);
        glTranslatef(-AXES_COOR.x, 0, -AXES_COOR.y);
        glRotatef(-90, 1.0, 0.0, 0.0);
        gluCylinder(objCylinder, 0.5, 0.5, AXES_COOR.z, 10, 5);           //Z
        glTranslatef(0, 0, AXES_COOR.z);
        gluCylinder(objCylinder, 0.5, 0.0, 0.5, 10, 5);                 //Z
        glRasterPos2i(0, 0);
        glutBitmapCharacter(GLUT_BITMAP_8_BY_13, int('Z'));
        glPopMatrix();

        glPushMatrix();
        glRotatef(-90, 1.0, 0.0, 0.0);
        for(auto bg = AXES_DATA.begin<uchar>();bg!= AXES_DATA.end<uchar>();++bg)
        {
            glPushMatrix();
            auto pt = bg.pos();
            glTranslatef(pt.x, -pt.y, 0);
            gluCylinder(objCylinder, 0.5, 0.5, *bg , 10, 5);           //Z
            glPopMatrix();
        }
        glPopMatrix();


        /*****取消反锯齿*****/
        glDisable(GL_BLEND);
        glDisable(GL_LINE_SMOOTH);
        glDisable(GL_POINT_SMOOTH);
        glDisable(GL_POLYGON_SMOOTH);
    }

#if 0
    void    draw_cubie()
    {
        //绘制立体坐标系
        GLUquadricObj *objCylinder = gluNewQuadric();
        glRotatef(-45, 0.0, 1.0, 0.0);

        //画坐标系原点
        glPushMatrix();
        glColor3f(1.0f, 1.0f, 1.0f);
        glTranslatef( -AXES_COOR.x , 0, -AXES_COOR.z ); 
        glutSolidSphere(0.2, 20, 20);
        //glutSolidTorus(0.2,1,10,10);圆环
        glPopMatrix();

        //画网格线

        //画坐标轴
        glPushMatrix();
        glColor3f(0.0f, 1.0f , 0.0f);
        glTranslatef(-AXES_COOR.x, 0, -AXES_COOR.z);  //从原点开始
        gluCylinder(objCylinder, 0.05, 0.05, AXES_COOR.x, 10, 5);           //X
        glTranslatef(0, 0, AXES_COOR.x);       //终点x轴结束
        gluCylinder(objCylinder, 0.2, 0.0, 0.5, 10, 5);                 //X
        glPopMatrix();

        glPushMatrix();
        glTranslatef(-AXES_COOR.x, 0, -AXES_COOR.z);
        glTranslatef(0, 0.2, AXES_COOR.x);
        glRotatef(90, 0.0, 1.0, 0.0);
        glPopMatrix();

        glPushMatrix();
        glColor3f(1, 0, 0.0f);
        glTranslatef(-AXES_COOR.x, 0, -AXES_COOR.z);
        glRotatef(90, 0.0, 1.0, 0.0);
        gluCylinder(objCylinder, 0.05, 0.05, AXES_COOR.y, 10, 5);           //Y
        glTranslatef(0, 0, AXES_COOR.y);
        gluCylinder(objCylinder, 0.2, 0.0, 0.5, 10, 5);                 //Y
        glPopMatrix();

        glPushMatrix();
        glTranslatef(-AXES_COOR.x, 0, -AXES_COOR.z);
        glRotatef(90, 0.0, 1.0, 0.0);
        glTranslatef(0, 0.2, AXES_COOR.y);
        glRotatef(90, 0.0, 1.0, 0.0);
        glPopMatrix();

        glPushMatrix();
        glColor3f(1, 1, 0.0f);
        glTranslatef(-AXES_COOR.x, 0, -AXES_COOR.z);
        glRotatef(-90, 1.0, 0.0, 0.0);
        gluCylinder(objCylinder, 0.05, 0.05, AXES_COOR.z, 10, 5);           //Z
        glTranslatef(0, 0, AXES_COOR.z);
        gluCylinder(objCylinder, 0.2, 0.0, 0.5, 10, 5);                 //Z
        glPopMatrix();

        glPushMatrix();
        glTranslatef(-AXES_COOR.x, 0, -AXES_COOR.z);
        glRotatef(-90, 1.0, 0.0, 0.0);
        glTranslatef(0.0, 0.6, AXES_COOR.z);
        glRotatef(90, 0.0, 1.0, 0.0);
        glRotatef(90, 0.0, 0.0, 1.0);
        glPopMatrix();

        /*****取消反锯齿*****/
        glDisable(GL_BLEND);
        glDisable(GL_LINE_SMOOTH);
        glDisable(GL_POINT_SMOOTH);
        glDisable(GL_POLYGON_SMOOTH);
    }
#endif


    void    draw()
    {
        //将窗口颜色清理为黑色
        glClearColor(0.3f, 0.0f, 0.0f, 0.0f);
        //将模板缓冲区值全部清理为1
        glClearStencil(1);
        //使能模板缓冲区
        glEnable(GL_STENCIL_TEST);
        //把整个窗口清理为当前清理颜色：黑色。清除深度缓冲区、模板缓冲区
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        glPushMatrix();

        //坐标系绕x轴旋转xRotAngle
        glRotatef(xRotAngle, 1.0f, 0.0f, 0.0f);
        //坐标系绕y轴旋转yRotAngle
        glRotatef(yRotAngle, 0.0f, 1.0f, 0.0f);
        //进行平滑处理　
        glEnable(GL_POINT_SMOOTH);
        glHint(GL_POINT_SMOOTH, GL_NICEST);
        glEnable(GL_LINE_SMOOTH);
        glHint(GL_LINE_SMOOTH, GL_NICEST);
        glEnable(GL_POLYGON_SMOOTH);
        glHint(GL_POLYGON_SMOOTH, GL_NICEST);

        draw_test();

        //恢复压入栈的Matrix
        glPopMatrix();
        //交换两个缓冲区的指针
        glutSwapBuffers();
    }

    //按键输入处理回调函数
    void keydown(int key, int x, int y) {

        if (key == GLUT_KEY_UP) {
            xRotAngle -= 2.0f;
        }
        else if (key == GLUT_KEY_DOWN) {
            xRotAngle += 2.0f;
        }
        else if (key == GLUT_KEY_LEFT) {
            yRotAngle -= 2.0f;
        }
        else if (key == GLUT_KEY_RIGHT) {
            yRotAngle += 2.0f;
        }
        //重新绘制
        glutPostRedisplay();
    }


}



int create_gl_gird(int argc , char **args , const cv::Point3i &coor , const cv::Mat &count)
{
    AXES_COOR = coor;
    AXES_DATA = count;

    glutInit(&argc, args);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_STENCIL);
    glutInitWindowSize(windowWidth, windowHeight);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("draw");

    glutReshapeFunc(reshape);
    glutDisplayFunc(draw);
    glutSpecialFunc(keydown);
    init();

    glutMainLoop();

    return 0;
}


int Gl_Gird(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_STENCIL);
    glutInitWindowSize(windowWidth, windowHeight);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("draw");
   
    glutReshapeFunc(reshape);
    glutDisplayFunc(draw);
    glutSpecialFunc(keydown);
    init();

    glutMainLoop();
    return 0;
}
