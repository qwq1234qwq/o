#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// 全局变量
vector<Point2f> points; // 存储鼠标点击的点
Mat image, transformed_image;
const int num_points = 4; // 需要标记的点数

// 鼠标回调函数
void onMouse(int event, int x, int y, int, void*) {
    if (event == EVENT_LBUTTONDOWN && points.size() < num_points) {
        points.push_back(Point2f(x, y));
        cout << "点 " << points.size() << ": (" << x << ", " << y << ")" << endl;
        circle(image, Point(x, y), 5, Scalar(0, 0, 255), -1); // 绘制标记点
        imshow("原图", image);
    }

    if (points.size() == num_points) {
        // 透视变换
        vector<Point2f> dst_points = {Point2f(0, 0), Point2f(400, 0), Point2f(400, 100), Point2f(0, 100)};
        Mat M = getPerspectiveTransform(points, dst_points);
        warpPerspective(image, transformed_image, M, Size(400, 100));
        imshow("变换后的图像", transformed_image);
    }
}

int main() {
    // 读取图像
    image = imread("/home/o/study/car/car.png");
    if (image.empty()) {
        cerr << "无法读取图像文件!" << endl;
        return -1;
    }

    transformed_image = Mat::zeros(100, 400, CV_8UC3); // 创建转换后的图像

    // 显示原图
    namedWindow("原图");
    imshow("原图", image);

    // 设置鼠标回调函数
    setMouseCallback("原图", onMouse);

    // 等待用户输入
    waitKey(0);

    return 0;
}
