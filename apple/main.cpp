#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

// 填充函数
void fillHole(const Mat srcBw, Mat &dstBw) {
    Size m_Size = srcBw.size();
    Mat Temp = Mat::zeros(m_Size.height + 2, m_Size.width + 2, srcBw.type()); // 延展图像
    srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));

    cv::floodFill(Temp, Point(0, 0), Scalar(255)); // 填充区域

    Mat cutImg; // 裁剪延展的图像
    Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);

    dstBw = srcBw | (~cutImg);
}

int main() {
    // 读取图像
    cv::Mat image = cv::imread("/home/o/study/apple/apple.png");
    if (image.empty()) {
        std::cerr << "无法读取图像文件!" << std::endl;
        return -1;
    }

    // 转换到 HSV 颜色空间
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // 定义苹果的颜色范围 (HSV)
    cv::Scalar lower_red1(0, 180, 103);
    cv::Scalar upper_red1(26, 255, 255);
    cv::Scalar lower_red2(160, 100, 100);
    cv::Scalar upper_red2(180, 255, 255);

    // 创建掩码
    cv::Mat mask1, mask2, mask;
    cv::inRange(hsv, lower_red1, upper_red1, mask1);
    cv::inRange(hsv, lower_red2, upper_red2, mask2);
    mask = mask1 | mask2;
    fillHole(mask, mask);

    // 进行形态学操作来去除噪声
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(20, 20));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

    // 找到轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point>> filteredContours;
    cv::Mat contourMask = cv::Mat::zeros(mask.size(), CV_8UC1); // 创建用于过滤的小轮廓掩码

    // 过滤大轮廓
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > 500) { // 只保留大于 500 像素的轮廓
            filteredContours.push_back(contour);
            cv::drawContours(contourMask, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), cv::FILLED);
        }
    }

    // 仅保留苹果区域
    cv::Mat apple_region;
    cv::bitwise_and(image, image, apple_region, mask);

    // 创建一个全黑的掩码来移除小轮廓区域
    cv::Mat removeSmallContours = cv::Mat::zeros(mask.size(), CV_8UC1);
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area <= 500) { // 移除小轮廓区域
            cv::drawContours(removeSmallContours, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), cv::FILLED);
        }
    }

    // 从苹果区域中移除小轮廓区域
    cv::Mat apple_region_without_small_contours;
    cv::bitwise_and(apple_region, apple_region, apple_region_without_small_contours, ~removeSmallContours);

    // 绘制大轮廓的外接框
    for (const auto& contour : filteredContours) {
        // 计算外接框
        cv::Rect boundingBox = cv::boundingRect(contour);
        // 绘制外接框
        cv::rectangle(apple_region_without_small_contours, boundingBox, cv::Scalar(0, 255, 0), 2);
    }

    // 显示结果
    cv::imshow("原图", image);
    cv::imshow("扣图", apple_region_without_small_contours);

    cv::waitKey(0);
    return 0;
}
