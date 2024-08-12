#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Helper function to sort points in a specific order
vector<Point2f> sortPoints(vector<Point2f> points) {
    // Calculate the centroid
    Point2f centroid(0, 0);
    for (const auto& pt : points) {
        centroid += pt;
    }
    centroid.x /= points.size();
    centroid.y /= points.size();

    // Sort points based on their angle with respect to the centroid
    sort(points.begin(), points.end(), [&centroid](const Point2f& a, const Point2f& b) {
        double angleA = atan2(a.y - centroid.y, a.x - centroid.x);
        double angleB = atan2(b.y - centroid.y, b.x - centroid.x);
        return angleA < angleB;
    });

    return points;
}

// Function to determine the new image boundaries based on pixel counts
Rect getImageBounds(const Mat& binaryImage) {
    int rows = binaryImage.rows;
    int cols = binaryImage.cols;

    // Divide image into two halves
    int halfRows = rows / 2;

    // Variables to store row and column boundaries
    int topBoundary = -1;
    int bottomBoundary = -1;
    int leftBoundary = -1;
    int rightBoundary = -1;

    // Analyze the top half for horizontal boundaries
    for (int y = 0; y < halfRows; y++) {
        int whiteCount = countNonZero(binaryImage.row(y));
        if (whiteCount < 1) {
            if (topBoundary == -1) topBoundary = y;
        }
    }

    // Analyze the bottom half for horizontal boundaries
    for (int y = halfRows; y < rows; y++) {
        int whiteCount = countNonZero(binaryImage.row(y));
        if (whiteCount < 1) {
            if (bottomBoundary == -1) bottomBoundary = y;
        }
    }

    // Analyze columns to find vertical boundaries
    for (int x = 0; x < cols; x++) {
        int blackCount = countNonZero(binaryImage.col(x) == 0);
        if (blackCount > 140) {
            if (leftBoundary == -1) leftBoundary = x;
            rightBoundary = x;
        }
    }

    // If boundaries are not found, use the image edges as default
    if (topBoundary == -1) topBoundary = 0;
    if (bottomBoundary == -1) bottomBoundary = rows - 1;
    if (leftBoundary == -1) leftBoundary = 0;
    if (rightBoundary == -1) rightBoundary = cols - 1;

    return Rect(leftBoundary, topBoundary, rightBoundary - leftBoundary, bottomBoundary - topBoundary);
}

int main() 
{
    // 加载汽车图片
    Mat img = imread("car.png");
    if (img.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    // 转换到 HSV 颜色空间
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);

    // 设定蓝色的 HSV 范围
    Scalar lower_blue(95, 137, 95);  // HSV 下限
    Scalar upper_blue(133, 255, 255);  // HSV 上限
    Mat blue_mask;
    inRange(hsv, lower_blue, upper_blue, blue_mask);
	//imshow("1", blue_mask);

    // 形态学操作 - 闭运算 (填补空洞)
    morphologyEx(blue_mask, blue_mask, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(50, 50)));
	morphologyEx(blue_mask, blue_mask, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(5, 5)));
	//imshow("2", blue_mask);

    Mat edges;
    Canny(blue_mask, edges, 50, 150);
	//imshow("3", edges);

    // Find contours
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Find the largest contour (assumed to be the quadrilateral)
    double maxArea = 0;
    vector<Point> largestContour;
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area > maxArea) {
            maxArea = area;
            largestContour = contour;
        }
    }

    // Approximate the contour to a polygon
    vector<Point> approx;
    approxPolyDP(largestContour, approx, arcLength(largestContour, true) * 0.02, true);

    // Check if the polygon has 4 points
    if (approx.size() != 4) {
        cerr << "No quadrilateral detected!" << endl;
        return -1;
    }

    // Sort points in a specific order
    vector<Point2f> srcPoints;
    for (const auto& pt : approx) {
        srcPoints.push_back(pt);
    }
    srcPoints = sortPoints(srcPoints);

    // Define destination points for the perspective transform
    vector<Point2f> dstPoints = { Point2f(0, 0), Point2f(400, 0), Point2f(400, 200), Point2f(0, 200) };

    // Compute the perspective transform matrix
    Mat M = getPerspectiveTransform(srcPoints, dstPoints);

    // Apply the perspective transform
    Mat dst;
    warpPerspective(img, dst, M, Size(400, 200));

    Mat bin;
    cvtColor(dst, bin, COLOR_BGR2GRAY);
    threshold(bin,bin,100,255,THRESH_BINARY);

	// Determine the new image boundaries
    Rect bounds = getImageBounds(bin);
    // Crop the image to the new boundaries
    Mat cropped = dst(bounds);
	cvtColor(cropped, cropped, COLOR_BGR2GRAY);
    threshold(cropped,cropped,100,255,THRESH_BINARY);


	imshow("原图", img);
    imshow("车牌正视", dst);
	imshow("二值化", bin);
	imshow("裁剪", cropped);
    waitKey(0);

    return 0;
}
