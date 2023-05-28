#include "light.h"
using namespace cv;
using namespace std;

int light(cv::Mat& image, cv::Scalar& lower_red1, cv::Scalar& upper_red1, cv::Scalar& lower_red2, cv::Scalar& upper_red2) {
    Rect roi(6,6, image.cols -12, image.rows-12);
    image = image(roi);
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);

    Mat red_mask1, red_mask2;
    inRange(hsv, lower_red1, upper_red1, red_mask1);
    inRange(hsv, lower_red2, upper_red2, red_mask2);
    int white_count = cv::countNonZero(red_mask1) + cv::countNonZero(red_mask2);
    if(white_count <= 80) {
        return 0;
    } else {
        return 1;
    }
}
