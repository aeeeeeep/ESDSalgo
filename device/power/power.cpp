#include"power.h"

int power(cv::Mat& image, cv::Scalar& lower_red, cv::Scalar& upper_red) {
    image.convertTo(image, -1, 1.2, 0);
    cv::Rect roi(6,6, image.cols -12, image.rows-12);
    image = image(roi);
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    cv::Mat red_mask;
    cv::inRange(hsv, lower_red, upper_red, red_mask);
    int white_count = cv::countNonZero(red_mask);
    if(white_count == 0) {
        return 0;
    } else {
        return 1;
    }
}
