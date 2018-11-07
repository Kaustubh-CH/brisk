#include <vector>
#include <iostream>
#include <cstdlib>
#include <fstream>
//#include <algorithm>
#include <opencv2/opencv.hpp>

using std::cout;
using std::cerr;
using std::vector;
using std::string;

using cv::Mat;
using cv::Point2f;
using cv::KeyPoint;
using cv::Scalar;
using cv::Ptr;


using cv::BRISK;

int main(int argc, char** argv) {

    Mat img = cv::imread("pdp.jpg", CV_LOAD_IMAGE_COLOR);
    if (img.channels() != 1) {
        cvtColor(img, img, cv::COLOR_RGB2GRAY);
    }

    cv::imshow("image",img);

    vector<KeyPoint> kpts;

    Mat desc1;
    Mat desc2;

  
    Ptr<BRISK> brisk = BRISK::create();
    brisk->detectAndCompute(img, Mat(), kpts, desc1);
    
    cout<<desc1;
  /*  for(auto a:kpts)
  */      
cv::imshow("Res",desc1);
    return 0;
}