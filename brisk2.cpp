
// Example 16-2. 2D Feature detectors and 2D Extra Features framework
//
// Note, while this code is free to use commercially, not all the algorithms are. For example
// sift is patented. If you are going to use this commercially, check out the non-free 
// algorithms and secure license to use them.
//

#include <vector>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/ocl.hpp>

using std::cout;
using std::cerr;
using std::vector;
using std::string;

using cv::Mat;
using cv::Point2f;
using cv::KeyPoint;
using cv::Scalar;
using cv::Ptr;

using cv::FastFeatureDetector;
using cv::SimpleBlobDetector;

using cv::DMatch;
using cv::BFMatcher;
using cv::DrawMatchesFlags;
using cv::Feature2D;
using cv::ORB;
using cv::BRISK;
using cv::AKAZE;
using cv::KAZE;

using cv::xfeatures2d::BriefDescriptorExtractor;
using cv::xfeatures2d::SURF;
using cv::xfeatures2d::SIFT;
using cv::xfeatures2d::DAISY;
using cv::xfeatures2d::FREAK;

const double kDistanceCoef = 4.0;
const int kMaxMatchingSize = 50;


inline void findKeyPointsHomography(vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2,
        vector<DMatch>& matches, vector<char>& match_mask) {
    if (static_cast<int>(match_mask.size()) < 3) {
        return;
    }
    vector<Point2f> pts1;
    vector<Point2f> pts2;
    for (int i = 0; i < static_cast<int>(matches.size()); ++i) {
        pts1.push_back(kpts1[matches[i].queryIdx].pt);
        pts2.push_back(kpts2[matches[i].trainIdx].pt);
    }
    findHomography(pts1, pts2, cv::RANSAC, 4, match_mask);
}

int main(int argc, char** argv) {
    // Program expects at least four arguments:
    //   - descriptors type ("surf", "sift", "orb", "brisk",
    //          "kaze", "akaze", "freak", "daisy", "brief").
    //          For "brief", "freak" and "daisy" you also need a prefix
    //          that is either "blob" or "fast" (e.g. "fastbrief", "blobdaisy")
    //   - match algorithm ("bf", "knn")
    //   - path to the object image file
    //







    // path to the scene image file
    
   

  //  string desc_type;
    //strcpy(desc_type,"brisk");

    //string match_type;

    //string img_file1(argv[3]);
    //string img_file2(argv[4]);

    Mat img1 = cv::imread("apple.jpg", CV_LOAD_IMAGE_COLOR);
    
    //Mat img2 = cv::imread(img_file2, CV_LOAD_IMAGE_COLOR);

    if (img1.channels() != 1) {
        cvtColor(img1, img1, cv::COLOR_RGB2GRAY);
    }

    CV::imshow("image",img1);

    vector<KeyPoint> kpts1;

    Mat desc1;
    Mat desc2;

    vector<DMatch> matches;
    Ptr<BRISK> brisk = BRISK::create();
    brisk->detectAndCompute(img, Mat(), kpts, desc1);
    
    cout<<desc1;
    //printf("%d",desc1);
    //detect_and_compute(desc_type, img1, kpts1, desc1);
    //detect_and_compute(desc_type, img2, kpts2, desc2);

    //match(match_type, desc1, desc2, matches);

    //vector<char> match_mask(matches.size(), 1);
    //findKeyPointsHomography(kpts1, kpts2, matches, match_mask);

    //Mat res;
    //c0v::drawMatches(img1, kpts1, img2, kpts2, matches, res, Scalar::all(-1),
        //Scalar::all(-1), match_mask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    //cv::imshow("result", res);
    //cv::waitKey(0);

    return 0;
}


