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
//#include <opencv2/xfeatures2d.hpp>
//#include <opencv2/xfeatures2d/nonfree.hpp>
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

//using cv::xfeatures2d::BriefDescriptorExtractor;
/*using cv::xfeatures2d::SURF;
using cv::xfeatures2d::SIFT;
using cv::xfeatures2d::DAISY;
using cv::xfeatures2d::FREAK;
*/
const double kDistanceCoef = 4.0;
const int kMaxMatchingSize = 50;

inline void detect_and_compute(Mat& img, vector<KeyPoint>& kpts, Mat& desc) {
    
   
        Ptr<BRISK> brisk = BRISK::create();
        brisk->detectAndCompute(img, Mat(), kpts, desc);
    
    
}

inline void match( Mat& desc1, Mat& desc2, vector<DMatch>& matches) {
    matches.clear();
   
        BFMatcher desc_matcher(cv::NORM_L2, true);
        vector< vector<DMatch> > vmatches;
        desc_matcher.knnMatch(desc1, desc2, vmatches, 1);
        for (int i = 0; i < static_cast<int>(vmatches.size()); ++i) {
            if (!vmatches[i].size()) {
                continue;
            }
            matches.push_back(vmatches[i][0]);
        }
    
    std::sort(matches.begin(), matches.end());
    while (matches.front().distance * kDistanceCoef < matches.back().distance) {
        matches.pop_back();
    }
    while (matches.size() > kMaxMatchingSize) {
        matches.pop_back();
    }
}

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
    //   - path to the scene image file
    //
   
    


    Mat img1 = cv::imread("bkg.png", CV_LOAD_IMAGE_COLOR);
    Mat img2 = cv::imread("bkg2.png", CV_LOAD_IMAGE_COLOR);
    mat img3 =cv::imread("pdp.jpg",CV_LOAD_IMAGE_COLOR);
    //cv::resize(img1,img1, (960, 540));
    //img2=cv::resize(img2);
    resize(img1, img1, cv::Size(img1.cols/2, img1.rows/2));
    resize(img2, img2, cv::Size(img2.cols/2, img2.rows/2));
    resize(img3, img3, cv::Size(img3.cols/2, img3.rows/2));
    if (img1.channels() != 1) {
        cvtColor(img1, img1, cv::COLOR_RGB2GRAY);
    }

    if (img2.channels() != 1) {
        cvtColor(img2, img2, cv::COLOR_RGB2GRAY);
    }
    
    if (img3.channels() != 1) {
        cvtColor(img3, img3, cv::COLOR_RGB2GRAY);
    }


    vector<KeyPoint> kpts1;
    vector<KeyPoint> kpts2;
    vector<KeyPoint> kpts3;

    Mat desc1;
    Mat desc2;
    Mat desc3;

    vector<DMatch> matches;

    detect_and_compute( img1, kpts1, desc1);
    detect_and_compute( img2, kpts2, desc2);

    match(desc1, desc2, matches);
    for(auto m:matches)
        cout<<m;
    vector<char> match_mask(matches.size(), 1);
    findKeyPointsHomography(kpts1, kpts2, matches, match_mask);

    Mat res;
    cv::drawMatches(img1, kpts1, img2, kpts2, matches, res, Scalar::all(-1),
        Scalar::all(-1), match_mask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imshow("result", res);
    cv::waitKey(0);

    return 0;
}