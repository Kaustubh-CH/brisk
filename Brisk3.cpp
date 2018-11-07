//#include "precomp.hpp"
#include <fstream>
#include <stdlib.h>

//#include "agast_score.hpp"
#include <opencv2/features2d.hpp>

#include <vector>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

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
using cv::makePtr;
using cv::InputArray;
using cv::OutputArray;
//using cv::_OVERRIDE;
class BRISK_Impl1:public BRISK
{
public:

    explicit BRISK_Impl1(int thresh=30, int octaves=3, float patternScale=1.0f);

    //virtual ~BRISK_Impl1();

 void generateKernel1(const std::vector<float> &radiusList,
        const std::vector<int> &numberList, float dMax=5.85f, float dMin=8.2f,
        const std::vector<int> &indexChange=std::vector<int>());

 void detectAndCompute( InputArray image, InputArray mask,
                     CV_OUT std::vector<KeyPoint>& keypoints,
                     OutputArray descriptors,
                     bool useProvidedKeypoints ) ;//CV_OVERRIDE;

protected:

    void computeKeypointsNoOrientation(InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints) const;
    void computeDescriptorsAndOrOrientation(InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints,
                                       OutputArray descriptors, bool doDescriptors, bool doOrientation,
                                       bool useProvidedKeypoints) const;
    CV_PROP_RW int threshold;
    CV_PROP_RW int octaves;

     struct BriskPatternPoint{
        float x;         // x coordinate relative to center
        float y;         // x coordinate relative to center
        float sigma;     // Gaussian smoothing sigma
    };
    struct BriskShortPair{
        unsigned int i;  // index of the first pattern point
        unsigned int j;  // index of other pattern point
    };
    struct BriskLongPair{
        unsigned int i;  // index of the first pattern point
        unsigned int j;  // index of other pattern point
        int weighted_dx; // 1024.0/dx
        int weighted_dy; // 1024.0/dy
    };

    inline int smoothedIntensity(const cv::Mat& image,
                const cv::Mat& integral,const float key_x,
                const float key_y, const unsigned int scale,
                const unsigned int rot, const unsigned int point) const;


    BriskPatternPoint* patternPoints_;     //[i][rotation][scale]
    unsigned int points_;                 // total number of collocation points
    float* scaleList_;                     // lists the scaling per scale index [scale]
    unsigned int* sizeList_;             // lists the total pattern size per scale index [scale]
    static const unsigned int scales_;    // scales discretization
    static const float scalerange_;     // span of sizes 40->4 Octaves - else, this needs to be adjusted...
    static const unsigned int n_rot_;    // discretization of the rotation look-up

    // pairs
    int strings_;                        // number of uchars the descriptor consists of
    float dMax_;                         // short pair maximum distance
    float dMin_;                         // long pair maximum distance
    BriskShortPair* shortPairs_;         // d<_dMax
    BriskLongPair* longPairs_;             // d>_dMin
    unsigned int noShortPairs_;         // number of shortParis
    unsigned int noLongPairs_;             // number of longParis

    // general
    static const float basicSize_;



private:
  //  BRISK_Impl1(const BRISK_Impl1 &); // copy disabled
    //BRISK_Impl1& operator=(const BRISK_Impl1 &); // assign disabled


  };              

//class BriskScaleSpace1
//{
//public:
  // construct telling the octaves number:
  //BriskScaleSpace1(int _octaves = 3);
  //~BriskScaleSpace1();

 /* // construct the image pyramids
  void
  constructPyramid(const cv::Mat& image);

  // get Keypoints
  void
  getKeypoints(const int _threshold, std::vector<cv::KeyPoint>& keypoints);

protected:
  // nonmax suppression:
  inline bool
  isMax2D(const int layer, const int x_layer, const int y_layer)
  {

  }
*/

 /*// the image pyramids:
  int layers_;
  //std::vector<BriskLayer> pyramid_;

  // some constant parameters:
  static const float safetyFactor_;
  static const float basicSize_;*/
//);
    



const float BRISK_Impl1::basicSize_ = 12.0f;
const unsigned int BRISK_Impl1::scales_ = 64;
const float BRISK_Impl1::scalerange_ = 30.f; // 40->4 Octaves - else, this needs to be adjusted...
const unsigned int BRISK_Impl1::n_rot_ = 1024; // discretization of the rotation look-up


BRISK_Impl1::BRISK_Impl1(int thresh, int octaves_in, float patternScale)
{
     threshold = thresh;
  octaves = octaves_in;

  std::vector<float> rList;
  std::vector<int> nList;

  // this is the standard pattern found to be suitable also
  rList.resize(5);
  nList.resize(5);
  const double f = 0.85 * patternScale;

  rList[0] = (float)(f * 0.);
  rList[1] = (float)(f * 2.9);
  rList[2] = (float)(f * 4.9);
  rList[3] = (float)(f * 7.4);
  rList[4] = (float)(f * 10.8);

  nList[0] = 1;
  nList[1] = 10;
  nList[2] = 14;
  nList[3] = 15;
  nList[4] = 20;

  generateKernel1(rList, nList, (float)(5.85 * patternScale), (float)(8.2 * patternScale));
}

void BRISK_Impl1::generateKernel1(const std::vector<float> &radiusList,
                           const std::vector<int> &numberList,
                           float dMax, float dMin,
                           const std::vector<int>& _indexChange)
{
    std::vector<int> indexChange = _indexChange;
  dMax_ = dMax;
  dMin_ = dMin;

  // get the total number of points
  const int rings = (int)radiusList.size();
  CV_Assert(radiusList.size() != 0 && radiusList.size() == numberList.size());
  points_ = 0; // remember the total number of points
  for (int ring = 0; ring < rings; ring++)
  {
    points_ += numberList[ring];                                                        //CAN USE REDUCE OPENMP
  }

  patternPoints_ = new BriskPatternPoint[points_ * scales_ * n_rot_];
  BriskPatternPoint* patternIterator = patternPoints_;               //A struct

  static const float lb_scale = (float)(std::log(scalerange_) / std::log(2.0));
  static const float lb_scale_step = lb_scale / (scales_);    

  scaleList_ = new float[scales_];
  sizeList_ = new unsigned int[scales_];

   const float sigma_scale = 1.3f;

   for (unsigned int scale = 0; scale < scales_; ++scale)
    {
            scaleList_[scale] = (float)std::pow((double) 2.0, (double) (scale * lb_scale_step));
            sizeList_[scale] = 0;
            // generate the pattern points look-up
            double alpha, theta;
            for (size_t rot = 0; rot < n_rot_; ++rot)
            {
              theta = double(rot) * 2 * CV_PI / double(n_rot_); // this is the rotation of the feature
              for (int ring = 0; ring < rings; ++ring)
              {
                for (int num = 0; num < numberList[ring]; ++num)
                {
                  // the actual coordinates on the circle
                  alpha = (double(num)) * 2 * CV_PI / double(numberList[ring]);
                  patternIterator->x = (float)(scaleList_[scale] * radiusList[ring] * cos(alpha + theta)); // feature rotation plus angle of the point
                  patternIterator->y = (float)(scaleList_[scale] * radiusList[ring] * sin(alpha + theta));
                  // and the gaussian kernel sigma
                  if (ring == 0)
                  {
                    patternIterator->sigma = sigma_scale * scaleList_[scale] * 0.5f;
                  }
                  else
                  {
                    patternIterator->sigma = (float)(sigma_scale * scaleList_[scale] * (double(radiusList[ring]))
                                             * sin(CV_PI / numberList[ring]));
                  }
                  // adapt the sizeList if necessary
                  const unsigned int size = cvCeil(((scaleList_[scale] * radiusList[ring]) + patternIterator->sigma)) + 1;
                  if (sizeList_[scale] < size)
                  {
                    sizeList_[scale] = size;
                  }

                  // increment the iterator
                  ++patternIterator;
                }
              }
            }
    }

  // now also generate pairings
  shortPairs_ = new BriskShortPair[points_ * (points_ - 1) / 2];
  longPairs_ = new BriskLongPair[points_ * (points_ - 1) / 2];
  noShortPairs_ = 0;
  noLongPairs_ = 0;

    unsigned int indSize = (unsigned int)indexChange.size();
    if (indSize == 0)
      {
        indexChange.resize(points_ * (points_ - 1) / 2);
        indSize = (unsigned int)indexChange.size();

        for (unsigned int i = 0; i < indSize; i++)
          indexChange[i] = i;
      }

 const float dMin_sq = dMin_ * dMin_;
 const float dMax_sq = dMax_ * dMax_;


 for (unsigned int i = 1; i < points_; i++)
  {
    for (unsigned int j = 0; j < i; j++)
    { //(find all the pairs)
      // point pair distance:
      const float dx = patternPoints_[j].x - patternPoints_[i].x;
      const float dy = patternPoints_[j].y - patternPoints_[i].y;
      const float norm_sq = (dx * dx + dy * dy);
      cout<<"NOrm Sq "<<norm_sq<<" dx "<<dx<<" Dy "<<dy<<"\n";
      if (norm_sq > dMin_sq)
      {
        // save to long pairs
        BriskLongPair& longPair = longPairs_[noLongPairs_];
        longPair.weighted_dx = int((dx / (norm_sq)) * 2048.0 + 0.5);
        longPair.weighted_dy = int((dy / (norm_sq)) * 2048.0 + 0.5);
        longPair.i = i;
        longPair.j = j;
        ++noLongPairs_;
      }

else if (norm_sq < dMax_sq)
      {
        // save to short pairs
        CV_Assert(noShortPairs_ < indSize);
        // make sure the user passes something sensible
        BriskShortPair& shortPair = shortPairs_[indexChange[noShortPairs_]];
        shortPair.j = j;
        shortPair.i = i;
        ++noShortPairs_;
      }
    }
  }
cout<<"No of long "<<noLongPairs_<<"No of short"<<noShortPairs_;
  // no bits:
  strings_ = (int) ceil((float(noShortPairs_)) / 128.0) * 4 * 4;


}

void
BRISK_Impl1::detectAndCompute( InputArray _image, InputArray _mask, std::vector<KeyPoint>& keypoints,
                              OutputArray _descriptors, bool useProvidedKeypoints)
{
  bool doOrientation=true;

  // If the user specified cv::noArray(), this will yield false. Otherwise it will return true.
  bool doDescriptors = _descriptors.needed();

  computeDescriptorsAndOrOrientation(_image, _mask, keypoints, _descriptors, doDescriptors, doOrientation,
                                       useProvidedKeypoints);
}
inline bool
RoiPredicate(const float minX, const float minY, const float maxX, const float maxY, const KeyPoint& keyPt)
{
  const Point2f& pt = keyPt.pt;
  return (pt.x < minX) || (pt.x >= maxX) || (pt.y < minY) || (pt.y >= maxY);
}

void
BRISK_Impl1::computeDescriptorsAndOrOrientation(InputArray _image, InputArray _mask, std::vector<KeyPoint>& keypoints,
                                     OutputArray _descriptors, bool doDescriptors, bool doOrientation,
                                     bool useProvidedKeypoints) const
{
  Mat image = _image.getMat(), mask = _mask.getMat();
  if( image.type() != CV_8UC1 )
      cvtColor(image, image, cv::COLOR_BGR2GRAY);

  if (!useProvidedKeypoints)
   {
     doOrientation = true;
     computeKeypointsNoOrientation(_image, _mask, keypoints);
   }

   //Remove keypoints very close to the border
  size_t ksize = keypoints.size();
  std::vector<int> kscales; // remember the scale per keypoint
  kscales.resize(ksize);
  static const float log2 = 0.693147180559945f;
  static const float lb_scalerange = (float)(std::log(scalerange_) / (log2));
  std::vector<cv::KeyPoint>::iterator beginning = keypoints.begin();
  std::vector<int>::iterator beginningkscales = kscales.begin();
  static const float basicSize06 = basicSize_ * 0.6f;


   for (size_t k = 0; k < ksize; k++)
  {
    unsigned int scale;
      scale = std::max((int) (scales_ / lb_scalerange * (std::log(keypoints[k].size / (basicSize06)) / log2) + 0.5), 0);
      // saturate
      if (scale >= scales_)
        scale = scales_ - 1;
      kscales[k] = scale;

    const int border = sizeList_[scale];
    const int border_x = image.cols - border;
    const int border_y = image.rows - border;

    if (RoiPredicate((float)border, (float)border, (float)border_x, (float)border_y, keypoints[k]))
    {

      keypoints.erase(beginning + k);
      kscales.erase(beginningkscales + k);
      if (k == 0)
      {
        beginning = keypoints.begin();
        beginningkscales = kscales.begin();
      }
      ksize--;
      k--;
    }
  }      

// first, calculate the integral image over the whole image:
  // current integral image
  cv::Mat _integral; // the integral image
  cv::integral(image, _integral);

  int* _values = new int[points_]; // for temporary use

  // resize the descriptors:
  cv::Mat descriptors;
  if (doDescriptors)
  {
    _descriptors.create((int)ksize, strings_, CV_8U);
    descriptors = _descriptors.getMat();
    descriptors.setTo(0);
  }

 // now do the extraction for all keypoints:

  // temporary variables containing gray values at sample points:
  int t1;
  int t2;

  // the feature orientation
    const uchar* ptr = descriptors.ptr();
  for (size_t k = 0; k < ksize; k++)
  {
    cv::KeyPoint& kp = keypoints[k];
    const int& scale = kscales[k];
    const float& x = kp.pt.x;
    const float& y = kp.pt.y;

    if (doOrientation)
    {
        // get the gray values in the unrotated pattern
        for (unsigned int i = 0; i < points_; i++)
        {
            _values[i] = smoothedIntensity(image, _integral, x, y, scale, 0, i);
        }
        int direction0=0;
        int direction1=0;

         // now iterate through the long pairings
        const BriskLongPair* max = longPairs_ + noLongPairs_;
        for (BriskLongPair* iter = longPairs_; iter < max; ++iter)
        {
            CV_Assert(iter->i < points_ && iter->j < points_);
          t1 = *(_values + iter->i);
          t2 = *(_values + iter->j);

          const int delta_t = (t1 - t2);
          // update the direction:
          const int tmp0 = delta_t * (iter->weighted_dx) / 1024;                            //parlalize
          const int tmp1 = delta_t * (iter->weighted_dy) / 1024;
          direction0 += tmp0;
          direction1 += tmp1;
        }        kp.angle = (float)(atan2((float) direction1, (float) direction0) / CV_PI * 180.0);

        if (!doDescriptors)
        {
          if (kp.angle < 0)
            kp.angle += 360.f;
        }
    }

    if (!doDescriptors)
      continue;    
     
     int theta;
    if (kp.angle==-1)
    {
         // don't compute the gradient direction, just assign a rotation of 0
        theta = 0;
    }
    else
    {
         theta = (int) (n_rot_ * (kp.angle / (360.0)) + 0.5);
        if (theta < 0)
          theta += n_rot_;
        if (theta >= int(n_rot_))
          theta -= n_rot_;
    }

    if(kp.angle<0)
        kp.angle+=360.f;
    // now also extract the stuff for the actual direction:
    // let us compute the smoothed values
    int shifter = 0;
    //unsigned int mean=0;
    // get the gray values in the rotated pattern

    for (unsigned int i = 0; i < points_; i++)
    {
        _values[i] = smoothedIntensity(image, _integral, x, y, scale, theta, i);
    }


    // now iterate through all the short pairings
    unsigned int* ptr2 = (unsigned int*) ptr;

    const BriskShortPair* max = shortPairs_ + noShortPairs_;
    for (BriskShortPair* iter = shortPairs_; iter < max; ++iter)
    {
        CV_Assert(iter->i < points_ && iter->j < points_);
      t1 = *(_values + iter->i);
      t2 = *(_values + iter->j);
      if (t1 > t2)
      {
        *ptr2 |= ((1) << shifter);

      } // else already initialized with zero
      // take care of the iterators:
       ++shifter;
      if (shifter == 32)
      {
        shifter = 0;
        ++ptr2;
      }

    }

ptr+=strings_;
}

 // clean-up
  delete[] _values;
}

void
BRISK_Impl1::computeKeypointsNoOrientation(InputArray _image, InputArray _mask, std::vector<KeyPoint>& keypoints) const
{
  Mat image = _image.getMat(), mask = _mask.getMat();
  if( image.type() != CV_8UC1 )
      cvtColor(_image, image, cv::COLOR_BGR2GRAY);
 /* BriskScaleSpace1 BriskScaleSpace1(octaves);
  BriskScaleSpace1.constructPyramid(image);
  BriskScaleSpace1.getKeypoints(threshold, keypoints);   
*/
  // remove invalid points
  cv::KeyPointsFilter::runByPixelsMask(keypoints, mask);
}


inline int
BRISK_Impl1::smoothedIntensity(const cv::Mat& image, const cv::Mat& integral, const float key_x,
                                            const float key_y, const unsigned int scale, const unsigned int rot,
                                            const unsigned int point) const
{
     // get the float position
  const BriskPatternPoint& briskPoint = patternPoints_[scale * n_rot_ * points_ + rot * points_ + point];
  const float xf = briskPoint.x + key_x;
  const float yf = briskPoint.y + key_y;
  const int x = int(xf);
  const int y = int(yf);
  const int& imagecols = image.cols;


  // get the sigma:
  const float sigma_half = briskPoint.sigma;
  const float area = 4.0f * sigma_half * sigma_half;

  // calculate output:
  int ret_val;
  if (sigma_half < 0.5)
  {    
    //interpolation multipliers:
    const int r_x = (int)((xf - x) * 1024);
    const int r_y = (int)((yf - y) * 1024);
    const int r_x_1 = (1024 - r_x);
    const int r_y_1 = (1024 - r_y);
    const uchar* ptr = &image.at<uchar>(y, x);
    size_t step = image.step;
    // just interpolate:
    ret_val = r_x_1 * r_y_1 * ptr[0] + r_x * r_y_1 * ptr[1] +
              r_x * r_y * ptr[step] + r_x_1 * r_y * ptr[step+1];
    return (ret_val + 512) / 1024;
  }

  // this is the standard case (simple, not speed optimized yet):

  // scaling:
  const int scaling = (int)(4194304.0 / area);
  const int scaling2 = int(float(scaling) * area / 1024.0);
  CV_Assert(scaling2 != 0);

  // the integral image is larger:
  const int integralcols = imagecols + 1;

  // calculate borders
  const float x_1 = xf - sigma_half;
  const float x1 = xf + sigma_half;
  const float y_1 = yf - sigma_half;
  const float y1 = yf + sigma_half;

  const int x_left = int(x_1 + 0.5);
  const int y_top = int(y_1 + 0.5);
  const int x_right = int(x1 + 0.5);
  const int y_bottom = int(y1 + 0.5);

  // overlap area - multiplication factors:
  const float r_x_1 = float(x_left) - x_1 + 0.5f;
  const float r_y_1 = float(y_top) - y_1 + 0.5f;
  const float r_x1 = x1 - float(x_right) + 0.5f;
  const float r_y1 = y1 - float(y_bottom) + 0.5f;
  const int dx = x_right - x_left - 1;
  const int dy = y_bottom - y_top - 1;
  const int A = (int)((r_x_1 * r_y_1) * scaling);
  const int B = (int)((r_x1 * r_y_1) * scaling);
  const int C = (int)((r_x1 * r_y1) * scaling);
  const int D = (int)((r_x_1 * r_y1) * scaling);
  const int r_x_1_i = (int)(r_x_1 * scaling);
  const int r_y_1_i = (int)(r_y_1 * scaling);
  const int r_x1_i = (int)(r_x1 * scaling);
  const int r_y1_i = (int)(r_y1 * scaling);

  if (dx + dy > 2)
  {
    // now the calculation:
    const uchar* ptr = image.ptr() + x_left + imagecols * y_top;
    // first the corners:
    ret_val = A * int(*ptr);
    ptr += dx + 1;
    ret_val += B * int(*ptr);
    ptr += dy * imagecols + 1;
    ret_val += C * int(*ptr);
    ptr -= dx + 1;
    ret_val += D * int(*ptr);

    // next the edges:
    const int* ptr_integral = integral.ptr<int>() + x_left + integralcols * y_top + 1;
    // find a simple path through the different surface corners
    const int tmp1 = (*ptr_integral);
    ptr_integral += dx;
    const int tmp2 = (*ptr_integral);
    ptr_integral += integralcols;
    const int tmp3 = (*ptr_integral);
    ptr_integral++;
    const int tmp4 = (*ptr_integral);
    ptr_integral += dy * integralcols;
    const int tmp5 = (*ptr_integral);
    ptr_integral--;
    const int tmp6 = (*ptr_integral);
    ptr_integral += integralcols;
    const int tmp7 = (*ptr_integral);
    ptr_integral -= dx;
    const int tmp8 = (*ptr_integral);
    ptr_integral -= integralcols;
    const int tmp9 = (*ptr_integral);
    ptr_integral--;
    const int tmp10 = (*ptr_integral);
    ptr_integral -= dy * integralcols;
    const int tmp11 = (*ptr_integral);
    ptr_integral++;
    const int tmp12 = (*ptr_integral);

    // assign the weighted surface integrals:
    const int upper = (tmp3 - tmp2 + tmp1 - tmp12) * r_y_1_i;
    const int middle = (tmp6 - tmp3 + tmp12 - tmp9) * scaling;
    const int left = (tmp9 - tmp12 + tmp11 - tmp10) * r_x_1_i;
    const int right = (tmp5 - tmp4 + tmp3 - tmp6) * r_x1_i;
    const int bottom = (tmp7 - tmp6 + tmp9 - tmp8) * r_y1_i;

    return (ret_val + upper + middle + left + right + bottom + scaling2 / 2) / scaling2;
  }

  // now the calculation:
  const uchar* ptr = image.ptr() + x_left + imagecols * y_top;
  // first row:
  ret_val = A * int(*ptr);
  ptr++;
  const uchar* end1 = ptr + dx;
  for (; ptr < end1; ptr++)
  {
    ret_val += r_y_1_i * int(*ptr);
  }
  ret_val += B * int(*ptr);
  // middle ones:
  ptr += imagecols - dx - 1;
  const uchar* end_j = ptr + dy * imagecols;
  for (; ptr < end_j; ptr += imagecols - dx - 1)
  {
    ret_val += r_x_1_i * int(*ptr);
    ptr++;
    const uchar* end2 = ptr + dx;
    for (; ptr < end2; ptr++)
    {
      ret_val += int(*ptr) * scaling;
    }
    ret_val += r_x1_i * int(*ptr);
  }
  // last row:
  ret_val += D * int(*ptr);
  ptr++;
  const uchar* end3 = ptr + dx;
  for (; ptr < end3; ptr++)
  {
    ret_val += r_y1_i * int(*ptr);
  }
  ret_val += C * int(*ptr);

  return (ret_val + scaling2 / 2) / scaling2;
}




/*

BriskScaleSpace1::BriskScaleSpace1(int _octaves)
{
  if (_octaves == 0)
    layers_ = 1;
  else
    layers_ = 2 * _octaves;
}
BriskScaleSpace1::~BriskScaleSpace1()
{

}

// construct the image pyramids
//void
BriskScaleSpace1::constructPyramid(const cv::Mat& image)
{

  // set correct size:
  pyramid_.clear();

   // fill the pyramid:
  pyramid_.push_back(BriskLayer(image.clone()));
  if (layers_ > 1)
  {
    pyramid_.push_back(BriskLayer(pyramid_.back(), BriskLayer::CommonParams::TWOTHIRDSAMPLE));
  }
  const int octaves2 = layers_;

  for (uchar i = 2; i < octaves2; i += 2)
  {
    pyramid_.push_back(BriskLayer(pyramid_[i - 2], BriskLayer::CommonParams::HALFSAMPLE));
    pyramid_.push_back(BriskLayer(pyramid_[i - 1], BriskLayer::CommonParams::HALFSAMPLE));
  }
}

 void
BriskScaleSpace1::getKeypoints(const int threshold_, std::vector<cv::KeyPoint>& keypoints)
{
 // make sure keypoints is empty
  /*keypoints.resize(0);
  keypoints.reserve(2000);

  // assign thresholds
  int safeThreshold_ = (int)(threshold_ * safetyFactor_);
  std::vector<std::vector<cv::KeyPoint> > agastPoints;
  agastPoints.resize(layers_);*/


//}
/*
const float BriskScaleSpace1::safetyFactor_ = 1.0f;
const float BriskScaleSpace1::basicSize_ = 12.0f;
*/

int main(int argc, char** argv) {

    Mat img = cv::imread("pdp.jpg", CV_LOAD_IMAGE_COLOR);
    if (img.channels() != 1) {
        cvtColor(img, img, cv::COLOR_RGB2GRAY);
    }

    cv::imshow("image",img);

    vector<KeyPoint> kpts;

    Mat desc1;
    Mat desc2;

  BRISK_Impl1 a=BRISK_Impl1(10,21,2.3);

//    Ptr<BRISK> brisk = cv::makePtr<BRISK_Impl1>(10,1, 0.2);
    //brisk->detectAndCompute(img, Mat(), kpts, desc1);
    
    cout<<desc1;
  /*  for(auto a:kpts)
  */      
//cv::imshow("Res",desc1);
    return 0;
}
