#ifndef FRAME_H
#define FRAME_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "client.h"

class Frame {
private:
  Client client;
  int strategy;
  bool strategyChanged;
  std::vector<int> tris;
  cv::Mat modelImage;
  cv::Mat faceImage;
  cv::Mat leftEyeImage;
  cv::Mat rightEyeImage;
  cv::Mat mouthImage;
  std::vector<cv::Point_<double> > modelUV;
  std::vector<cv::Point_<double> > faceUV;
  std::vector<cv::Point_<double> > leftEyeUV;
  std::vector<cv::Point_<double> > rightEyeUV;
  std::vector<cv::Point_<double> > mouthUV;
  std::vector<cv::Point3_<double> > vertices;
  void cutImage(cv::Mat& input, cv::Mat& output, cv::Rect_<double> rect);
  void cutUV(std::vector<cv::Point_<double> >& input, std::vector<cv::Point_<double> >& output, cv::Rect_<double> rect);
  void readTris();
  cv::Rect_<double> getRegionRect(int trisBeginId, int trisNumber);
  /*void saveUV(std::string fileName, std::vector<cv::Point_<double> >& uv);
  void saveVertices(std::string fileName, std::vector<cv::Point3_<double> >& vertices);
  void saveTris(std::string fileName, std::vector<int>& tris);*/
public:
  Frame();
  ~Frame();
  void setStrategy(int strategy);
  cv::Rect_<double> getLeftEyeRect();
  cv::Rect_<double> getRightEyeRect();
  cv::Rect_<double> getMouthRect();
  void cutLeftEyeRegion();
  void cutRightEyeRegion();
  void cutMouthRegion();
  void start(cv::Mat modelImage, std::vector<cv::Point_<double> > modelUV, std::vector<int> tris);
  void update(cv::Mat faceImage, std::vector<cv::Point_<double> > faceUV, std::vector<cv::Point3_<double> > vertices);
};

#endif //FRAME_H
