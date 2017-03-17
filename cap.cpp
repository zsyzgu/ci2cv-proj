#include <utils/helpers.hpp>
#include <utils/command-line-arguments.hpp>
#include <utils/points.hpp>
#include <tracker/FaceTracker.hpp>
#include <avatar/myAvatar.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "client.h"

using namespace FACETRACKER;
using namespace AVATAR;

struct Configuration {
  std::string model_pathname;
  std::string params_pathname;
  int tracking_threshold;
  std::string window_title;
  int circle_radius;
  int circle_thickness;
  int circle_linetype;
  int circle_shift;
};

int runImageMode(const Configuration &cfg, const cv::Mat& image, Avatar* avatar);
void displayData(const Configuration &cfg, const cv::Mat &image, const std::vector<cv::Point_<double> > &points, const Pose &pose);

int main(int argc, char** argv) {
  Client client;
  client.start();
  int n = 4000;
  char* data = new char[n];
  data[1234] = 0;
  while (true) {
    data[1234]++;
    client.sendByteArray(0, n, data);
    std::cout << "sent" << std::endl;
  }
  delete[] data;
  /*Configuration cfg;
  cfg.model_pathname = DefaultFaceTrackerModelPathname();
  cfg.params_pathname = DefaultFaceTrackerParamsPathname();
  cfg.tracking_threshold = 1;
  cfg.window_title = "CSIRO Face Fit";
  cfg.circle_radius = 2;
  cfg.circle_thickness = 2;
  cfg.circle_linetype = 8;
  cfg.circle_shift = 0;

  Avatar* avatar = LoadAvatar("zsyzgu.model");
  if (!avatar)
  throw make_runtime_error("Failed to load avatar.");
  avatar->setAvatar(0);
  cv::Mat_<cv::Vec<uint8_t,3> > calibration_image = cv::imread("zsyzgu.jpg");
  std::vector<cv::Point_<double> > calibration_points;
  std::ifstream uvFile("zsyzgu.uv");
  double u, v;
  while (uvFile >> u >> v) {
    calibration_points.push_back(cv::Point_<double>(u * calibration_image.cols, (1 - v) * calibration_image.rows));
  }
  avatar->Initialise(calibration_image, calibration_points);

  cv::VideoCapture capture;
  if (!capture.open(0)) {
    make_runtime_error("Can not open web camera");
    return 0;
  }

  for (;;) {
    cv::Mat frame;
    capture >> frame;
    if (frame.empty()) {
      break;
    }
    runImageMode(cfg, frame, avatar);
  }

  return 0;*/
}

int runImageMode(const Configuration &cfg, const cv::Mat& image, Avatar* avatar) {
  FaceTracker* tracker = LoadFaceTracker(cfg.model_pathname.c_str());
  FaceTrackerParams* trackerParams  = LoadFaceTrackerParams(cfg.params_pathname.c_str());

  cv::Mat inp;
  cv::cvtColor(image, inp, CV_RGB2GRAY);
  int result = tracker->NewFrame(inp, trackerParams);

  std::vector<cv::Point_<double> > shape;
  Pose pose;

  if (result >= cfg.tracking_threshold) {
    shape = tracker->getShape();
    pose = tracker->getPose();
    cv::Mat_<cv::Vec<uint8_t,3> > draw = image.clone();
    avatar->Animate(draw, image, shape);
    cv::imshow(cfg.window_title, draw);
    char ch = cv::waitKey(1);
  } else {
    //cv::imshow(cfg.window_title, image);
    displayData(cfg, image, shape, pose);
  }

  delete tracker;
  delete trackerParams;
  return 0;
}

cv::Mat computePoseImage(const Pose &pose, int height, int width) {
  cv::Mat_<cv::Vec<uint8_t,3> > rv = cv::Mat_<cv::Vec<uint8_t,3> >::zeros(height,width);
  cv::Mat_<double> axes = pose_axes(pose);
  cv::Mat_<double> scaling = cv::Mat_<double>::eye(3,3);

  for (int i = 0; i < axes.cols; i++) {
    axes(0,i) = -0.5*double(width)*(axes(0,i) - 1);
    axes(1,i) = -0.5*double(height)*(axes(1,i) - 1);
  }
  
  cv::Point centre(width/2, height/2);
  cv::line(rv, centre, cv::Point(axes(0,0), axes(1,0)), cv::Scalar(255,0,0));
  cv::line(rv, centre, cv::Point(axes(0,1), axes(1,1)), cv::Scalar(0,255,0));
  cv::line(rv, centre, cv::Point(axes(0,2), axes(1,2)), cv::Scalar(0,0,255));

  return rv;
}

void displayData(const Configuration &cfg, const cv::Mat &image, const std::vector<cv::Point_<double> > &points, const Pose &pose) {
  cv::Scalar colour;
  if (image.type() == cv::DataType<uint8_t>::type)
    colour = cv::Scalar(255);
  else if (image.type() == cv::DataType<cv::Vec<uint8_t,3> >::type)
    colour = cv::Scalar(0,0,255);
  else
    colour = cv::Scalar(255);

  cv::Mat displayed_image;
  if (image.type() == cv::DataType<cv::Vec<uint8_t,3> >::type)
    displayed_image = image.clone();
  else if (image.type() == cv::DataType<uint8_t>::type)
    cv::cvtColor(image, displayed_image, CV_GRAY2BGR);
  else 
    throw make_runtime_error("Unsupported camera image type for displayData function.");

  for (size_t i = 0; i < points.size(); i++) {
    cv::circle(displayed_image, points[i], cfg.circle_radius, colour, cfg.circle_thickness, cfg.circle_linetype, cfg.circle_shift);
  }

  int pose_image_height = 100;
  int pose_image_width = 100;
  cv::Mat pose_image = computePoseImage(pose, pose_image_height, pose_image_width);
  for (int i = 0; i < pose_image_height; i++) {
    for (int j = 0; j < pose_image_width; j++) {
      displayed_image.at<cv::Vec<uint8_t,3> >(displayed_image.rows - pose_image_height + i,
            displayed_image.cols - pose_image_width + j)
       
         = pose_image.at<cv::Vec<uint8_t,3> >(i,j);
    }
  }

  cv::imshow(cfg.window_title, displayed_image);
  char ch = cv::waitKey(1);
}
