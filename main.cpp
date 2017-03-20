#include <utils/helpers.hpp>
#include <utils/command-line-arguments.hpp>
#include <utils/points.hpp>
#include <tracker/FaceTracker.hpp>
#include <avatar/myAvatar.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "frame.h"

using namespace FACETRACKER;
using namespace AVATAR;

struct Configuration {
  std::string modelPathname;
  std::string paramsPathname;
  int trackingThreshold;
  double faceBoxSize;
  std::string window_title;
  int circle_radius;
  int circle_thickness;
  int circle_linetype;
  int circle_shift;
};

Frame* frame = NULL;

void printUsage();
void initialize();
void capture();

void produceModel(std::string modelPathName, cv::Mat& image, std::vector<cv::Point_<double> >& uv);
cv::Mat computePoseImage(const Pose &pose, int height, int width);
cv::Mat displayFeatures(const Configuration &cfg, const cv::Mat &image, const std::vector<cv::Point_<double> > &uv, const Pose &pose);
int calnFeatures(cv::Mat& image, cv::Mat& displayImage, std::vector<cv::Point_<double> >& uv, std::vector<cv::Point3_<double> >& vertices);

int main() {
  printUsage();

  while (true) {
    char ch;
    std::cin >> ch;
    if (ch == 'h') {
      printUsage();
    }
    if (ch == 'i') {
      initialize();
    }
    if (ch == 'c') {
      capture();
    }
    if (ch == 'e') {
      break;
    }
  }

  if (frame != NULL) {
    delete frame;
  }
  return 0;
}

void printUsage() {
  std::string usage = 
    "h -- help\n"
    "i -- initialize\n"
    "c -- capture\n"
    "e -- exit";

  std::cout << usage << std::endl;
}

void initialize() {
  cv::VideoCapture capture;
  if (!capture.open(0)) {
    perror("Can not open web camera");
    return;
  }

  char ch;
  cv::Mat image;
  std::vector<cv::Point_<double> > uv;
  std::vector<cv::Point3_<double> > vertices;

  while (true) {
    capture >> image;
    cv::Mat displayImage;
    int result = calnFeatures(image, displayImage, uv, vertices);
    //cv::imshow("capturing", displayImage);
    //cv::waitKey(1);
    std::cout << "result = " << result << ". Is it ok? [y/n]" << std::endl;
    std::cin >> ch;
    if (ch == 'y') {
      break;
    }
  }

  produceModel("Data/model", image, uv);

  std::vector<int> tris;
  std::ifstream fin("Data/face.tri");
  int x;
  while (fin >> x) {
    tris.push_back(x);
  }

  std::cout << "initialize done. enter any key to connect." << std::endl;
  std::cin >> ch;
  frame = new Frame();
  frame->start(image, uv, tris);
  std::cout << "connect done" << std::endl;
  printUsage();
}

void capture() {
  if (frame == NULL) {
    std::cout << "please first initialize." << std::endl;
    return;
  }

  Avatar* avatar = LoadAvatar("Data/model");
  if (!avatar) {
    throw make_runtime_error("Failed to load avatar.");
  }
  avatar->setAvatar(0);
  cv::Mat_<cv::Vec<uint8_t,3> > calibration_image = cv::imread("Pictures/model.jpg");
  std::vector<cv::Point_<double> > calibration_points;
  std::ifstream uvFile("Data/model.uv");
  double u, v;
  while (uvFile >> u >> v) {
    calibration_points.push_back(cv::Point_<double>(u * calibration_image.cols, (1 - v) * calibration_image.rows));
  }
  avatar->Initialise(calibration_image, calibration_points);

  cv::VideoCapture capture;
  if (!capture.open(0)) {
    perror("Can not open web camera");
    return;
  }

  cv::Mat image;
  std::vector<cv::Point_<double> > uv;
  std::vector<cv::Point3_<double> > vertices;

  while (true) {
    capture >> image;
    cv::Mat displayImage;
    int result = calnFeatures(image, displayImage, uv, vertices);
    //cv::imshow("capturing", displayImage);
    //cv::waitKey(1);
    if (result > 0) {
      frame->update(image, uv, vertices);
    }
    std::cout << "Hello" << std::endl;
  }
}

void produceModel(std::string modelPathName, cv::Mat& image, std::vector<cv::Point_<double> >& uv) {
  Avatar* abstarctModel = LoadAvatar();
  myAvatar* model = dynamic_cast<myAvatar*>(abstarctModel);
  assert(model);

  model->_scale.clear();
  model->_textr.clear();
  model->_images.clear();  
  model->_shapes.clear();
  model->_reg.clear();
  model->_expr.clear();
  model->_lpupil.clear();
  model->_rpupil.clear();

  cv::Mat_<double> saragih_points = vectorise_points(uv);
  cv::Mat_<double> eyes;
  model->AddAvatar(image, saragih_points, eyes);

  std::ofstream out(modelPathName.c_str(), std::ios::out | std::ios::binary);
  model->Write(out, true);
  out.close();

  delete model;
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

cv::Mat displayFeatures(const Configuration &cfg, const cv::Mat &image, const std::vector<cv::Point_<double> > &uv, const Pose &pose) {
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

  for (size_t i = 0; i < uv.size(); i++) {
    cv::circle(displayed_image, uv[i], cfg.circle_radius, colour, cfg.circle_thickness, cfg.circle_linetype, cfg.circle_shift);
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

  return displayed_image;
}

int calnFeatures(cv::Mat& image, cv::Mat& displayImage, std::vector<cv::Point_<double> >& uv, std::vector<cv::Point3_<double> >& vertices) {
  Configuration cfg;
  cfg.modelPathname = DefaultFaceTrackerModelPathname();
  cfg.paramsPathname = DefaultFaceTrackerParamsPathname();
  cfg.trackingThreshold = 1;
  cfg.faceBoxSize = 40;
  cfg.window_title = "UV image";
  cfg.circle_radius = 2;
  cfg.circle_thickness = 2;
  cfg.circle_linetype = 8;
  cfg.circle_shift = 0;

  FaceTracker* tracker = LoadFaceTracker(cfg.modelPathname.c_str());
  FaceTrackerParams* trackerParams = LoadFaceTrackerParams(cfg.paramsPathname.c_str());

  cv::Mat grayImage;
  cv::cvtColor(image, grayImage, CV_RGB2GRAY);
  int result = tracker->NewFrame(grayImage, trackerParams);

  uv = tracker->getShape();
  Pose pose = tracker->getPose();
  //displayImage = displayFeatures(cfg, image, uv, pose);

  for (int i = 0; i < uv.size(); i++) {
    uv[i].x = uv[i].x / image.cols;
    uv[i].y = 1 - uv[i].y / image.rows;
  }

  vertices = tracker->get3DShape();
  for (int i = 0; i < vertices.size(); i++) {
    vertices[i].x = vertices[i].x / cfg.faceBoxSize;
    vertices[i].y = vertices[i].y / cfg.faceBoxSize;
    vertices[i].z = vertices[i].z / cfg.faceBoxSize;
  }

  delete tracker;
  delete trackerParams;

  return result;
}
