#include <utils/helpers.hpp>
#include <utils/command-line-arguments.hpp>
#include <utils/points.hpp>
#include <tracker/FaceTracker.hpp>
#include <avatar/myAvatar.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "frame.h"

using namespace std;
using namespace FACETRACKER;
using namespace AVATAR;

Frame* frame = NULL;

void printUsage();
void initialize(cv::Mat& image);
void capture();
cv::Mat chooseImage();

void produceModel(std::string modelPathName, cv::Mat& image, std::vector<cv::Point_<double> >& uv);
cv::Mat displayFeatures(const cv::Mat &image, const std::vector<cv::Point_<double> > &uv);
int calnFeatures(cv::Mat& image, std::vector<cv::Point_<double> >& uv, std::vector<cv::Point3_<double> >& vertices);

int main() {
  printUsage();

  while (true) {
    char ch;
    ch = getchar();
    if (ch == 'h') {
      printUsage();
    }
    if (ch == 'i') {
      cv::Mat image = chooseImage();
      initialize(image);
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

void initialize(cv::Mat& image) {
  std::vector<cv::Point_<double> > uv;
  std::vector<cv::Point3_<double> > vertices;
  calnFeatures(image, uv, vertices);

  produceModel("Data/model", image, uv);

  std::vector<int> tris;
  std::ifstream fin("Data/face.tri");
  int x;
  while (fin >> x) {
    tris.push_back(x);
  }

  std::cout << "initialize done." << std::endl;
  //frame = new Frame();
  //frame->start(image, uv, vertices, tris);
  std::cout << "connect done" << std::endl;
  printUsage();
}

void capture() {
  if (frame == NULL) {
    std::cout << "use old initialized image" << std::endl;
    cv::Mat image = cv::imread("Pictures/model.jpg");
    initialize(image);
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
    pyrDown(image, image, cv::Size(image.cols / 2, image.rows / 2));
    int result = calnFeatures(image, uv, vertices);
    if (result > 0) {
      //frame->update(image, uv, vertices);
    }
    cv::Mat displayImage = displayFeatures(image, uv);
    cv::imshow("window", displayImage);
    cv::waitKey(1);
  }
}

cv::Mat chooseImage() {
  cv::VideoCapture capture;
  if (!capture.open(0)) {
    perror("Can not open web camera");
  }

  cv::Mat image;
  std::vector<cv::Point_<double> > uv;
  std::vector<cv::Point3_<double> > vertices;

  while (true) {
    capture >> image;
    pyrDown(image, image, cv::Size(image.cols / 2, image.rows / 2));
    int result = calnFeatures(image, uv, vertices);

    cv::Mat displayImage = displayFeatures(image, uv);
    cv::imshow("window", displayImage);
    std::cout << "result = " << result << ". Is it ok? [y/n]" << std::endl;
    char ch = cv::waitKey(0);
    if (ch == 'y') {
      break;
    }
  }

  cv::imwrite("Pictures/model.jpg", image);
  return image;
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

cv::Mat displayFeatures(const cv::Mat &image, const std::vector<cv::Point_<double> > &uv) {
  int circle_radius = 1;
  int circle_thickness = 2;
  int circle_linetype = 8;
  int circle_shift = 0;

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
    cv::Point_<double> point = uv[i];
    point.x = displayed_image.cols * point.x;
    point.y = displayed_image.rows * (1 - point.y);
    cv::circle(displayed_image, point, circle_radius, colour, circle_thickness, circle_linetype, circle_shift);
  }

  return displayed_image;
}

int calnFeatures(cv::Mat& image, std::vector<cv::Point_<double> >& uv, std::vector<cv::Point3_<double> >& vertices) {
  double mToCM = 100;

  FaceTracker* tracker = LoadFaceTracker(DefaultFaceTrackerModelPathname().c_str());
  FaceTrackerParams* trackerParams = LoadFaceTrackerParams(DefaultFaceTrackerParamsPathname().c_str());

  cv::Mat grayImage;
  cv::cvtColor(image, grayImage, CV_RGB2GRAY);
  int result = tracker->NewFrame(grayImage, trackerParams);

  uv = tracker->getShape();

  for (int i = 0; i < uv.size(); i++) {
    uv[i].x = uv[i].x / image.cols;
    uv[i].y = 1 - uv[i].y / image.rows;
  }

  vertices = tracker->get3DShape();
  for (int i = 0; i < vertices.size(); i++) {
    vertices[i].x = vertices[i].x / mToCM;
    vertices[i].y = vertices[i].y / mToCM;
    vertices[i].z = vertices[i].z / mToCM;
  }

  delete tracker;
  delete trackerParams;

  return result;
}
