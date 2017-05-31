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
void initialize();
void capture();

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

void onMouseCalibrate(int event, int x, int y, int flags, void*) {
  if (event == 1) {
    cout << x << " " << y << endl;
  }
}

void initialize() {
  cv::Mat image;
  std::vector<cv::Point_<double> > uv;
  std::vector<cv::Point3_<double> > vertices;

  cv::VideoCapture capture;
  if (!capture.open(0)) {
    perror("Can not open web camera");
  }

  cv::namedWindow("calibrate");  
  cv::setMouseCallback("calibrate", onMouseCalibrate, 0);  

  while (true) {
    capture >> image;
    pyrDown(image, image, cv::Size(image.cols / 2, image.rows / 2));
    int result = calnFeatures(image, uv, vertices);

    cv::Mat displayImage = displayFeatures(image, uv);
    cv::imshow("calibrate", displayImage);
    std::cout << "result = " << result << ". Is it ok? [y/n]" << std::endl;
    char ch = cv::waitKey(0);
    if (ch == 'y' && result != -1) {
      break;
    }
  }

  cv::destroyWindow("calibrate");
  cv::waitKey(1);
  cv::imwrite("Data/model.jpg", image);

  produceModel("Data/model", image, uv);

  std::ofstream fout;
  fout.open("Data/model.uv");
  for (int i = 0; i < uv.size(); i++) {
    fout << uv[i].x << " " << uv[i].y << endl;
  }
  fout.close();

  std::cout << "initialize done." << std::endl;
  printUsage();
}

void capture() {
  cv::Mat image;
  std::vector<cv::Point3_<double> > vertices;
  std::vector<cv::Point_<double> > uv;
  std::vector<int> tris;

  image = cv::imread("Data/model.jpg");

  std::ifstream fin;
  fin.open("Data/model.tri");
  int x;
  while (fin >> x) {
    tris.push_back(x);
  }
  fin.close();

  fin.open("Data/model.uv");
  int u, v;
  while (fin >> u >> v) {
    uv.push_back(cv::Point_<double>(u, v));
  }
  fin.close();

  frame = new Frame();
  frame->setStrategy(0);
  frame->start(image, uv, tris);
  std::cout << "connect done" << std::endl;

  Avatar* avatar = LoadAvatar("Data/model");
  if (!avatar) {
    throw make_runtime_error("Failed to load avatar.");
  }
  avatar->setAvatar(0);

  cv::VideoCapture capture;
  if (!capture.open(0)) {
    perror("Can not open web camera");
    return;
  }

  while (true) {
    capture >> image;
    pyrDown(image, image, cv::Size(image.cols / 2, image.rows / 2));
    int result = calnFeatures(image, uv, vertices);
    if (result > 0) {
      frame->update(image, uv, vertices);
    }
    cv::Mat displayImage = displayFeatures(image, uv);
    cv::imshow("capture", displayImage);
    char cmd = cv::waitKey(1);
    if (cmd == '0') {
      frame->setStrategy(0);
    }
    if (cmd == '1') {
      frame->setStrategy(1);
    }
    if (cmd == '2') {
      frame->setStrategy(2);
    }
    if (cmd == 'e') {
      break;
    }
  }

  cv::destroyWindow("capture");
  cv::waitKey(1);
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
