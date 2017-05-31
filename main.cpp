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
void update(int);

void produceModel(std::string modelPathName, cv::Mat& image, std::vector<cv::Point_<double> >& uv);
cv::Mat displayFeatures(const cv::Mat &image, const std::vector<cv::Point_<double> > &uv);
int calnFeatures(cv::Mat& image, std::vector<cv::Point_<double> >& uv, std::vector<cv::Point3_<double> >& vertices);

int main(int argc, char** argv) {
  if (argc == 2 && strlen(argv[1]) == 1) {
    char ch = argv[1][0];
    if (ch == 'i') {
      initialize();
    }
    if (ch == '0') {
      update(0);
    }
    if (ch == '1') {
      update(1);
    }
    if (ch == '2') {
      update(2);
    }
  } else {
    printUsage();
  }

  if (frame != NULL) {
    delete frame;
  }
  return 0;
}

void printUsage() {
  std::string usage = 
    "i -- initialize\n"
    "0 -- high\n"
    "1 -- middle\n"
    "2 -- low";

  std::cout << usage << std::endl;
}

std::vector<cv::Point_<double> > initUV;
cv::Mat initImage;

void onMouseCalibrate(int event, int intX, int intY, int flags, void*) {
  int imageX = 640;
  int imageY = 360;
  if (event == 1) {
    double x = (double)intX / imageX;
    double y = 1 - (double)intY / imageY;
    int id = -1;
    double minDist2 = 1e9;
    for (int i = 0; i < initUV.size(); i++) {
      double dist2 = (x - initUV[i].x) * (x - initUV[i].x) + (y - initUV[i].y) * (y - initUV[i].y);
      if (dist2 < minDist2) {
        id = i;
        minDist2 = dist2;
      }
    }
    if (id != -1) {
      initUV[id].x = x;
      initUV[id].y = y;
    }
    cv::Mat displayImage = displayFeatures(initImage, initUV);
    cv::imshow("calibrate", displayImage);
  }
}

void initialize() {
  cv::Mat image;
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
    initImage = image;
    int result = calnFeatures(image, initUV, vertices);

    cv::Mat displayImage = displayFeatures(image, initUV);
    cv::imshow("calibrate", displayImage);
    std::cout << "result = " << result << ". Is it ok? [y/n]" << std::endl;
    char ch = cv::waitKey(0);
    if (ch == 'y' && result != -1) {
      break;
    }
  }

  std::vector<cv::Point_<double> > uv = initUV;

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

void update(int strategy) {
  cv::Mat image;
  std::vector<cv::Point3_<double> > vertices;
  std::vector<cv::Point_<double> > uv;
  std::vector<int> tris;

  image = cv::imread("Data/model.jpg");

  std::ifstream fin;
  fin.open("Data/model.uv");
  double u, v;
  while (fin >> u >> v) {
    uv.push_back(cv::Point_<double>(u, v));
  }
  fin.close();

  fin.open("Data/model.tri");
  int x;
  while (fin >> x) {
    tris.push_back(x);
  }
  fin.close();

  frame = new Frame();
  frame->setStrategy(strategy);
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

  cv::Mat displayImage;
  while (true) {
    capture >> image;
    pyrDown(image, image, cv::Size(image.cols / 2, image.rows / 2));
    int result = calnFeatures(image, uv, vertices);
    if (result > 0) {
      frame->update(image, uv, vertices);
      displayImage = displayFeatures(image, uv);
    } else {
      displayImage = image;
    }

    cv::imshow("capture", displayImage);
    cv::waitKey(1);
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
