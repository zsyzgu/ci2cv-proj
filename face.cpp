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

void displayData(const Configuration &cfg, const cv::Mat &image, const std::vector<cv::Point_<double> > &points, const Pose &pose);
void produceModel(std::string modelPathName, cv::Mat& image, std::vector<cv::Point_<double> >& points);

void run(char* file, cv::Mat& image, std::vector<cv::Point_<double> >& shape, std::vector<cv::Point3_<double> >& shape3) {
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

  image = cv::imread((std::string(file) + ".jpg").c_str());
  cv::Mat grayImage;
  cv::cvtColor(image, grayImage, CV_RGB2GRAY);
  int result = tracker->NewFrame(grayImage, trackerParams);

  if (result >= cfg.trackingThreshold) {
    shape = tracker->getShape();
    Pose pose = tracker->getPose();
    displayData(cfg, image, shape, pose);

    for (int i = 0; i < shape.size(); i++) {
      shape[i].x = shape[i].x / image.cols;
      shape[i].y = 1 - shape[i].y / image.rows;
    }
    produceModel((std::string(file) + ".model").c_str(), image, shape);

    shape3 = tracker->get3DShape();
    for (int i = 0; i < shape3.size(); i++) {
      shape3[i].x = shape3[i].x / cfg.faceBoxSize;
      shape3[i].y = shape3[i].y / cfg.faceBoxSize;
      shape3[i].z = shape3[i].z / cfg.faceBoxSize;
    }
  } else {
    throw make_runtime_error("result not good enough.");
  }

  delete tracker;
  delete trackerParams;
}

int main(int argc, char** argv) {
  cv::Mat modelImage;
  std::vector<cv::Point_<double> > modelUV;
  std::vector<cv::Point3_<double> > vertices;
  cv::Mat faceImage;
  std::vector<cv::Point_<double> > faceUV;
  if (argc >= 2) {
    run(argv[1], modelImage, modelUV, vertices);
  }
  if (argc >= 3) {
    run(argv[2], faceImage, faceUV, vertices);

    std::vector<int> tris;
    std::ifstream fin("face.tri");
    int x;
    while (fin >> x) {
      tris.push_back(x);
    }
    Frame frame;
    frame.start(modelImage, modelUV, tris);
    frame.update(faceImage, faceUV, vertices);
  }

  return 0;
}

void produceModel(std::string modelPathName, cv::Mat& image, std::vector<cv::Point_<double> >& points) {
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

  cv::Mat_<double> saragih_points = vectorise_points(points);
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
  cv::waitKey(500);
}
