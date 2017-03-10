#include "utils/helpers.hpp"
#include "utils/command-line-arguments.hpp"
#include "utils/points.hpp"
#include "tracker/FaceTracker.hpp"
#include <opencv2/highgui/highgui.hpp>

using namespace FACETRACKER;

struct Configuration
{
  std::string model_pathname;
  std::string params_pathname;
  int tracking_threshold;
  std::string window_title;

  int circle_radius;
  int circle_thickness;
  int circle_linetype;
  int circle_shift;
};

int run_image_mode(const Configuration &cfg, const std::string& imageArg, const std::string& landmarksArg);
void display_data(const Configuration &cfg, const cv::Mat &image, const std::vector<cv::Point_<double> > &points, const Pose &pose);

int main(int argc, char** argv) {
  std::string imageArg;
  std::string landmarksArg;

  Configuration cfg;
  cfg.model_pathname = DefaultFaceTrackerModelPathname();
  cfg.params_pathname = DefaultFaceTrackerParamsPathname();
  cfg.tracking_threshold = 5;
  cfg.window_title = "CSIRO Face Fit";
  cfg.circle_radius = 5;
  cfg.circle_thickness = 2;
  cfg.circle_linetype = 8;
  cfg.circle_shift = 0;

  for (int i = 1; i < argc; i++) {
    std::string arg = std::string(argv[i]);
    if (imageArg == "") {
      imageArg = arg;
    } else if (landmarksArg == "") {
      landmarksArg = arg;
    } else {
      make_runtime_error("Unable to process argv '%s'", arg.c_str());
    }
  }

  return run_image_mode(cfg, imageArg, landmarksArg);
}

int run_image_mode(const Configuration &cfg, const std::string& imageArg, const std::string& landmarksArg) {  
  FaceTracker* tracker = LoadFaceTracker(cfg.model_pathname.c_str());
  FaceTrackerParams* trackerParams  = LoadFaceTrackerParams(cfg.params_pathname.c_str());

  cv::Mat image;
  cv::Mat_<uint8_t> grayImage = load_grayscale_image(imageArg.c_str(), &image);

  int result = tracker->NewFrame(grayImage, trackerParams);

  std::vector<cv::Point_<double> > shape;
  Pose pose;

  if (result >= cfg.tracking_threshold) {
    shape = tracker->getShape();
    pose = tracker->getPose();
  }

  if (landmarksArg == "") {
    display_data(cfg, image, shape, pose);
  } else if (shape.size() > 0) {
    save_points(landmarksArg.c_str(), shape);
  }

  delete tracker;
  delete trackerParams;
  return 0;
}

cv::Mat
compute_pose_image(const Pose &pose, int height, int width)
{
  cv::Mat_<cv::Vec<uint8_t,3> > rv = cv::Mat_<cv::Vec<uint8_t,3> >::zeros(height,width);
  cv::Mat_<double> axes = pose_axes(pose);
  cv::Mat_<double> scaling = cv::Mat_<double>::eye(3,3);

  for (int i = 0; i < axes.cols; i++) {
    axes(0,i) = -0.5*double(width)*(axes(0,i) - 1);
    axes(1,i) = -0.5*double(height)*(axes(1,i) - 1);
  }
  
  cv::Point centre(width/2, height/2);
  // pitch
  cv::line(rv, centre, cv::Point(axes(0,0), axes(1,0)), cv::Scalar(255,0,0));
  // yaw
  cv::line(rv, centre, cv::Point(axes(0,1), axes(1,1)), cv::Scalar(0,255,0));
  // roll
  cv::line(rv, centre, cv::Point(axes(0,2), axes(1,2)), cv::Scalar(0,0,255));

  return rv;
}

void
display_data(const Configuration &cfg,
	     const cv::Mat &image,
	     const std::vector<cv::Point_<double> > &points,
	     const Pose &pose)
{

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
    throw make_runtime_error("Unsupported camera image type for display_data function.");

  for (size_t i = 0; i < points.size(); i++) {
    cv::circle(displayed_image, points[i], cfg.circle_radius, colour, cfg.circle_thickness, cfg.circle_linetype, cfg.circle_shift);
  }

  int pose_image_height = 100;
  int pose_image_width = 100;
  cv::Mat pose_image = compute_pose_image(pose, pose_image_height, pose_image_width);
  for (int i = 0; i < pose_image_height; i++) {
    for (int j = 0; j < pose_image_width; j++) {
      displayed_image.at<cv::Vec<uint8_t,3> >(displayed_image.rows - pose_image_height + i,
					      displayed_image.cols - pose_image_width + j)
			 
			 = pose_image.at<cv::Vec<uint8_t,3> >(i,j);
    }
  }

  cv::imshow(cfg.window_title, displayed_image);
  char ch = cv::waitKey(0);
}
