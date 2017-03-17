#ifndef CLIENT_H
#define CLIENT_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

class Client {
private:
  int port;
  std::string addr;
  int server_socket;
public:
  Client();
  ~Client();
  bool start();
  void sendByteArray(char id, int len, char* data);
  void sendIntArray(char id, std::vector<int> intArray);
  void sendFloatArray(char id, std::vector<float> floatArray);
  void sendImage(char id, cv::Mat image);
  void sendPointArray(char id, std::vector<cv::Point_<double> > pointArray);
  void sendPoint3Array(char id, std::vector<cv::Point3_<double> > point3Array);
  void end();
};

#endif //CLIENT_H