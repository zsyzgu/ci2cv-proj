#include "client.h"
#include <stdio.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <string.h>

Client::Client() {
  port = 6789;
  addr = "192.168.1.135";
}

Client::~Client() {

}

bool Client::start() {
  struct sockaddr_in server_addr;
  server_addr.sin_len = sizeof(sockaddr_in);
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(port);
  server_addr.sin_addr.s_addr = inet_addr(addr.c_str());
  bzero(&(server_addr.sin_zero), 8);

  server_socket = socket(AF_INET, SOCK_STREAM, 0);
  if (server_socket == -1) {
    perror("socket error");
    return false;
  }

  return connect(server_socket, (struct sockaddr*)& server_addr, sizeof(struct sockaddr_in)) == 0;
}

void Client::sendByteArray(char id, int len, char* data) {
  char confirm;
  char* info = new char[5];
  info[0] = id;
  memcpy(info + 1, &len, sizeof(len));
  send(server_socket, info, 5, 0);
  send(server_socket, data, len, 0);
  delete[] info;
}

void Client::sendIntArray(char id, std::vector<int> intArray) {
  int n = intArray.size();
  char* data = new char[n * 4];
  for (int i = 0; i < n; i++) {
    memcpy(data + i * 4, &intArray[i], 4);
  }
  sendByteArray(id, n * 4, data);
  delete[] data;
}

void Client::sendFloatArray(char id, std::vector<float> floatArray) {
  int n = floatArray.size();
  char* data = new char[n * 4];
  for (int i = 0; i < n; i++) {
    memcpy(data + i * 4, &floatArray[i], 4);
  }
  sendByteArray(id, n * 4, data);
  delete[] data;
}

void Client::sendImage(char id, cv::Mat image) {
  std::vector<uchar> buffer;
  imencode(".jpg", image, buffer);
  int n = buffer.size();
  char* data = new char[n];
  for (int i = 0; i < n; i++) {
    data[i] = (char)buffer[i];
  }
  sendByteArray(id, n, data);
  delete[] data;
}

void Client::sendPointArray(char id, std::vector<cv::Point_<double> > pointArray) {
  std::vector<float> floatArray;
  for (int i = 0; i < pointArray.size(); i++) {
    floatArray.push_back(pointArray[i].x);
    floatArray.push_back(pointArray[i].y);
  }
  sendFloatArray(id, floatArray);
}

void Client::sendPoint3Array(char id, std::vector<cv::Point3_<double> > point3Array) {
  std::vector<float> floatArray;
  for (int i = 0; i < point3Array.size(); i++) {
    floatArray.push_back(point3Array[i].x);
    floatArray.push_back(point3Array[i].y);
    floatArray.push_back(point3Array[i].z);
  }
  sendFloatArray(id, floatArray);
}

void Client::end() {
  
}
