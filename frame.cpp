#include "frame.h"
#include <fstream>
#include <iostream>

void Frame::cutImage(cv::Mat& input, cv::Mat& output, cv::Rect_<double> rect) {
  output = input.colRange(input.cols * rect.x, input.cols * (rect.x + rect.width))
                .rowRange(input.rows * (1 - (rect.y + rect.height)), input.rows * (1 - rect.y));
}

void Frame::cutUV(std::vector<cv::Point_<double> >& input, std::vector<cv::Point_<double> >& output, cv::Rect_<double> rect) {
  output.clear();
  for (int i = 0; i < input.size(); i++) {
    cv::Point_<double> point = input[i];
    point.x = (point.x - rect.x) / rect.width;
    point.y = (point.y - rect.y) / rect.height;
    output.push_back(point);
  }
}

void Frame::readTris() {
  std::ifstream fin("face.tri");
  tris.clear();
  int u, v, w;
  while (fin >> u >> v >> w) {
    tris.push_back(u);
    tris.push_back(v);
    tris.push_back(w);
  }
}

cv::Rect_<double> Frame::getRegionRect(int trisBeginId, int trisNumber) {
  assert(0 <= trisBeginId * 3 && (trisBeginId + trisNumber) * 3 <= tris.size());
  double minX = 1, maxX = 0;
  double minY = 1, maxY = 0;
  for (int i = trisBeginId * 3; i < (trisBeginId + trisNumber) * 3; i++) {
    int u = tris[i];
    if (faceUV[u].x < minX) {
      minX = faceUV[u].x;
    }
    if (faceUV[u].x > maxX) {
      maxX = faceUV[u].x;
    }
    if (faceUV[u].y < minY) {
      minY = faceUV[u].y;
    }
    if (faceUV[u].y > maxY) {
      maxY = faceUV[u].y;
    }
  }
  return cv::Rect_<double>(minX, minY, maxX - minX, maxY - minY);
}

void Frame::saveUV(std::string fileName, std::vector<cv::Point_<double> >& uv) {
  std::ofstream fout(fileName.c_str());
  for (int i = 0; i < uv.size(); i++) {
    fout << uv[i].x << " " << uv[i].y << std::endl;
  }
  fout.close();
}

void Frame::saveVertices(std::string fileName, std::vector<cv::Point3_<double> >& vertices) {
  std::ofstream fout(fileName.c_str());
  for (int i = 0; i < vertices.size(); i++) {
    fout << vertices[i].x << " " << vertices[i].y << " " << vertices[i].z << std::endl;
  }
  fout.close();
}

void Frame::saveTris(std::string fileName, std::vector<int>& vertices) {
  std::ofstream fout(fileName.c_str());
  for (int i = 0; i < tris.size(); i++) {
    fout << tris[i] << std::endl;
  }
  fout.close();
}

Frame::Frame() {
  client.start();
}

Frame::~Frame() {
  client.end();
}

void Frame::setModelImage(const cv::Mat& modelImage) {
  this->modelImage = modelImage;
}

void Frame::setModelUV(const std::vector<cv::Point_<double> >& modelUV) {
  this->modelUV = modelUV;
}

void Frame::setTris(const std::vector<int>& tris) {
  this->tris = tris;
}

void Frame::setFaceImage(const cv::Mat& faceImage) {
  this->faceImage = faceImage;
}

void Frame::setLeftEyeImage(const cv::Mat& leftEyeImage) {
  this->leftEyeImage = leftEyeImage;
}

void Frame::setRightEyeImage(const cv::Mat& rightEyeImage) {
  this->rightEyeImage = rightEyeImage;
}

void Frame::setMouseImage(const cv::Mat& mouseImage) {
  this->mouseImage = mouseImage;
}

void Frame::setVertices(const std::vector<cv::Point3_<double> >& vertices) {
  this->vertices = vertices;
}

void Frame::setFaceUV(const std::vector<cv::Point_<double> >& faceUV) {
  this->faceUV = faceUV;
}

void Frame::setLeftEyeUV(const std::vector<cv::Point_<double> >& leftEyeUV) {
  this->leftEyeUV = leftEyeUV;
}

void Frame::setRightEyeUV(const std::vector<cv::Point_<double> >& rightEyeUV) {
  this->rightEyeUV = rightEyeUV;
}

void Frame::setMouseUV(const std::vector<cv::Point_<double> >& mouseUV) {
  this->mouseUV = mouseUV;
}

cv::Rect_<double> Frame::getLeftEyeRect() {
  assert(tris.size() / 3 >= 14);
  return getRegionRect(tris.size() / 3 - 14, 4);
}

cv::Rect_<double> Frame::getRightEyeRect() {
  assert(tris.size() / 3 >= 10);
  return getRegionRect(tris.size() / 3 - 10, 4);
}

cv::Rect_<double> Frame::getMouseRect() {
  assert(tris.size() / 3 >= 6);
  return getRegionRect(tris.size() / 3 - 6, 6);
}

void Frame::cutLeftEyeRegion() {
  cv::Rect_<double> rect = getLeftEyeRect();
  cutImage(faceImage, leftEyeImage, rect);
  cutUV(faceUV, leftEyeUV, rect);
}

void Frame::cutRightEyeRegion() {
  cv::Rect_<double> rect = getRightEyeRect();
  cutImage(faceImage, rightEyeImage, rect);
  cutUV(faceUV, rightEyeUV, rect);
}

void Frame::cutMouseRegion() {
  cv::Rect_<double> rect = getMouseRect();
  cutImage(faceImage, mouseImage, rect);
  cutUV(faceUV, mouseUV, rect);
}

void Frame::start(cv::Mat modelImage, std::vector<cv::Point_<double> > modelUV, std::vector<int> tris) {
  setModelImage(modelImage);
  setModelUV(modelUV);
  setTris(tris);
  startSave();
}

void Frame::update(cv::Mat faceImage, std::vector<cv::Point_<double> > faceUV, std::vector<cv::Point3_<double> > vertices) {
  setFaceImage(faceImage);
  setFaceUV(faceUV);
  setVertices(vertices);
  cutLeftEyeRegion();
  cutRightEyeRegion();
  cutMouseRegion();
  updateSave();
}

void Frame::startSave() {
  imwrite("Pictures/model.jpg", modelImage);
  saveUV("Data/model.uv", modelUV);
}

void Frame::updateSave() {
  imwrite("Pictures/face.jpg", faceImage);
  saveUV("Data/face.uv", faceUV);
  saveVertices("Data/face.ver", vertices);
  imwrite("Pictures/lefteye.jpg", leftEyeImage);
  saveUV("Data/lefteye.uv", leftEyeUV);
  imwrite("Pictures/righteye.jpg", rightEyeImage);
  saveUV("Data/righteye.uv", rightEyeUV);
  imwrite("Pictures/mouse.jpg", mouseImage);
  saveUV("Data/mouse.uv", mouseUV);
}

  void sendByteArray(char id, int len, char* data);
  void sendIntArray(char id, std::vector<int> intArray);
  void sendFloatArray(char id, std::vector<float> floatArray);
  void sendImage(char id, cv::Mat image);
  void sendPointArray(char id, std::vector<cv::Point_<double> > pointArray);
  void sendPoint3Array(char id, std::vector<cv::Point3_<double> > point3Array);

void Frame::startSend() {
  client.sendImage(0, modelImage);
  client.sendPointArray(1, modelUV);
  client.sendPoint3Array(2, vertices);
  client.sendIntArray(3, tris);
}

void Frame::updateSend() {
  client.sendPoint3Array(2, vertices);
  client.sendImage(4, leftEyeImage);
  client.sendPointArray(5, leftEyeUV);
  client.sendImage(6, rightEyeImage);
  client.sendPointArray(7, rightEyeUV);
  client.sendImage(8, mouseImage);
  client.sendPointArray(9, mouseUV);
}
