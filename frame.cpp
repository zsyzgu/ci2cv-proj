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

Frame::Frame() {
  client.start();
}

Frame::~Frame() {
  client.end();
}

void Frame::setStrategy(int strategy) {
  std::cout << "strategy = " << strategy << std::endl;
  this->strategy = strategy;
}

cv::Rect_<double> Frame::getLeftEyeRect() {
  assert(tris.size() / 3 >= 14);
  return getRegionRect(tris.size() / 3 - 14, 4);
}

cv::Rect_<double> Frame::getRightEyeRect() {
  assert(tris.size() / 3 >= 10);
  return getRegionRect(tris.size() / 3 - 10, 4);
}

cv::Rect_<double> Frame::getMouthRect() {
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

void Frame::cutMouthRegion() {
  cv::Rect_<double> rect = getMouthRect();
  cutImage(faceImage, mouthImage, rect);
  cutUV(faceUV, mouthUV, rect);
}

void Frame::start(cv::Mat modelImage, std::vector<cv::Point_<double> > modelUV, std::vector<int> tris) {
  this->modelImage = modelImage;
  this->modelUV = modelUV;
  this->tris = tris;
  client.sendImage(0, modelImage);
  client.sendPointArray(1, modelUV);
  client.sendIntArray(3, tris);
  this->faceImage = modelImage;
  this->faceUV = modelUV;
  cutLeftEyeRegion();
  cutRightEyeRegion();
  cutMouthRegion();
  client.sendImage(4, leftEyeImage);
  client.sendPointArray(5, leftEyeUV);
  client.sendImage(6, rightEyeImage);
  client.sendPointArray(7, rightEyeUV);
  client.sendImage(8, mouthImage);
  client.sendPointArray(9, mouthUV);
}

void Frame::update(cv::Mat faceImage, std::vector<cv::Point_<double> > faceUV, std::vector<cv::Point3_<double> > vertices) {
  if (strategy == 0) {
    this->faceImage = faceImage;
    this->faceUV = faceUV;
    this->vertices = vertices;
    cutLeftEyeRegion();
    cutRightEyeRegion();
    cutMouthRegion();
    client.sendImage(0, faceImage);
    client.sendPointArray(1, faceUV);
    client.sendPoint3Array(2, vertices);
    client.sendImage(4, leftEyeImage);
    client.sendPointArray(5, leftEyeUV);
    client.sendImage(6, rightEyeImage);
    client.sendPointArray(7, rightEyeUV);
    client.sendImage(8, mouthImage);
    client.sendPointArray(9, mouthUV);
  }
  if (strategy == 1) {
    this->faceImage = faceImage;
    this->faceUV = faceUV;
    this->vertices = vertices;
    cutLeftEyeRegion();
    cutRightEyeRegion();
    cutMouthRegion();
    client.sendPoint3Array(2, vertices);
    client.sendImage(4, leftEyeImage);
    client.sendPointArray(5, leftEyeUV);
    client.sendImage(6, rightEyeImage);
    client.sendPointArray(7, rightEyeUV);
    client.sendImage(8, mouthImage);
    client.sendPointArray(9, mouthUV);
  }
  if (strategy == 2) {
    this->vertices = vertices;
    client.sendPoint3Array(2, vertices);
  }
}

/*void Frame::saveUV(std::string fileName, std::vector<cv::Point_<double> >& uv) {
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

void Frame::updateSave() {
  imwrite("Pictures/face.jpg", faceImage);
  saveUV("Data/face.uv", faceUV);
  saveVertices("Data/face.ver", vertices);
  imwrite("Pictures/lefteye.jpg", leftEyeImage);
  saveUV("Data/lefteye.uv", leftEyeUV);
  imwrite("Pictures/righteye.jpg", rightEyeImage);
  saveUV("Data/righteye.uv", rightEyeUV);
  imwrite("Pictures/mouth.jpg", mouthImage);
  saveUV("Data/mouth.uv", mouthUV);
}*/
