/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <stdint.h>
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>

#include "mnist.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace mnist {

namespace {

// This function interprets a 4 byte array as an unsigned,
// big-endian integer. sizeof(data) must be 4.
// From Rob Pike's Byte Order Fallacy:
// https://commandcenter.blogspot.com/2012/04/byte-order-fallacy.html
uint32_t ConvertBigEndian(unsigned char data[]) {
  uint32_t result =
      (data[3] << 0) | (data[2] << 8) | (data[1] << 16) | (data[0] << 24);
  return result;
}

}  // namespace

void MNISTPrint(const MNISTImage& image) {
  for (int row = 0; row < MNISTImage::kSize; ++row) {
    for (int column = 0; column < MNISTImage::kSize; ++column) {
      std::string pixel = image.buf[row][column] > 0 ? "X" : " ";
      std::cout << pixel << " ";
    }
    std::cout << "\n";
  }
}

tensorflow::Tensor MNISTImageToTensor(const MNISTImage& image) {
  // https://github.com/tensorflow/tensorflow/issues/8033#issuecomment-520977062
  tensorflow::Tensor input_image(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape({MNISTImage::kSize, MNISTImage::kSize}));
  float* img_tensor_flat = input_image.flat<float>().data();

  std::copy(&image.buf[0][0],
            &image.buf[0][0] + (MNISTImage::kSize * MNISTImage::kSize),
            img_tensor_flat);

  return input_image;
}

MNISTImageReader::MNISTImageReader(const std::string& path)
    : mnist_path_(path) {}

// MNIST's file format is documented here: http://yann.lecun.com/exdb/mnist/
// Note(bmzhao): The serialized integers are in big endian, and the magic
// number is documented to be 2051.
tensorflow::Status MNISTImageReader::ReadMnistImages(
    std::vector<MNISTImage>* images) {
  std::ifstream image_file(mnist_path_, std::ios::binary);
  if (!image_file.is_open()) {
    return tensorflow::errors::NotFound("File ", mnist_path_, " not found");
  }

  unsigned char buf[4];
  uint32_t num_images;
  uint32_t num_rows;
  uint32_t num_columns;

  // Read the magic number.
  image_file.read(reinterpret_cast<char*>(&buf[0]), sizeof(buf));
  uint32_t magic_number = ConvertBigEndian(buf);
  if (magic_number != 2051) {
    return tensorflow::errors::Internal("Magic Number of Mnist Data File ",
                                        mnist_path_, " was ", magic_number,
                                        " expected 2051");
  }

  // Read the number of images.
  image_file.read(reinterpret_cast<char*>(&buf[0]), sizeof(buf));
  num_images = ConvertBigEndian(buf);

  // Read the number of rows.
  image_file.read(reinterpret_cast<char*>(&buf[0]), sizeof(buf));
  num_rows = ConvertBigEndian(buf);
  if (num_rows != MNISTImage::kSize) {
    return tensorflow::errors::FailedPrecondition(
        "Num Rows of Mnist Data File was ", num_rows, " expected 28");
  }

  // Read the number of columns
  image_file.read(reinterpret_cast<char*>(&buf[0]), sizeof(buf));
  num_columns = ConvertBigEndian(buf);
  if (num_columns != MNISTImage::kSize) {
    return tensorflow::errors::FailedPrecondition(
        "Num Columns of Mnist Data File was ", num_columns, " expected 28");
  }

  // Iterate through the images, and create an MNISTImage struct for each
  for (int i = 0; i < num_images; ++i) {
    images->emplace_back();
    uint8_t img_buf[MNISTImage::kSize * MNISTImage::kSize];
    image_file.read(reinterpret_cast<char*>(&img_buf[0]), sizeof(img_buf));

    // Convert the buffer into float MNISTImage
    for (int row = 0; row < num_rows; ++row) {
      for (int column = 0; column < num_columns; ++column) {
        (*images)[i].buf[row][column] =
            img_buf[row * MNISTImage::kSize + column];
      }
    }
  }
  return tensorflow::Status();
}

}  // namespace mnist