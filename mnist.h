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

#ifndef SAVED_MODEL_EXAMPLE_MNIST_H_
#define SAVED_MODEL_EXAMPLE_MNIST_H_

#include <stdint.h>
#include <string>
#include <vector>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace mnist {

// MNISTImage contains the 28 x 28 pixel data for
// an MNIST picture, in row major order.
struct MNISTImage {
  static const int kSize = 28;
  float buf[kSize][kSize];
};

// Prints to stdout an ASCII art representation
// of the MNISTImage, useful for debugging purposes.
void MNISTPrint(const MNISTImage& image);

// Converts an MNIST Image to a tensorflow Tensor.
tensorflow::Tensor MNISTImageToTensor(const MNISTImage& image);

// MNISTImageReader helps read a vector of MNISTImages
// from a file path to the MNIST Image dataset. Note
// that the file must already be decompressed.
class MNISTImageReader {
 public:
  MNISTImageReader(const std::string& mnist_path);
  tensorflow::Status ReadMnistImages(std::vector<MNISTImage>* images);

 private:
  std::string mnist_path_;
};

}  // namespace mnist

#endif  // SAVED_MODEL_EXAMPLE_MNIST_H_
