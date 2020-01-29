#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>

#include "mnist.h"

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/public/session.h"

int main(int argc, char** argv) {
  // Load the saved model from the provided path.
  std::string export_dir =
      "/usr/local/google/home/bmzhao/Code/saved-model-example/mnist_model";
  tensorflow::SavedModelBundle bundle;
  tensorflow::RunOptions run_options;
  tensorflow::Status status =
      tensorflow::LoadSavedModel(tensorflow::SessionOptions(), run_options,
                                 export_dir, {"serve"}, &bundle);
  if (!status.ok()) {
    std::cerr << "Error loading saved model " << status << std::endl;
    return 1;
  }

  // Print the signature defs.
  // This keras model should have an input named
  // "serving_default_flatten_input", and an output named
  // "StatefulPartitionedCall".
  auto signature_map = bundle.GetSignatures();
  for (const auto& name_and_signature_def : signature_map) {
    const auto& name = name_and_signature_def.first;
    const auto& signature_def = name_and_signature_def.second;
    std::cerr << "Name: " << name << std::endl;
    std::cerr << "SignatureDef: " << signature_def.DebugString() << std::endl;
  }

  // Load the MNIST images from the given path.
  std::vector<mnist::MNISTImage> images;
  mnist::MNISTImageReader image_reader(
      "/usr/local/google/home/bmzhao/Code/saved-model-example/"
      "t10k-images.idx3-ubyte");
  status = image_reader.ReadMnistImages(&images);
  if (!status.ok() || images.empty()) {
    std::cerr << "Error reading MNIST images" << status << std::endl;
    return 2;
  }

  // Convert the first image to a tensorflow::Tensor.
  tensorflow::Tensor input_image = mnist::MNISTImageToTensor(images[0]);
  mnist::MNISTPrint(images[0]);

  tensorflow::Session* session = bundle.GetSession();
  std::vector<tensorflow::Tensor> output_tensors;
  output_tensors.push_back({});

  status = session->Run({{"serving_default_flatten_input", input_image}},
                        {"StatefulPartitionedCall"}, {}, &output_tensors);
  if (!status.ok()) {
    std::cerr << "Error executing session.Run() " << status << std::endl;
    return 3;
  }

  for (const auto& output_tensor : output_tensors) {
    tensorflow::TensorProto proto;
    output_tensor.AsProtoField(&proto);
    std::cerr << "TensorProto Debug Representation: " << proto.DebugString()
              << std::endl;

    const auto& vec = output_tensor.flat_inner_dims<float>();
    float max = 0;
    int argmax = 0;
    for (int i = 0; i < vec.size(); ++i) {
      if (vec(i) > max) {
        argmax = i;
      }
    }
    std::cerr << "Predicted Number: " << argmax << std::endl;
  }
}