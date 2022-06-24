// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>


// Create a macro to quickly mark a function for export
#define DLLExport __declspec (dllexport)

// Wrap code to prevent name-mangling issues
extern "C" {

	// The input width for the model
	int input_w = 216;
	// The input height for the model
	int input_h = 216;
	// The current torchscript model
	torch::jit::Module network;
	// The current list of inputs
	std::vector<torch::jit::IValue> inputs;

	// The mean normalization stats for the current model
	std::vector<float> mean_stats;
	// The std normalization stats for the current model
	std::vector<float> std_stats;

	// Update the input dimensions
	DLLExport void SetInputDims(int width, int height) {
		input_w = width;
		input_h = height;
	}

	// Load a torchscript model from the specified file path
	DLLExport int LoadModel(char* modelPath, float mean[3], float std[3]) {

		try {
			// Deserialize the ScriptModule from a file using torch::jit::load().
			network = torch::jit::load(modelPath);
			network.eval();

			mean_stats.clear();
			std_stats.clear();

			for (int i = 0; i < 3; i++) {
				mean_stats.push_back(mean[i]);
				std_stats.push_back(std[i]);
			}			
		}
		catch (const c10::Error& e) {
			// Return a value of -1 if the model fails to load
			return -1;
		}

		// Return a value of 0 if the model loads successfully
		return 0;
	}

	// Perform inference with the provided texture data
	DLLExport int PerformInference(uchar* inputData) {
		
		// Initialize predicted class index to an invalid value
		int class_idx = -1;
		// Empty the inputs vector
		inputs.clear();

		{
			// Enable inference mode
			torch::InferenceMode guard(true);

			// Store the pixel data for the source input image
			cv::Mat texture = cv::Mat(input_h, input_w, CV_8UC4);

			if (!texture.isContinuous()) { texture = texture.clone(); }

			// Assign the inputData to the OpenCV Mat
			texture.data = inputData;
			// Remove the alpha channel
			cv::cvtColor(texture, texture, cv::COLOR_RGBA2RGB);
			// Convert RGB image to a three-channel matrix of 32-bit floats
			texture.convertTo(texture, CV_32FC3);

			// Initialize a tensor using the texture data
			torch::Tensor input = torch::from_blob(texture.data, { 1, input_h, input_w, 3 });
			// Permute tensor dimensions
			input = input.permute({ 0, 3, 1, 2 });
			// Scale and normalize color channel values
			input[0][0] = input[0][0].div_(255.0f).sub_(mean_stats[0]).div_(std_stats[0]);
			input[0][1] = input[0][1].div_(255.0f).sub_(mean_stats[1]).div_(std_stats[1]);
			input[0][2] = input[0][2].div_(255.0f).sub_(mean_stats[2]).div_(std_stats[2]);
			// Add input tensor to inputs vector
			inputs.push_back(input);
			
			try {
				// Perform inference and extract the predicted class index
				class_idx = torch::softmax(network.forward(inputs).toTensor(), 1).argmax().item<int>();
			}
			catch (...) {
				class_idx = -2;
			}
		}

		return class_idx;
	}
}