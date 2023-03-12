// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"


// Create a macro to quickly mark a function for export
#define DLLExport __declspec (dllexport)

// Wrap code to prevent name-mangling issues
extern "C" {

	// The current torchscript model
	torch::jit::Module network;
	
	// The mean normalization stats for the current model
	std::vector<float> mean_stats;
	// The std normalization stats for the current model
	std::vector<float> std_stats;

	// Load a torchscript model from the specified file path
	DLLExport int LoadModel(char* modelPath, float mean[3], float std[3]) {

		try {
			// Deserialize the ScriptModule from a file using torch::jit::load().
			network = torch::jit::load(modelPath);
			
			// Empty the normalization vectors
			mean_stats.clear();
			std_stats.clear();

			// Update the normalization vectors
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
	DLLExport int PerformInference(uchar* inputData, int width, int height) {
		
		// Store the pixel data for the source input image in an OpenCV Mat
		cv::Mat texture = cv::Mat(height, width, CV_8UC4, inputData);
		// Remove the alpha channel
		cv::cvtColor(texture, texture, cv::COLOR_RGBA2RGB);
		// Convert RGB image to a three-channel matrix of 32-bit floats
		texture.convertTo(texture, CV_32FC3);

		// Initialize a tensor using the texture data
		torch::Tensor input = torch::from_blob(texture.data, { 1, height, width, 3 });
		// Permute tensor dimensions
		input = input.permute({ 0, 3, 1, 2 });
		// Scale and normalize color channel values
		for (int i=0; i < 3; i++) input[0][i].div_(255.0f).sub_(mean_stats[i]).div_(std_stats[i]);
		
		// Initialize a vector to store model inputs
		std::vector<torch::jit::IValue> inputs;
		// Add input tensor to inputs vector
		inputs.push_back(input);

		// Initialize predicted class index to an invalid value
		int class_idx = -1;

		try {
			// Enable inference mode for improved performance
			torch::InferenceMode guard(true);
			// Perform inference and extract the predicted class index
			class_idx = torch::softmax(network.forward(inputs).toTensor(), 1).argmax().item<int>();
		}
		catch (...) {
			// Return a value of -2 if an error occurs during the forward pass
			class_idx = -2;
		}

		return class_idx;
	}
}