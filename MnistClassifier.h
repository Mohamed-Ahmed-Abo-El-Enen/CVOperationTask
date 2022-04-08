#pragma once
#include <iostream>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <numeric> 

using namespace std;

struct ModelOutput
{
	int predicted_number;
	float predicted_confidence;

	ModelOutput(int _predicted_number, float _predicted_confidence)
	{
		predicted_number = _predicted_number;
		predicted_confidence = _predicted_confidence;
	}

	ModelOutput& operator=(const ModelOutput& a)
	{
		predicted_number = a.predicted_number;
		predicted_confidence = a.predicted_confidence;
		return *this;
	}
};

template <typename T>
float roundoff(T value, unsigned char prec)
{
	T pow_10 = pow(10.0f, (T)prec);
	return round(value * pow_10) / pow_10;
}

template <typename T>
static T softmax(T& vector)
{
	T e(vector.size());
	float sum = 0.0f;
	for (size_t i = 0; i < vector.size(); ++i)
		sum += e[i] = std::exp(vector[i]);

	for (size_t i = 0; i < vector.size(); ++i)
		e[i] = e[i] / sum;

	return e;
}

template <typename T>
T VectorProduct(const vector<T>& vec)
{
	return accumulate(vec.begin(), vec.end(), 1, multiplies<T>());
}

class MnistClassifier
{
private:
	Ort::Env env;
	Ort::Session session;
	Ort::SessionOptions sessionOptions;
	Ort::AllocatorWithDefaultOptions allocator;

	const char* inputName;
	vector<int64_t> inputDims;

	const char* outputName;
	vector<int64_t> outputDims;

public:
	MnistClassifier(string model_weights_path);
	cv::Mat preprocess(cv::Mat image);
	ModelOutput postprocess(vector<float> model_output);
	ModelOutput predict(string image_path);
};