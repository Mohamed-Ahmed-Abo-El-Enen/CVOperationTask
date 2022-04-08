#include "MnistClassifier.h"


MnistClassifier::MnistClassifier(string model_weights_path) : session(env, wstring(model_weights_path.begin(), model_weights_path.end()).c_str(), sessionOptions)
{
	Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
	auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
	inputDims = inputTensorInfo.GetShape();

	inputName = session.GetInputName(0, allocator);
	outputName = session.GetOutputName(0, allocator);

	Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
	auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
	outputDims = outputTensorInfo.GetShape();
}

cv::Mat MnistClassifier::preprocess(cv::Mat image)
{
	cv::Mat gray, input;
	cv::cvtColor(image, gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);
	cv::resize(gray, gray, cv::Size(inputDims.at(3), inputDims.at(2)));
	gray.convertTo(input, CV_32F, 1.0 / 255);
	cv::dnn::blobFromImage(input, input);
	return input;
}

ModelOutput MnistClassifier::postprocess(vector<float> model_output)
{
	vector<float> result = softmax(model_output);
	int predicted_number = max_element(result.begin(), result.end()) - result.begin();
	float predicted_confidence = result[predicted_number];
	return ModelOutput(predicted_number, predicted_confidence);
}

ModelOutput MnistClassifier::predict(string image_path)
{
	cv::Mat image = cv::imread(image_path);
	cv::Mat preprocessed_image = preprocess(image);

	vector<const char*> inputNames{ inputName };
	vector<const char*> outputNames{ outputName };
	vector<Ort::Value> inputTensors;
	vector<Ort::Value> outputTensors;

	size_t inputTensorSize = VectorProduct(inputDims);
	vector<float> inputTensorValues(inputTensorSize);
	inputTensorValues.assign(preprocessed_image.begin<float>(),
		preprocessed_image.end<float>());

	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
		OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
	inputTensors.push_back(Ort::Value::CreateTensor<float>(
		memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
		inputDims.size()));

	size_t outputTensorSize = VectorProduct(outputDims);
	vector<float> outputTensorValues(outputTensorSize);
	outputTensors.push_back(Ort::Value::CreateTensor<float>(
		memoryInfo, outputTensorValues.data(), outputTensorSize,
		outputDims.data(), outputDims.size()));

	session.Run(Ort::RunOptions{ nullptr }, inputNames.data(),
		inputTensors.data(), 1, outputNames.data(),
		outputTensors.data(), 1);

	ModelOutput modelOutput = postprocess(outputTensorValues);
	return modelOutput;
}