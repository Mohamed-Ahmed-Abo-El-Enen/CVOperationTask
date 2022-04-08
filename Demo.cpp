#include "MnistClassifier.h"

int main()
{
	string model_weights_path = "../resources/models/model.onnx";
	string test_image_path = "../resources/test_data/5.png";
	MnistClassifier classificationAgent(model_weights_path);
	ModelOutput modelOutput = classificationAgent.predict(test_image_path);
	cout << "Predicted Number = " << modelOutput.predicted_number << endl;
	cout << "Prediction Confidence = " << roundoff(modelOutput.predicted_confidence * 100, 2) << endl;
	return 0;
}