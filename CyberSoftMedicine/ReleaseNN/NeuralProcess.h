/******************************************************************************************/
/***********			Author : Tran Minh Duc (Snowdence)				*******************/
/***********			Email  : Snowdence2911@gmail.com				*******************/
/***********			Phone  : 0982 259 245    						*******************
/***********			Project: CyberSoftMedicine - CyberCorp Inc. 	*******************/
/***********			Date   : 9:51 09/04/2017						*******************/
/******************************************************************************************/
#pragma once
#include <fstream>
#include <vector>
#include "NeuralLayer.h"

class NeuralProcess
{
private:
	NeuralLayer* neuralLayer;
	double learningRate, momentum, desiredAccuracy;
	long epoch, maxEpochs;
	double** deltaInputToHidden, ** deltaHiddenToOutput;
	double* hiddenErrorGradients, *outputErrorGradients;
	
	double trainingSetAccuracy, validationSetAccuracy, generalizationSetAccuracy, trainingSetMSE, validationSetMSE, generalizationSetMSE;
	bool BatchMode;
public:
	NeuralProcess(NeuralLayer* architectNN);
	~NeuralProcess();
	void setTrainingParameters(double lR, double m, bool batch);
	void setStoppingConditions(int mEpochs, double dAccuracy);
	void useBatchLearning(bool flag) { BatchMode = flag; }
	void trainNetwork(NeuralDataPackage* dataPackage);
	void testNetwork(NeuralDataPackage* dataPackage);

private:
	inline double getOutputErrorGradient(double desiredValue, double outputValue);
	double getHiddenErrorGradient(int j);
	void runTrainingEpoch(std::vector<DataEntry*> trainingDataSet);
	void backpropagate(double* desiredOutputs);
	void updateWeights();
};