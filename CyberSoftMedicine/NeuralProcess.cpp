/******************************************************************************************/
/***********			Author : Tran Minh Duc (Snowdence)				*******************/
/***********			Email  : Snowdence2911@gmail.com				*******************/
/***********			Phone  : 0982 259 245    						*******************
/***********			Project: CyberSoftMedicine - CyberCorp Inc. 	*******************/
/***********			Date   : 9:51 09/04/2017						*******************/
/***********			Compile: Visual Studio 2017						*******************/
/******************************************************************************************/
#include <iostream>
#include <fstream>
#include <math.h>
#include "NeuralProcess.h"
using namespace std;
NeuralProcess::NeuralProcess(NeuralLayer* architectneuralLayer) :neuralLayer(architectneuralLayer),
epoch(0),
learningRate(0.001), momentum(0.9), desiredAccuracy(70), trainingSetAccuracy(0), validationSetAccuracy(0), generalizationSetAccuracy(0), trainingSetMSE(0), validationSetMSE(0), generalizationSetMSE(0)
{
	deltaInputToHidden = new(double*[neuralLayer->numInputLayer + 1]);
	for (int i = 0; i <= neuralLayer->numInputLayer; i++)
	{
		deltaInputToHidden[i] = new (double[neuralLayer->numHiddenLayer]);
		for (int j = 0; j < neuralLayer->numHiddenLayer; j++) {
			deltaInputToHidden[i][j] = 0;
		}

	}

	deltaHiddenToOutput = new(double*[neuralLayer->numHiddenLayer + 1]);
	for (int i = 0; i <= neuralLayer->numHiddenLayer; i++)
	{
		deltaHiddenToOutput[i] = new (double[neuralLayer->numOutputLayer]);
		for (int j = 0; j < neuralLayer->numOutputLayer; j++) {
			deltaHiddenToOutput[i][j] = 0;
		}
	}

	//create error gradient vector
	//--------------------------------------------------------------------------------------------------------
	hiddenErrorGradients = new(double[neuralLayer->numHiddenLayer + 1]);
	for (int i = 0; i <= neuralLayer->numHiddenLayer; i++) {
		hiddenErrorGradients[i] = 0;
	}

	outputErrorGradients = new(double[neuralLayer->numOutputLayer + 1]);
	for (int i = 0; i <= neuralLayer->numOutputLayer; i++) {
		outputErrorGradients[i] = 0;
	}

}


NeuralProcess::~NeuralProcess()
{
	// release all if necessary
}

void NeuralProcess::setTrainingParameters(double lR, double m, bool batch)
{
	learningRate = lR;
	momentum = m;
	BatchMode = batch;
}

void NeuralProcess::setStoppingConditions(int mEpochs, double dAccuracy)
{
	maxEpochs = mEpochs;
	desiredAccuracy = dAccuracy;
}

void NeuralProcess::trainNetwork(NeuralDataPackage * dataPackage)
{

	cout << endl << " Neural Network For Dectect Cancer [CyBerMedicine] System: " << endl
		<< "==========================================================================" << endl
		<< " Author : Snowdence " << endl
		<< " LearningRate: " << learningRate << ", Momen: " << momentum << ", Epoch: " << maxEpochs << endl
		<< " " << neuralLayer->numInputLayer << " Input Neurons, " << neuralLayer->numHiddenLayer << " Hidden Neurons, " << neuralLayer->numOutputLayer << " Output Neurons" << endl
		<< "==========================================================================" << endl << endl;

	// default value of epoch
	epoch = 0;


	// Train my network
	//--------------------------------------------------------------------------------------------------------
	while ((trainingSetAccuracy < desiredAccuracy || generalizationSetAccuracy < desiredAccuracy) && epoch < maxEpochs)
	{
		// store previous epoch
		double previousTAccuracy = trainingSetAccuracy;
		double previousGAccuracy = generalizationSetAccuracy;

		//use training set  to train
		runTrainingEpoch(dataPackage->trainingSet);

		//get ERROR
		generalizationSetAccuracy = neuralLayer->getAccuracy(dataPackage->generalizationSet);
		generalizationSetMSE = neuralLayer->getMSE(dataPackage->generalizationSet);



		// IF this Greater, update into screen
		if (ceil(previousTAccuracy) != ceil(trainingSetAccuracy) || ceil(previousGAccuracy) != ceil(generalizationSetAccuracy))
		{
			cout << "Epoch [The he] :" << epoch;
			cout << " Training [Tap huan luyen]:" << trainingSetAccuracy << "%, MSE [SAI SO]: " << trainingSetMSE;
			cout << " Generation [Tap thu nghiem]:" << generalizationSetAccuracy << "%, MSE [SAI SO]: " << generalizationSetMSE << endl;
		}
		//increase epoch
		epoch++;

	}

	//ERROR
	validationSetAccuracy = neuralLayer->getAccuracy(dataPackage->validationSet);
	validationSetMSE = neuralLayer->getMSE(dataPackage->validationSet);


	//REsult error
	cout << endl << "[CyBerMedicine] Dao tao hoan thanh voi the he: " << epoch << endl;
	cout << " Accuracy (Do chinh xac): " << validationSetAccuracy << endl;
	cout << " MSE (SAI SO HAM BINH PHUONG LOI): " << validationSetMSE << endl << endl;
}


inline double NeuralProcess::getOutputErrorGradient(double desiredValue, double outputValue)
{
	return outputValue * (1 - outputValue) * (desiredValue - outputValue);
}

double NeuralProcess::getHiddenErrorGradient(int j)
{
	double weightedSum = 0;
	for (int k = 0; k < neuralLayer->numOutputLayer; k++) {
		weightedSum += neuralLayer->weightHiddenToOutput[j][k] * outputErrorGradients[k];
	}

	//ERROR gradient
	return neuralLayer->hiddenLayer[j] * (1 - neuralLayer->hiddenLayer[j]) * weightedSum;
}

void NeuralProcess::runTrainingEpoch(std::vector<DataEntry*> trainingDataSet)
{
	//incorrect Pattern 
	double incorrectPatterns = 0;
	double mse = 0;

	for (int tp = 0; tp < (int)trainingDataSet.size(); tp++)
	{

		neuralLayer->feedForward(trainingDataSet[tp]->pattern);
		backpropagate(trainingDataSet[tp]->target);


		bool patternCorrect = true;

		// Kiem tra so voi gia tri ky vong
		for (int k = 0; k < neuralLayer->numOutputLayer; k++)
		{

			if (neuralLayer->transferOutput(neuralLayer->outputLayer[k]) != trainingDataSet[tp]->target[k]) { patternCorrect = false; }

			//tinh toan loi
			mse += pow((neuralLayer->outputLayer[k] - trainingDataSet[tp]->target[k]), 2);
		}

		//count ++
		if (!patternCorrect) incorrectPatterns++;


	}

	// Batch learning update weight
	if (BatchMode) updateWeights();

	//update MSE
	trainingSetAccuracy = 100 - (incorrectPatterns / trainingDataSet.size() * 100);
	trainingSetMSE = mse / (neuralLayer->numOutputLayer * trainingDataSet.size());
}

void NeuralProcess::backpropagate(double * desiredOutputs)
{
	// lan truyen nguoc
	//--------------------------------------------------------------------------------------------------------
	for (int k = 0; k < neuralLayer->numOutputLayer; k++)
	{
		//output neuron
		outputErrorGradients[k] = getOutputErrorGradient(desiredOutputs[k], neuralLayer->outputLayer[k]);

		//hidden neuron
		for (int j = 0; j <= neuralLayer->numHiddenLayer; j++)
		{
			// tinh toan do sai khac trong so
			if (!BatchMode) {
				deltaHiddenToOutput[j][k] = learningRate * neuralLayer->hiddenLayer[j] * outputErrorGradients[k] + momentum * deltaHiddenToOutput[j][k];
			}
			else {
				deltaHiddenToOutput[j][k] += learningRate * neuralLayer->hiddenLayer[j] * outputErrorGradients[k];
			}
		}
	}

	// EDIT 
	//--------------------------------------------------------------------------------------------------------
	for (int j = 0; j < neuralLayer->numHiddenLayer; j++)
	{
		//error hidden
		hiddenErrorGradients[j] = getHiddenErrorGradient(j);

		//input + bias to hidden
		for (int i = 0; i <= neuralLayer->numInputLayer; i++)
		{
			//tinh toan thay doi
			if (!BatchMode) deltaInputToHidden[i][j] = learningRate * neuralLayer->inputLayer[i] * hiddenErrorGradients[j] + momentum * deltaInputToHidden[i][j];
			else deltaInputToHidden[i][j] += learningRate * neuralLayer->inputLayer[i] * hiddenErrorGradients[j];

		}
	}


	if (!BatchMode) updateWeights();

}

void NeuralProcess::updateWeights()
{

	//input -> hidden weights
	//--------------------------------------------------------------------------------------------------------
	for (int i = 0; i <= neuralLayer->numInputLayer; i++)
	{
		for (int j = 0; j < neuralLayer->numHiddenLayer; j++)
		{
			//update weight
			neuralLayer->weightInputToHidden[i][j] += deltaInputToHidden[i][j];

			//batchmode clear previous
			if (BatchMode) deltaInputToHidden[i][j] = 0;
		}
	}

	//hidden -> output weights
	//--------------------------------------------------------------------------------------------------------
	for (int j = 0; j <= neuralLayer->numHiddenLayer; j++)
	{
		for (int k = 0; k < neuralLayer->numOutputLayer; k++)
		{
			//update weight
			neuralLayer->weightHiddenToOutput[j][k] += deltaHiddenToOutput[j][k];

			//clear if batch mode
			if (BatchMode)deltaHiddenToOutput[j][k] = 0;
		}
	}
}