/******************************************************************************************/
/***********			Author : Tran Minh Duc (Snowdence)				*******************/
/***********			Email  : Snowdence2911@gmail.com				*******************/
/***********			Phone  : 0982 259 245    						*******************
/***********			Project: CyberSoftMedicine - CyberCorp Inc. 	*******************/
/***********			Date   : 9:51 09/04/2017						*******************/
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
		for (int j = 0; j < neuralLayer->numHiddenLayer; j++) deltaInputToHidden[i][j] = 0;
	}

	deltaHiddenToOutput = new(double*[neuralLayer->numHiddenLayer + 1]);
	for (int i = 0; i <= neuralLayer->numHiddenLayer; i++)
	{
		deltaHiddenToOutput[i] = new (double[neuralLayer->numOutputLayer]);
		for (int j = 0; j < neuralLayer->numOutputLayer; j++) deltaHiddenToOutput[i][j] = 0;
	}

	//create error gradient storage
	//--------------------------------------------------------------------------------------------------------
	hiddenErrorGradients = new(double[neuralLayer->numHiddenLayer + 1]);
	for (int i = 0; i <= neuralLayer->numHiddenLayer; i++) hiddenErrorGradients[i] = 0;

	outputErrorGradients = new(double[neuralLayer->numOutputLayer + 1]);
	for (int i = 0; i <= neuralLayer->numOutputLayer; i++) outputErrorGradients[i] = 0;

}


NeuralProcess::~NeuralProcess()
{

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

	cout << endl << " Neural Network Training Starting: " << endl
		<< "==========================================================================" << endl
		<< " LR: " << learningRate << ", Momentum: " << momentum << ", Max Epochs: " << maxEpochs << endl
		<< " " << neuralLayer->numInputLayer << " Input Neurons, " << neuralLayer->numHiddenLayer << " Hidden Neurons, " << neuralLayer->numOutputLayer << " Output Neurons" << endl
		<< "==========================================================================" << endl << endl;

	//reset epoch and log counters
	epoch = 0;


	//train network using training dataset for training and generalization dataset for testing
	//--------------------------------------------------------------------------------------------------------
	while ((trainingSetAccuracy < desiredAccuracy || generalizationSetAccuracy < desiredAccuracy) && epoch < maxEpochs)
	{
		//store previous accuracy
		double previousTAccuracy = trainingSetAccuracy;
		double previousGAccuracy = generalizationSetAccuracy;

		//use training set to train network
		runTrainingEpoch(dataPackage->trainingSet);

		//get generalization set accuracy and MSE
		generalizationSetAccuracy = neuralLayer->getAccuracy(dataPackage->generalizationSet);
		generalizationSetMSE = neuralLayer->getMSE(dataPackage->generalizationSet);



		//print out change in training /generalization accuracy (only if a change is greater than a percent)
		if (ceil(previousTAccuracy) != ceil(trainingSetAccuracy) || ceil(previousGAccuracy) != ceil(generalizationSetAccuracy))
		{
			cout << "Epoch :" << epoch;
			cout << " TSet Acc:" << trainingSetAccuracy << "%, MSE: " << trainingSetMSE;
			cout << " GSet Acc:" << generalizationSetAccuracy << "%, MSE: " << generalizationSetMSE << endl;
		}

		//once training set is complete increment epoch
		epoch++;
		//cout << "Epoch :" << epoch << endl;
	}//end while

	 //get validation set accuracy and MSE
	validationSetAccuracy = neuralLayer->getAccuracy(dataPackage->validationSet);
	validationSetMSE = neuralLayer->getMSE(dataPackage->validationSet);


	//out validation accuracy and MSE
	cout << endl << "Training Complete!!! - > Elapsed Epochs: " << epoch << endl;
	cout << " Validation Set Accuracy: " << validationSetAccuracy << endl;
	cout << " Validation Set MSE: " << validationSetMSE << endl << endl;
}


inline double NeuralProcess::getOutputErrorGradient(double desiredValue, double outputValue)
{
	return outputValue * (1 - outputValue) * (desiredValue - outputValue);
}

double NeuralProcess::getHiddenErrorGradient(int j)
{
	double weightedSum = 0;
	for (int k = 0; k < neuralLayer->numOutputLayer; k++) weightedSum += neuralLayer->weightHiddenToOutput[j][k] * outputErrorGradients[k];

	//return error gradient
	return neuralLayer->hiddenLayer[j] * (1 - neuralLayer->hiddenLayer[j]) * weightedSum;
}

void NeuralProcess::runTrainingEpoch(std::vector<DataEntry*> trainingDataSet)
{
	//incorrect patterns
	double incorrectPatterns = 0;
	double mse = 0;

	//for every training pattern
	for (int tp = 0; tp < (int)trainingDataSet.size(); tp++)
	{
		//feed inputs through network and backpropagate errors
		neuralLayer->feedForward(trainingDataSet[tp]->pattern);
		backpropagate(trainingDataSet[tp]->target);

		//pattern correct flag
		bool patternCorrect = true;

		//check all outputs from neural network against desired values
		for (int k = 0; k < neuralLayer->numOutputLayer; k++)
		{
			//pattern incorrect if desired and output differ
			if (neuralLayer->transferOutput(neuralLayer->outputLayer[k]) != trainingDataSet[tp]->target[k]) patternCorrect = false;

			//calculate MSE
			mse += pow((neuralLayer->outputLayer[k] - trainingDataSet[tp]->target[k]), 2);
		}

		//if pattern is incorrect add to incorrect count
		if (!patternCorrect) incorrectPatterns++;
		

	}//end for

	 //if using batch learning - update the weights
	if (BatchMode) updateWeights();

	//update training accuracy and MSE
	trainingSetAccuracy = 100 - (incorrectPatterns / trainingDataSet.size() * 100);
	trainingSetMSE = mse / (neuralLayer->numOutputLayer * trainingDataSet.size());
}

void NeuralProcess::backpropagate(double * desiredOutputs)
{
	//modify deltas between hidden and output layers
	//--------------------------------------------------------------------------------------------------------
	for (int k = 0; k < neuralLayer->numOutputLayer; k++)
	{
		//get error gradient for every output node
		outputErrorGradients[k] = getOutputErrorGradient(desiredOutputs[k], neuralLayer->outputLayer[k]);

		//for all nodes in hidden layer and bias neuron
		for (int j = 0; j <= neuralLayer->numHiddenLayer; j++)
		{
			//calculate change in weight
			if (!BatchMode) deltaHiddenToOutput[j][k] = learningRate * neuralLayer->hiddenLayer[j] * outputErrorGradients[k] + momentum * deltaHiddenToOutput[j][k];
			else deltaHiddenToOutput[j][k] += learningRate * neuralLayer->hiddenLayer[j] * outputErrorGradients[k];
		}
	}

	//modify deltas between input and hidden layers
	//--------------------------------------------------------------------------------------------------------
	for (int j = 0; j < neuralLayer->numHiddenLayer; j++)
	{
		//get error gradient for every hidden node
		hiddenErrorGradients[j] = getHiddenErrorGradient(j);

		//for all nodes in input layer and bias neuron
		for (int i = 0; i <= neuralLayer->numInputLayer; i++)
		{
			//calculate change in weight 
			if (!BatchMode) deltaInputToHidden[i][j] = learningRate * neuralLayer->inputLayer[i] * hiddenErrorGradients[j] + momentum * deltaInputToHidden[i][j];
			else deltaInputToHidden[i][j] += learningRate * neuralLayer->inputLayer[i] * hiddenErrorGradients[j];

		}
	}

	//if using stochastic learning update the weights immediately
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

			//clear delta only if using batch (previous delta is needed for momentum
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

			//clear delta only if using batch (previous delta is needed for momentum)
			if (BatchMode)deltaHiddenToOutput[j][k] = 0;
		}
	}
}