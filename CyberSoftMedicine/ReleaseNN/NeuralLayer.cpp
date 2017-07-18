/******************************************************************************************/
/***********			Author : Tran Minh Duc (Snowdence)				*******************/
/***********			Email  : Snowdence2911@gmail.com				*******************/
/***********			Phone  : 0982 259 245    						*******************
/***********			Project: CyberSoftMedicine - CyberCorp Inc. 	*******************/
/***********			Date   : 9:51 09/04/2017						*******************/
/******************************************************************************************/
#include "NeuralLayer.h"


#ifndef NNetwork
#define NNetwork

NeuralLayer::NeuralLayer(int nI, int nH, int nO) : numInputLayer(nI), numHiddenLayer(nH), numOutputLayer(nO)
{
	//Create Array
	inputLayer = new (double[nI + 1]);
	hiddenLayer = new (double[nH + 1]);
	outputLayer = new (double[nO + 1]);
	weightInputToHidden = new(double*[nI + 1]);
	weightHiddenToOutput = new(double*[nH + 1]);


	//Init All to 0
	for (int i = 0; i < nI; i++) inputLayer[i] = 0;
	for (int i = 0; i < nH; i++) hiddenLayer[i] = 0;
	for (int i = 0; i < nO; i++) outputLayer[i] = 0;
	//Init Bias to -1
	inputLayer[nI] = -1;
	hiddenLayer[nH] = -1;
	outputLayer[nO] = -1;
	//
	for (int i = 0; i <= nI; i++)
	{
		weightInputToHidden[i] = new (double[nH]);
		for (int j = 0; j < nH; j++) weightInputToHidden[i][j] = 0;
	}
	for (int i = 0; i <= nH; i++)
	{
		weightHiddenToOutput[i] = new (double[nO]);
		for (int j = 0; j < nO; j++) weightHiddenToOutput[i][j] = 0;
	}

	initWeights();
}

NeuralLayer::~NeuralLayer()
{
	delete[] inputLayer;
	delete[] outputLayer;
	delete[] hiddenLayer;
	for (int i = 0; i < numInputLayer; i++) delete[] weightInputToHidden[i];
	for (int i = 0; i < numOutputLayer; i++) delete[] weightHiddenToOutput[i];
	delete[] weightHiddenToOutput;
	delete[] weightInputToHidden;

}

int * NeuralLayer::feedForwardPattern(double * pattern)
{
	feedForward(pattern);
	int* results = new int[numOutputLayer];
	for (int i = 0; i < numOutputLayer; i++) results[i] = transferOutput(outputLayer[i]);
	return results;
}

void NeuralLayer::feedForward1(double * pattern)
{//ERROR
	for (int i = 0; i < numInputLayer; i++) {
		inputLayer[i] = pattern[i];
	}
	for (int j = 0; j < numHiddenLayer; j++) {

		hiddenLayer[j] = 0;

		for(int i=0; i <= numInputLayer; i++){
			hiddenLayer[j] += inputLayer[i] * weightInputToHidden[i][j];
			hiddenLayer[j] = activationFunction(hiddenLayer[j]);
		}
	}
	for (int k = 0; k < numOutputLayer; k++)
	{
		outputLayer[k] = 0;
		for (int j = 0; j <= numHiddenLayer; j++) { outputLayer[k] += hiddenLayer[j] * weightHiddenToOutput[j][k]; }
		outputLayer[k] = activationFunction(outputLayer[k]);
	}
}
void NeuralLayer::feedForward(double * pattern)
{
	//set input neurons to input values
	for (int i = 0; i < numInputLayer; i++) inputLayer[i] = pattern[i];

	//Calculate Hidden Layer values - include bias neuron
	//--------------------------------------------------------------------------------------------------------
	for (int j = 0; j < numHiddenLayer; j++)
	{
		//clear value
		hiddenLayer[j] = 0;

		//get weighted sum of pattern and bias neuron
		for (int i = 0; i <= numInputLayer; i++) hiddenLayer[j] += inputLayer[i] * weightInputToHidden[i][j];

		//set to result of sigmoid
		hiddenLayer[j] = activationFunction(hiddenLayer[j]);
	}

	//Calculating Output Layer values - include bias neuron
	//--------------------------------------------------------------------------------------------------------
	for (int k = 0; k < numOutputLayer; k++)
	{
		//clear value
		outputLayer[k] = 0;

		//get weighted sum of pattern and bias neuron
		for (int j = 0; j <= numHiddenLayer; j++) outputLayer[k] += hiddenLayer[j] * weightHiddenToOutput[j][k];

		//set to result of sigmoid
		outputLayer[k] = activationFunction(outputLayer[k]);
	}
}
double NeuralLayer::getMSE(std::vector<DataEntry*>& set)
{
	double mse = 0;

	//for every training input array
	for (int tp = 0; tp < (int)set.size(); tp++)
	{
		//feed inputs through network and backpropagate errors
		feedForward(set[tp]->pattern);

		//check all outputs against desired output values
		for (int k = 0; k < numOutputLayer; k++)
		{
			//sum all the MSEs together
			mse += pow((outputLayer[k] - set[tp]->target[k]), 2);
		}

	}//end for

	 //calculate error and return as percentage
	return mse / (numOutputLayer * set.size());
}

double NeuralLayer::getAccuracy(std::vector<DataEntry*>& set)
{
	double incorrectResults = 0;

	//for every training input array
	for (int tp = 0; tp < (int)set.size(); tp++)
	{
		//feed inputs through network and backpropagate errors
		feedForward(set[tp]->pattern);

		//correct pattern flag
		bool correctResult = true;

		//check all outputs against desired output values
		for (int k = 0; k < numOutputLayer; k++)
		{
			//set flag to false if desired and output differ
			if (transferOutput(outputLayer[k]) != set[tp]->target[k]) correctResult = false;
		}

		//inc training error for a incorrect result
		if (!correctResult) incorrectResults++;

	}//end for

	 //calculate error and return as percentage
	return 100 - (incorrectResults / set.size() * 100);
}

void NeuralLayer::initWeights()
{
	//set range
	double rH = 1/sqrt( (double) numInputLayer);
	double rO = 1/sqrt( (double) numHiddenLayer);
	
	//set weights between input and hidden 		
	//--------------------------------------------------------------------------------------------------------
	for(int i = 0; i <= numInputLayer; i++)
	{		
		for(int j = 0; j < numHiddenLayer; j++) 
		{
			//set weights to random values
			weightInputToHidden[i][j] = ( ( (double)(rand()%100)+1)/100  * 2 * rH ) - rH;			
		}
	}
	
	//set weights between input and hidden
	//--------------------------------------------------------------------------------------------------------
	for(int i = 0; i <= numHiddenLayer; i++)
	{		
		for(int j = 0; j < numOutputLayer; j++) 
		{
			//set weights to random values
			weightHiddenToOutput[i][j] = ( ( (double)(rand()%100)+1)/100 * 2 * rO ) - rO;
		}
	}
}

inline double NeuralLayer::activationFunction(double x)
{
	// Use inline Function for use mutiple times. Make speed faster
	return 1 / (1 + exp(-x));
}


inline int NeuralLayer::transferOutput(double x)
{
	if (x < 0.1) return 0;
	else if (x > 0.9) return 1;
	else return -1;
}

#endif
