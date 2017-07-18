/******************************************************************************************/
/***********			Author : Tran Minh Duc (Snowdence)				*******************/
/***********			Email  : Snowdence2911@gmail.com				*******************/
/***********			Phone  : 0982 259 245    						*******************
/***********			Project: CyberSoftMedicine - CyberCorp Inc. 	*******************/
/***********			Date   : 9:51 09/04/2017						*******************/
/******************************************************************************************/
#pragma once
#include <iostream>
#include <math.h>
#include <fstream>
#include <vector>
#include "SupportFile.h"
class NeuralProcess;
enum { LOGISTIC, TANH, RELu};
class NeuralLayer
{
private:
	friend NeuralProcess;
	int numInputLayer, numHiddenLayer, numOutputLayer;
	double* inputLayer, *hiddenLayer, *outputLayer;
	double** weightInputToHidden, **weightHiddenToOutput;
public:
	NeuralLayer(int nI, int nH, int nO);
	~NeuralLayer();
	bool loadOldWeight(char * fName);
	void saveNewWeight(char * fName);
	int* feedForwardPattern(double* pattern);
	void feedForward(double* pattern);
	void feedForward1(double* pattern);
	double getMSE(std::vector<DataEntry*>&set);
	double getAccuracy(std::vector<DataEntry*>&dataset);
private:
	void initWeights();
	inline double activationFunction(double x);
	inline int transferOutput(double x);
};

/*

}
*/