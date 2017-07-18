/******************************************************************************************/
/***********			Author : Tran Minh Duc (Snowdence)				*******************/
/***********			Email  : Snowdence2911@gmail.com				*******************/
/***********			Phone  : 0982 259 245    						*******************
/***********			Project: CyberSoftMedicine - CyberCorp Inc. 	*******************/
/***********			Date   : 9:51 09/04/2017						*******************/
/***********			Compile: Visual Studio 2017						*******************/
/******************************************************************************************/
#include "NeuralLayer.h"


#ifndef NNetwork
#define NNetwork

NeuralLayer::NeuralLayer(int nI, int nH, int nO) : numInputLayer(nI), numHiddenLayer(nH), numOutputLayer(nO)
{
	//Create Array For Neural Architect
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
	//Release Memory 
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
	// Feed forward 
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

	//Calculate hidden value + BIAS 
	//--------------------------------------------------------------------------------------------------------
	for (int j = 0; j < numHiddenLayer; j++)
	{
		hiddenLayer[j] = 0;

		//get sum of all value
		for (int i = 0; i <= numInputLayer; i++) hiddenLayer[j] += inputLayer[i] * weightInputToHidden[i][j];

		//transfer to sigmoid function
		hiddenLayer[j] = activationFunction(hiddenLayer[j]);
	}

	// The same with output
	//--------------------------------------------------------------------------------------------------------
	for (int k = 0; k < numOutputLayer; k++)
	{
	
		outputLayer[k] = 0;

		//get sum all value
		for (int j = 0; j <= numHiddenLayer; j++) outputLayer[k] += hiddenLayer[j] * weightHiddenToOutput[j][k];

		//transfer result to sigmoid
		outputLayer[k] = activationFunction(outputLayer[k]);
	}
}
//binh phuong loi theo ham binh phuong
double NeuralLayer::getMSE(std::vector<DataEntry*>& set)
{
	double mse = 0;

	// Get all Error 
	for (int tp = 0; tp < (int)set.size(); tp++)
	{
		// Feed Forward System
		feedForward(set[tp]->pattern);

		// calculate Error square
		for (int k = 0; k < numOutputLayer; k++)
		{
			//sum all  ERRROR
			mse += pow((outputLayer[k] - set[tp]->target[k]), 2);
		}

	}

	 // Calculate percent of Error
	return mse / (numOutputLayer * set.size());
}
// binh phuong loi theo ket qua
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

		// check output desired and output cal
		for (int k = 0; k < numOutputLayer; k++)
		{
			//set flag to false if desired and output differ
			if (transferOutput(outputLayer[k]) != set[tp]->target[k]) correctResult = false;
		}

		// increase error
		if (!correctResult) incorrectResults++;

	}

	 //return percent
	return 100 - (incorrectResults / set.size() * 100);
}

void NeuralLayer::initWeights()
{
	// Constructure
	double rH = 1/sqrt( (double) numInputLayer);
	double rO = 1/sqrt( (double) numHiddenLayer);
	
	//INPUT VS HIDDEN
	//--------------------------------------------------------------------------------------------------------
	for(int i = 0; i <= numInputLayer; i++)
	{		
		for(int j = 0; j < numHiddenLayer; j++) 
		{
			// Init Radom
			weightInputToHidden[i][j] = ( ( (double)(rand()%100)+1)/100  * 2 * rH ) - rH;			
		}
	}
	
	//SET HIDEN OUTPUT
	//--------------------------------------------------------------------------------------------------------
	for(int i = 0; i <= numHiddenLayer; i++)
	{		
		for(int j = 0; j < numOutputLayer; j++) 
		{
			// The same
			weightHiddenToOutput[i][j] = ( ( (double)(rand()%100)+1)/100 * 2 * rO ) - rO;
		}
	}
}

inline double NeuralLayer::activationFunction(double x)
{
	// SigMoid Function. Inline is useful for performance
	return 1 / (1 + exp(-x));
}


inline int NeuralLayer::transferOutput(double x)
{
	//convert into 0 vs 1 with arrange
	if (x < 0.1) return 0;
	else if (x > 0.9) return 1;
	else return -1;
}


bool NeuralLayer::saveWeights(char* filename)
{
	// open
	std::fstream outputFile;
	outputFile.open(filename, std::ios::out);

	if (outputFile.is_open())
	{
		outputFile.precision(50);

		//Output 
		for (int i = 0; i <= numInputLayer; i++)
		{
			for (int j = 0; j < numHiddenLayer; j++)
			{
				outputFile << weightInputToHidden[i][j] << ",";
			}
		}

		for (int i = 0; i <= numHiddenLayer; i++)
		{
			for (int j = 0; j < numOutputLayer; j++)
			{
				outputFile << weightHiddenToOutput[i][j];
				if (i * numOutputLayer + j + 1 != (numHiddenLayer + 1) * numOutputLayer) outputFile << ",";
			}
		}

	
		std::cout << std::endl << "[CyBerMedicine] All Neurons has been save to File:  '" << filename << "'" << std::endl;
		std::cout << std::endl << "[CyBerMedicine] Tat ca kinh nghiem duoc luu o File:  '" << filename << "'" << std::endl;
		outputFile.close();

		return true;
	}
	else
	{
		std::cout << std::endl << "ERROR: [ '" << filename << "'] couldn't save. Please check file again " << std::endl;
		return false;
	}
}
bool NeuralLayer::loadWeights(char* filename)
{
	//open file for reading
	std::fstream inputFile;
	inputFile.open(filename, std::ios::in);
	if (inputFile.is_open())
	{
		std::vector<double> weights;
		std::string line = "";

		// Read all line 
		while (!inputFile.eof())
		{
			getline(inputFile, line);

			// Preocess with structure "value,value"
			if (line.length() > 2)
			{
				//INPUT value
				char* cstr = new char[line.size() + 1];
				char* t;
				strcpy_s(cstr, line.size() + 1, line.c_str());


				//This state 'll process split with , character
				int i = 0;
				char* nextToken = NULL;
				t = strtok_s(cstr, ",", &nextToken);

				while (t != NULL)
				{
					weights.push_back(atof(t));

					// move all , 
					t = strtok_s(NULL, ",", &nextToken);
					i++;
				}

				//free memory is very important
				delete[] cstr;
			}
		}

		// Incorrect Number
		if (weights.size() != ((numInputLayer + 1) * numHiddenLayer + (numHiddenLayer + 1) * numOutputLayer))
		{
			std::cout << std::endl << "ERROR number of neural network. Please check IN, HIDDEN, OUT: " << filename << std::endl;

			//close file
			inputFile.close();

			return false;
		}
		else
		{
			//init weights
			int pos = 0;

			for (int i = 0; i <= numInputLayer; i++)
			{
				for (int j = 0; j < numHiddenLayer; j++)
				{
					weightInputToHidden[i][j] = weights[pos++];
				}
			}

			for (int i = 0; i <= numHiddenLayer; i++)
			{
				for (int j = 0; j < numOutputLayer; j++)
				{
					weightHiddenToOutput[i][j] = weights[pos++];
				}
			}

			// Success
			std::cout << std::endl << "[CyBerMedicine] Load successful - Tai Thanh cong tap kinh nghiem tu file . '" << filename << "'" << std::endl;

			//close file
			inputFile.close();

			return true;
		}
	}
	else
	{
		std::cout << std::endl << "Weights file has occured an error. Please check : [ '" << filename << "'] : " << std::endl;
		return false;
	}
}
#endif
