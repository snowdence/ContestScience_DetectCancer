/******************************************************************************************/
/***********			Author : Tran Minh Duc (Snowdence)				*******************/
/***********			Email  : Snowdence2911@gmail.com				*******************/
/***********			Phone  : 0982 259 245    						*******************
/***********			Project: CyberSoftMedicine - CyberCorp Inc. 	*******************/
/***********			Date   : 9:51 09/04/2017						*******************/
/***********			Compile: Visual Studio 2017						*******************/
/******************************************************************************************/
#pragma once
#include <iostream> // DEBUG
#include <vector>
#include <fstream> // Library require for workig with File
#include <string> // Process using String 
#include <math.h>   
#include <algorithm> 
class DataEntry {
public:
	double* pattern; // Pointer to dataset int[] array
	double* target; // pointer to output in training int[] array
public:
	DataEntry(double*p, double*t) : pattern(p), target(t) {}; // Constructor to init data
	~DataEntry() { delete[] pattern; delete[] target; } // Destructor to delete and release memory
};
class NeuralDataPackage {
public:
	std::vector<DataEntry*> trainingSet;
	std::vector<DataEntry*> generalizationSet;
	std::vector<DataEntry*> validationSet;

	NeuralDataPackage() {}

	void clear()
	{
		trainingSet.clear();
		generalizationSet.clear();
		validationSet.clear();
	}
};
class SupportFile
{
private:
	NeuralDataPackage dataPackage; // Oject to save Entry
	int	neuralInput, neuralTarget; // to init  Array in DataEntry
	std::vector<DataEntry* > trainingDataSet;
	int numberTrainingSet, pEndIndex; // to mark the element of data set
public:
	SupportFile() : numberTrainingSet(-1) {  };
	~SupportFile(); // Can be removed. It will be use for release memory
	int getNumTraining();
	void initFileData(int numberInput, int numberOutput);
	bool loadFileData(char* fname);
	NeuralDataPackage* getTrainingDataSet();
private:
	void parseLine(std::string &line);
	void releaseData();
	void createDataSet();
};