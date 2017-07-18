/******************************************************************************************/
/***********			Author : Tran Minh Duc (Snowdence)				*******************/
/***********			Email  : Snowdence2911@gmail.com				*******************/
/***********			Phone  : 0982 259 245    						*******************
/***********			Project: CyberSoftMedicine - CyberCorp Inc. 	*******************/
/***********			Date   : 8:51 09/04/2017						*******************/
/***********			Compile: Visual Studio 2017						*******************/
/******************************************************************************************/
#include "SupportFile.h"
//#define DEBUG FALSE
SupportFile::~SupportFile()
{
	releaseData();
}


void SupportFile::initFileData(int numberInput, int numberOutput)
{
	neuralInput = numberInput;
	neuralTarget = numberOutput;
}

bool SupportFile::loadFileData(char * fname)
{
	releaseData();
	dataPackage.clear();
	std::fstream file;
	file.open(fname, std::ios::in);
	if (file.is_open())
	{
		std::string l = "";
		while (!file.eof())
		{
			std::getline(file, l);
			if(l.length() >= 3) parseLine(l);
			
		}
		
		//Radom code if needed
		//std::random_shuffle(trainingDataSet.begin(), trainingDataSet.end());
		pEndIndex = (int)(0.6 * trainingDataSet.size());
		int gSize = (int)(ceil(0.2 * trainingDataSet.size()));
		int vSize = (int)(trainingDataSet.size() - pEndIndex - gSize);
		for (int i = pEndIndex; i < pEndIndex + gSize; i++) dataPackage.generalizationSet.push_back(trainingDataSet[i]);
		//std::cout << trainingDataSet.size() << std::endl;
		//validation set
		for (int i = pEndIndex + gSize; i < (int)trainingDataSet.size(); i++) dataPackage.validationSet.push_back(trainingDataSet[i]);

#ifdef DEBUG1
		std::cout << "debug " << std::endl;
		//std::cout << trainingDataSet.size() << std::endl;
		for (int k = 0; k < trainingDataSet.size(); k++) {
			std::cout << " *********** k: " << k << std::endl;
			for (int i = 0; i < neuralInput; i++)
			{
				std::cout << *(trainingDataSet[k]->pattern + i) << std::endl;
			}
		}
#endif // DEBUG
		file.close();
		return true; 
	}
	else {
		std::cout << "ERROR when open [" << fname << " ]" << std::endl;
		return false;
	}
}
void SupportFile::createDataSet()
{
	if (pEndIndex == 0)
		pEndIndex = 1;
	for (int i = 0; i < pEndIndex; i++)
	{
		dataPackage.trainingSet.push_back(trainingDataSet[i]); 
	}
}
NeuralDataPackage * SupportFile::getTrainingDataSet()
{
	createDataSet();
	return &dataPackage;
}

void SupportFile::parseLine(std::string & l)
{
	// Algorithm use to parse 1,2,3,4,5 => [1,2,3,4] [5]
	double *pattern = new double[neuralInput];
	double *target = new double[neuralTarget];
	char* copyString = new char[l.size() + 1];
	char* token = "";
	int i = 0;
	char* nT = NULL;
	strcpy_s(copyString, l.size() + 1, l.c_str());
	token = strtok_s(copyString, ",", &nT);
	for (i = 0; (i < (neuralInput + neuralTarget)) && (token != NULL); i++) {
		if (i < neuralInput) {
			pattern[i] = atof(token); // Convert string to double 
		}
		else {
			target[i - neuralInput] = atof(token);
		}
		token = strtok_s(NULL, ",", &nT);
	}
	trainingDataSet.push_back(new DataEntry(pattern, target)); //Adding pointer to this
}

void SupportFile::releaseData() {
	for (int i = 0; i < (int)trainingDataSet.size(); i++) delete trainingDataSet[i];
	trainingDataSet.clear();
}

