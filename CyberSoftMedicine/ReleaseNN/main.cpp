/******************************************************************************************/
/***********			Author : Tran Minh Duc (Snowdence)				*******************/
/***********			Email  : Snowdence2911@gmail.com				*******************/
/***********			Phone  : 0982 259 245    						*******************
/***********			Project: CyberSoftMedicine - CyberCorp Inc. 	*******************/
/***********			Date   : 9:51 09/04/2017						*******************/
/******************************************************************************************/
#include <iostream>
#include <ctime>
#include <string>

#include "SupportFile.h"
#include "NeuralLayer.h"
#include "NeuralProcess.h"

using namespace std;
SupportFile sf;
void main()
{
	srand((unsigned int)time(0));
	sf.initFileData(9, 1);
	sf.loadFileData("test.txt");
	NeuralLayer nn(9, 10, 1);
	NeuralProcess nT(&nn);
	nT.setTrainingParameters(0.001, 0.9 , false);
	nT.setStoppingConditions(1000, 90);
	nT.trainNetwork(sf.getTrainingDataSet());
	system("pause");
}