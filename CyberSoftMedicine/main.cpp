/******************************************************************************************/
/***********			Author : Tran Minh Duc (Snowdence)				*******************/
/***********			Email  : Snowdence2911@gmail.com				*******************/
/***********			Phone  : 0982 259 245    						*******************
/***********			Project: CyberSoftMedicine - CyberCorp Inc. 	*******************/
/***********			Date   : 9:51 09/04/2017						*******************/
/***********			Compile: Visual Studio 2017						*******************/
/******************************************************************************************/
#include <iostream>
#include <ctime>
#include <string>

#include "SupportFile.h"
#include "NeuralLayer.h"
#include "NeuralProcess.h"

using namespace std;
SupportFile sf;

/// <summary>  Admin </summary>
/// <param name="inputNeurons">Số nơ ron đầu vào.</param>
/// <param name="hiddenNeurons">Số nơ ron lớp ẩn.</param>
/// <param name="outputNeurons">Số nơ ron đầu ra.</param>
/// <param name="learningRate">Tốc độ học .</param>
/// <param name="momentum">Hằng số momen mặc định 0.9 .</param>
void trainExperience(int inputNeurons, int hiddenNeurons, int outputNeurons, float learningRate, float momentum, int epochs, char* fileTraining, char* saveWeights)
{
	sf.initFileData(inputNeurons, outputNeurons);
	if(sf.loadFileData(fileTraining) == false)
	{
		cout << "Co loi xay ra voi file dao tao. " << endl;
		return;
	}

	NeuralLayer nn(inputNeurons, hiddenNeurons, outputNeurons);
	NeuralProcess nT(&nn);
	nT.setTrainingParameters(learningRate, momentum, false);
	nT.setStoppingConditions(epochs, 95);
	nT.trainNetwork(sf.getTrainingDataSet());
	nn.saveWeights(saveWeights);
}
void useExperience(int inputNeurons, int hiddenNeurons, int outputNeurons, char* fileTest, char* fileWeights)
{
	sf.initFileData(inputNeurons, outputNeurons);
	sf.loadFileData(fileTest);
	NeuralLayer nn(inputNeurons, hiddenNeurons, outputNeurons);
	
	if (!nn.loadWeights(fileWeights))
	{
		cout << "Co Loi Xay Ra Voi File Kinh Nghiem " << endl;
		return;
	}
	NeuralProcess nT(&nn);
	cout << "\n\n\n" << endl;
	cout << "Ket qua tinh toan duoc la : [" << ((*nn.feedForwardPattern(sf.getTrainingDataSet()->trainingSet[0]->pattern) == 0) ? "Lanh tinh" : "Ac tinh") << "]" << endl;
	

}
void display()
{
	cout << "************************************* " << endl;
	cout << "------------------------------------- " << endl;
	cout << "1. Dao tao mang neuron" << endl;
	cout << "2. Su dung kinh nghiem da dao tao " << endl;
	cout << "3. Thoat chuong trinh " << endl;
	cout << "------------------------------------- " << endl;
	cout << "************************************* " << endl;
	cout << "" << endl;
	cout << "Nhap lua chon cua ban : ";
}
void train()
{
	string file = "";
	string w = "";
	int thehe = 1000;
	cout << "Nhap file dao tao: ";
	cin >> file;
	cout << "Nhap ten file de luu kinh nghiem sau khi dao tao: ";
	cin >> w;
	char tab1[1024];
	char tab2[1024];
	trainExperience(9, 16, 1, 0.001, 0.9, thehe, strcpy(tab1, file.c_str()), strcpy(tab2, w.c_str()) );
}
void use() {
	string file = "";
	string exp = "";
	cout << "Nhap ten file can de du doan: ";
	cin >> file;
	cout << "Nhap ten file kinh nghiem da luu: ";
	cin >> exp;
	char temp[1024]; 
	char tempp[1024];
	useExperience(9, 16, 1,strcpy(temp, file.c_str()) , strcpy(tempp, exp.c_str()) );
}
void main()
{
	int option = 0;
LABEL:
	display();
	cin >> option;
	switch (option)
	{
	case 1:
		train();
		break;
	case 2:
		use();
		break;
	case 3:
		goto END;
		break;
	default:
		cout << "Gia tri ban vua nhap khong dung" << endl;
		break;
	}
	cout << "\n\n\n" << endl;
goto LABEL;
	//trainExperience(9, 16, 1, 0.001, 0.9, 100, "cancer.txt", "weights.txt");
	//useExperience(9, 16, 1, "test.txt", "weights.txt");
END:
	system("exit");
}