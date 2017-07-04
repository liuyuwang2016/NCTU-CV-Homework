#ifdef _MSC_VER
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif //!_CRT_SECURE_NO_WARNINGS
#endif //_MSC_VER

#include <cstdlib>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "PhotometricStereo.hpp"


/* Read .txt file */
std::vector<std::string> PhotometricStereo::txtfile(std::string filePath)
{
	std::vector<std::string> stream;
	std::ifstream file;
	//c_str: Indicates that the function returns a C-style character string
	file.open(filePath.c_str());

	std::stringstream readStreams;
	readStreams << file.rdbuf();
	file.close();

	auto bufferChars = readStreams.str();
	std::string tempChars;
	for (auto letter : bufferChars) {
		if (letter == '\n')
			stream.push_back(std::move(tempChars));
		else
			tempChars += letter;
	}
	return stream;
}
/* Read LightSource.txt */
std::vector<std::vector<float>> PhotometricStereo::readLightSource(std::string filePath)
{

	std::cout << filePath << '\n';
	auto lightData = PhotometricStereo::txtfile(filePath);

	std::vector<std::vector<float>> lightSource;

	//Allocate space for at least n elements
	lightSource.reserve(lightData.size());

	for (auto line : lightData) {
		std::cout << '\t';
		if (line.find("(") < line.size()) {
			line.erase(line.begin(), line.begin() + line.find("(") + 1);
			std::vector<float> temp;
			temp.reserve(3);
			while (line.find(",") < line.size()) {
				std::string number(line, 0, line.find(","));
				temp.push_back(std::stof(number));
				line.erase(line.begin(), line.begin() + line.find(",") + 1);
			}
			if (line.find(")") < line.size()) {
				line.erase(line.find(")"));
				temp.push_back(std::stof(line));
			}
			std::cout << temp[0] << '\t' << temp[1] << '\t' << temp[2] << '\n';
			std::vector<float> lightVector{ temp[0], temp[1], temp[2] };
			lightSource.push_back(std::move(lightVector));
		}
	}
	std::cout << "end of file \n";
	return lightSource;
}

void PhotometricStereo::printMat(cv::Mat matrix) {

	for (auto rowIndex = 0; rowIndex < matrix.rows; rowIndex++) {
		std::cout << '\t';
		for (auto colIndex = 0; colIndex < matrix.cols; colIndex++) {
			std::cout << matrix.at<float>(rowIndex, colIndex) << '\t';
		}
		std::cout << '\n';
	}

};
/* Pseudo Inverse */
cv::Mat PhotometricStereo::pseudoInverse(cv::Mat input) {

#define NAME(x) (#x)
	bool wideMatrix = input.rows < input.cols;

	auto smallSide = std::min(input.rows, input.cols);
	auto largeSide = std::max(input.rows, input.cols);
	std::cout << NAME(input) << " : \n";
	printMat(input);

	auto wideMat = cv::Mat(largeSide, smallSide, CV_32F);
	auto longMat = cv::Mat(smallSide, largeSide, CV_32F);

	if (wideMatrix) {
		wideMat = input;
		cv::transpose(input, longMat);
	}
	else {
		cv::transpose(input, wideMat);
		longMat = input;
	}
	std::cout << NAME(wideMat) << " : \n";
	printMat(wideMat);


	// L * LT
	auto rank = std::min(input.rows, input.cols);
	cv::Mat LLT(rank, rank, CV_32F);
	LLT = wideMat * longMat;
	std::cout << NAME(LLT) << " : \n";
	printMat(LLT);

	cv::Mat invLLT(rank, rank, CV_32F);
	invLLT = LLT.inv();
	std::cout << NAME(invLLT) << " : \n";
	printMat(invLLT);

	auto pseudoInv = cv::Mat(input.cols, input.rows, CV_32F);
	if (wideMatrix) {
		pseudoInv = longMat * invLLT;
	}
	else {
		pseudoInv = invLLT * wideMat;
	}
	std::cout << NAME(pseudoInv) << " : \n";
	printMat(pseudoInv);
	return pseudoInv;
}
/* Calculate Gradients */

int main()
{
	const int picNums = 6;

	// 1: Get light from .txt file
	std::string Light_file{ "test/bunny_self/LightSource.txt" };
	auto LightVector = PhotometricStereo::readLightSource(Light_file);

	// 2: Use lightsource file to get source vector S1
	auto LightMatric = cv::Mat(LightVector.size(), LightVector[0].size(), CV_32FC1);
	for (auto row = 0; row < LightVector.size(); ++row) {
		for (auto col = 0; col < LightVector[0].size(); ++col) {
			LightMatric.at<float>(row, col) = LightVector[row][col];
		}
	}

	// 3: Get pseudo inverse of light matrix
	auto lightInv = cv::Mat(LightMatric.cols, LightMatric.rows, CV_32FC1);
	lightInv = PhotometricStereo::pseudoInverse(LightMatric);

	// 4: Reading our picture datas
	std::vector<std::string> picturesPath;
	// Allocate space for at least n elements
	picturesPath.reserve(picNums);
	std::string nowPath{ "test/bunny_self/pic" };
	for (auto i = 0; i < picNums; ++i) {
		std::string temp{ nowPath + std::to_string(i + 1) + ".bmp" };
		/*對push_back的調用在container尾部創建一個新的元素，將每個元素放到末尾*/
		picturesPath.push_back(std::move(temp));
	}

	std::vector<cv::Mat> images;
	images.reserve(picNums);
	for (auto imgPath : picturesPath) {
		// use emplace not push_back
		// emplace函數在容器中直接構造元素
		images.emplace_back(cv::imread(imgPath.c_str(), cv::IMREAD_GRAYSCALE));
	}


	// 5: Emerge all data into one big matrix: I(x,y)
	auto objMat = cv::Mat(picNums, images[0].cols * images[0].rows, CV_32FC1);
	for (auto i = 0; i < picNums; ++i) {
		for (auto rowIndex = 0; rowIndex < images[0].rows; ++rowIndex) {
			for (auto colIndex = 0; colIndex < images[0].cols; ++colIndex) {
				auto temp = static_cast<float>(images[i].at<uchar>(rowIndex, colIndex));
				objMat.at<float>(i, images[0].cols * rowIndex + colIndex) = std::move(temp);
			}
		}
	}
	std::cout << "objMat(" << objMat.rows << ", " << objMat.cols << ")\n";
	//std::cout << objMat << std::endl;
	/*Calculate normals by inverse * input picture*/
	auto normals = cv::Mat(lightInv.rows, objMat.rows, CV_32F);
	normals = lightInv * objMat;
	// 6: Calculate normalx, normaly, normalz 
	cv::Mat u_normalx = cv::Mat::zeros(images[0].rows, images[0].cols, CV_32FC1);
	cv::Mat u_normaly = cv::Mat::zeros(images[0].rows, images[0].cols, CV_32FC1);
	cv::Mat u_normalz = cv::Mat::zeros(images[0].rows, images[0].cols, CV_32FC1);
	cv::Mat gradientMapx = cv::Mat::zeros(images[0].rows, images[0].cols, CV_32FC1);
	cv::Mat gradientMapy = cv::Mat::zeros(images[0].rows, images[0].cols, CV_32FC1);
	int k = 0;
	for (int rowIndex = 0; rowIndex < images[0].rows; ++rowIndex) {
		for (int colIndex = 0; colIndex < images[0].cols; ++colIndex) {
			float normalx = normals.at<float>(0, k);
			float normaly = normals.at<float>(1, k);
			float normalz = normals.at<float>(2, k);

			float L2 = sqrt(normalx*normalx + normaly*normaly + normalz*normalz);
			if (L2 != 0) {
				u_normalx.at<float>(rowIndex, colIndex) = normalx / L2;
				u_normaly.at<float>(rowIndex, colIndex) = normaly / L2;
				u_normalz.at<float>(rowIndex, colIndex) = normalz / L2;
			}
			else {
				u_normalx.at<float>(rowIndex, colIndex) = 0;
				u_normaly.at<float>(rowIndex, colIndex) = 0;
				u_normalz.at<float>(rowIndex, colIndex) = 0;
			}
			if (u_normalz.at<float>(rowIndex, colIndex) != 0) {
				gradientMapx.at<float>(rowIndex, colIndex) = (-u_normalx.at<float>(rowIndex, colIndex) / u_normalz.at<float>(rowIndex, colIndex));
				gradientMapy.at<float>(rowIndex, colIndex) = (-u_normaly.at<float>(rowIndex, colIndex) / u_normalz.at<float>(rowIndex, colIndex));
			}
			else {
				gradientMapx.at<float>(rowIndex, colIndex) = 0;
				gradientMapy.at<float>(rowIndex, colIndex) = 0;
			}
			k++;

			//std::cout << "normals(" << normals.rows << ", " << normals.cols << ")\n";

			/*float normal1 = normals.at<float>(0);
			float normal2 = normals.at<float>(1);*/
			//float normal3 = normals.at<float>(2);

			//std::cout << normalMapy << std::endl;
		}
	}
	//std::ofstream file1;
	//file1.open("gradient.txt");
	//for (i = 0; i < images[0].rows; i++)
	//{
	//	for (j = 0; j < images[0].cols; j++)
	//	{
	//		file1 << gradientMapx.at<float>(i, j)[0] << " " << gradient_.at<Vec3f>(i, j)[1] << " " << gradient_.at<Vec3f>(i, j)[2] << endl;
	//	}
	//}
	//file1.close();
	//return true;


	// 7: to calculate the picture 
	/*7.A To calculate the picture from left to right*/
	float sumx = 0;
	float sumy = 0;
	float recordy = 0;
	//float sumtotal = 0;
	cv::Mat heightMap = cv::Mat::zeros(images[0].rows, images[0].cols, CV_32FC1);
	for (int rowIndex = 0; rowIndex < images[0].rows; rowIndex++) {
		for (int colIndex = 0; colIndex < images[0].cols; colIndex++) {
			sumx = sumx + gradientMapx.at<float>(rowIndex, colIndex);
			heightMap.at<float>(rowIndex, colIndex) = sumx;
			//sumy = sumy + gradientMapy.at<float>(rowIndex, colIndex);
			//heightMap.at<float>(rowIndex, colIndex) = (sumx + sumy) / 2;
		}
		sumx = 0;
		//sumy = 0;
	}
	/*7.B Other method to calculate the picture from left to right*/
	//for (auto rowIndex = 0; rowIndex < images[0].rows; ++rowIndex) {
	//	recordy = rowIndex;
	//	if (recordy != 0) {
	//		sumy = sumy + gradientMapy.at<float>(rowIndex, 0);
	//		sumx = sumx + sumy;
	//		//cout << "sumx1\n " << sumx << endl;
	//	}
	//	else {
	//	}
	//       for (auto colIndex = 0; colIndex < images[0].cols; ++colIndex) {
	//		sumx = sumx + gradientMapx.at<float>(rowIndex, colIndex);
	//		heightMap.at<float>(rowIndex, colIndex) = sumx;
	//	}
	//	sumx = 0;
	//}
	

	//int i, j;
	//double maxErr;
	//double sigma = 1.99;
	//maxErr = 1e-4;
	//int maxCount = 5000;
	/*7.C Other method to calculate the picture */
	//ofstream file3;
	//file3.open("err.txt");
	//for (int k = 0; k < maxCount; k++)
	//{
	//	double err = 0.0;
	//	for (i = 0; i < height_; i++)
	//	{
	//		for (j = width_ - 1; j >= 0; j--)
	//		{
	//			if (heightmap.at<float>(i, j))
	//			{
	//				double buf = 0.0;
	//				double temperr = 0.0;
	//				double b = (gradient_.at<Vec3f>(i, j + 1)[0] - gradient_.at<Vec3f>(i, j)[0]) + (gradient_.at<Vec3f>(i - 1, j)[1] - gradient_.at<Vec3f>(i, j)[1]);
	//				if (gradientMapx.at<float>(i - 1, j))//(X,Y+1)
	//				{
	//					buf += heightMap.at<float>(i - 1, j);
	//				}
	//				if (gradientMapx.at<float>(i, j - 1))//(X-1,Y)
	//				{
	//					buf += heightMap.at<float>(i, j - 1);
	//				}
	//				buf -= 4 * depth_.at<float>(i, j);
	//				if (gradientMapx.at<float>(i, j + 1))//(X+1,Y)
	//				{
	//					buf += heightMap.at<float>(i, j + 1);
	//				}
	//				if (gradientMapx.at<float>(i + 1, j))//(X,Y-1)
	//				{
	//					buf += heightMap.at<float>(i + 1, j);
	//				}
	//				temperr = (-0.25) * (b - buf)*sigma;
	//				if (abs(temperr) > abs(err))
	//				{
	//					err = temperr;
	//				}
	//				depth_.at<float>(i, j) += temperr;
	//				file3<<err<<" ";
	//			}
	//		}

	//	}
	//	file3 << err<<" ";
	//	if (abs(err) < maxErr)
	//	{
	//		cout << "The interation is accomplished in" << k << "times" << endl;
	//		return true;
	//		break;
	//	}
	//}
	//file3.close();

	std::ofstream file;
	file.open("bunny_self.ply");
	file << "ply" << std::endl;
	file << "format ascii 1.0" << std::endl;
	file << "comment alpha = 1.0" << std::endl;

	//file << "element vertex " << height_*width_ << std::endl;
	file << "element vertex " << images[0].cols * images[0].rows << std::endl;
	file << "property float x" << std::endl;
	file << "property float y" << std::endl;
	file << "property float z" << std::endl;
	file << "property uchar red " << std::endl;
	file << "property uchar green" << std::endl;
	file << "property uchar blue" << std::endl;
	file << "end_header" << std::endl;
	for (auto rowIndex = 0; rowIndex < images[0].rows; rowIndex++)
	{
		for (auto colIndex = 0; colIndex < images[0].cols; colIndex++)
		{
			//std::cout << heightMap.at<float>(rowIndex, colIndex) << " ";
			file << rowIndex << " " << colIndex << " " 
				 << heightMap.at<float>(rowIndex, colIndex) << " "
				 << 255 << " " << 255 << " " << 255 << std::endl;
		}
	}
	file.close();
	//cv::imshow("CV");
	cv::waitKey();
}
