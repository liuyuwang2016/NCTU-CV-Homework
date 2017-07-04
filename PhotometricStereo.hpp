#pragma once
#pragma once
#ifndef HW1_TOOL_HPP
#define HW1_TOOL_HPP


#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

namespace PhotometricStereo {

	std::vector<std::string> txtfile(std::string file_path);

	std::vector<std::vector<float>> readLightSource(std::string file_path);

	void printMat(cv::Mat matrix);

	cv::Mat pseudoInverse(cv::Mat input);

}

#endif #pragma once
#pragma once
