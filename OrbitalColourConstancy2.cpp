// OrbitalColourConstancy2.cpp : This file contains the 'main' function. Program execution begins and ends there.
//This code is the implementation of several colour constancy algorithms by Tatsuya yatagawa
//(https://github.com/tatsy/ColorConstancy).
//Some functions have been modified such as cvtColor and imread functions to make it compatible with
//more current versions of opencv.
//The first part of the file is the clcnst.h header file, followed by the clcnst.cpp file and finally
//the algorithms.
//This file was run in visual studio 2019 with opencv 4.5.1. Installation of opencv in visual studio can
//be found easily online.
//When running this file, just uncomment the algorithm you wish to run. The output image will eventually
//be saved in the visual studio repos(within the folder of your project name)

//clcnst.h
#ifndef _CLCNST_H_
#define _CLCNST_H_

#include <opencv2/opencv.hpp>
#include <cassert>

#if defined(WIN32)		// MS Windows
#define IDAAPI __stdcall
#ifdef __DLL_EXPORT
#define __PORT __declspec(dllexport)
#else
#define __PORT __declspec(dllimport)
#endif
#else
#define __PORT
#endif

// Utility functions for color constancy projects
class clcnst {
private:
	static const int offset[4][2];
	static const float eps;

public:
	// Compute exp of cv::Mat.
	// Input and output can be the same instance.
	// Type of arguments must be CV_32F.
	__PORT static void exponential(cv::Mat& input, cv::Mat& output);

	// Compute log of cv::Mat.
	// Input and output can be the same instance.
	// Type of arguments must be CV_32F.
	__PORT static void logarithm(cv::Mat& input, cv::Mat& output);

	// Solve poisson equation using Gauss-Seldel method.
	__PORT static void gauss_seidel(cv::Mat& I, cv::Mat& L, int maxiter);

	// Apply Laplacian filter.
	// Input and output can be the same instance.
	// Type of arguments must be CV_32FC.
	__PORT static void laplacian(cv::Mat& input, cv::Mat& output);

	// Apply Gaussian filter.
	__PORT static void gaussian(cv::Mat& input, cv::Mat& output, float sigma, int ksize);

	// Apply thresholding operation.
	__PORT static void threshold(cv::Mat& input, cv::Mat& output, float threshold);

	// Normalize output range as the maximum value come to be 1.
	__PORT static void normalize(cv::Mat& input, cv::Mat& output);

	// Normalize output range into [lower, upper]
	__PORT static void normalize(cv::Mat& input, cv::Mat& output, float lower, float upper);

	// High emphasis filter
	__PORT static void hef(cv::Mat& input, cv::Mat& output, float lower, float upper, float threshold);
};

#endif
//clcnst.cpp
#define __DLL_EXPORT

#include <vector>
using namespace std;

const int clcnst::offset[4][2] = { {-1, 0}, {1, 0}, {0, -1}, {0, 1} };

const float clcnst::eps = 0.0001f;

__PORT void clcnst::exponential(cv::Mat& input, cv::Mat& output) {
	cv::Mat* i_ptr = &input;
	int width = i_ptr->cols;
	int height = i_ptr->rows;
	int channel = i_ptr->channels();

	cv::Mat* o_ptr = &output;
	if (i_ptr != o_ptr) {
		*o_ptr = cv::Mat(height, width, CV_MAKETYPE(CV_32F, channel));
	}

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			for (int c = 0; c < channel; c++) {
				o_ptr->at<float>(y, x * channel + c) = exp(i_ptr->at<float>(y, x * channel + c)) - clcnst::eps;
			}
		}
	}
}

__PORT void clcnst::logarithm(cv::Mat& input, cv::Mat& output) {
	cv::Mat* i_ptr = &input;
	int width = i_ptr->cols;
	int height = i_ptr->rows;
	int channel = i_ptr->channels();

	cv::Mat* o_ptr = &output;
	if (i_ptr != o_ptr) {
		*o_ptr = cv::Mat(height, width, CV_MAKETYPE(CV_32F, channel));
	}

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			for (int c = 0; c < channel; c++) {
				o_ptr->at<float>(y, x * channel + c) = log(i_ptr->at<float>(y, x * channel + c) + clcnst::eps);
			}
		}
	}
}

__PORT void clcnst::gauss_seidel(cv::Mat& I, cv::Mat& L, int maxiter) {
	int width = I.cols;
	int height = I.rows;
	int channel = I.channels();
	assert(width == L.cols && height == L.rows && channel == L.channels());

	while (maxiter--) {
		for (int c = 0; c < channel; c++) {
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					int count = 0;
					float sum = 0.0f;
					for (int i = 0; i < 4; i++) {
						int xx = x + clcnst::offset[i][0];
						int yy = y + clcnst::offset[i][1];
						if (xx >= 0 && yy >= 0 && xx < width && yy < height) {
							sum += I.at<float>(yy, xx * channel + c);
							count += 1;
						}
					}
					I.at<float>(y, x * channel + c) = (sum - L.at<float>(y, x * channel + c)) / (float)count;
				}
			}
		}
	}
}

__PORT void clcnst::laplacian(cv::Mat& input, cv::Mat& output) {
	cv::Mat* i_ptr = &input;
	int width = i_ptr->cols;
	int height = i_ptr->rows;
	int channel = i_ptr->channels();

	cv::Mat* o_ptr = &output;
	if (i_ptr != o_ptr) {
		*o_ptr = cv::Mat(height, width, CV_MAKETYPE(CV_32F, channel));
	}

	for (int c = 0; c < channel; c++) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int count = 0;
				float sum = 0.0f;
				for (int i = 0; i < 4; i++) {
					int xx = x + clcnst::offset[i][0];
					int yy = y + clcnst::offset[i][1];
					if (xx >= 0 && yy >= 0 && xx < width && yy < height) {
						count += 1;
						sum += i_ptr->at<float>(yy, xx * channel + c);
					}
				}
				o_ptr->at<float>(y, x * channel + c) = sum - (float)count * i_ptr->at<float>(y, x * channel + c);
			}
		}
	}
}

__PORT void clcnst::gaussian(cv::Mat& input, cv::Mat& output, float sigma, int ksize) {
	cv::Mat* i_ptr = &input;
	cv::Mat* o_ptr = &output;
	assert(i_ptr != o_ptr);

	int width = i_ptr->cols;
	int height = i_ptr->rows;
	int channel = i_ptr->channels();
	float s2 = 2.0f * sigma * sigma;

	vector<float> table(ksize + 1, 0.0f);
	for (int i = 0; i <= ksize; i++) {
		table[i] = exp(-(i * i) / s2);
	}

	*o_ptr = cv::Mat(height, width, CV_MAKETYPE(CV_32F, channel));
	for (int c = 0; c < channel; c++) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				float sum = 0.0f;
				float weight = 0.0f;
				for (int dy = -ksize; dy <= ksize; dy++) {
					for (int dx = -ksize; dx <= ksize; dx++) {
						int xx = x + dx;
						int yy = y + dy;
						if (xx >= 0 && yy >= 0 && xx < width && yy < height) {
							float w = table[abs(dx)] * table[abs(dy)];
							sum += i_ptr->at<float>(yy, xx * channel + c) * w;
							weight += w;
						}
					}
				}
				o_ptr->at<float>(y, x * channel + c) = weight != 0.0f ? sum / weight : 0.0f;
			}
		}
	}
}

__PORT void clcnst::threshold(cv::Mat& input, cv::Mat& output, float threshold) {
	cv::Mat* i_ptr = &input;
	int width = i_ptr->cols;
	int height = i_ptr->rows;
	int channel = i_ptr->channels();

	cv::Mat* o_ptr = &output;
	if (i_ptr != o_ptr) {
		*o_ptr = cv::Mat(height, width, CV_MAKETYPE(CV_32F, channel));
	}

	for (int c = 0; c < channel; c++) {
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				if (fabs(i_ptr->at<float>(y, x * channel + c)) < threshold) {
					o_ptr->at<float>(y, x * channel + c) = 0.0f;
				}
				else {
					o_ptr->at<float>(y, x * channel + c) = i_ptr->at<float>(y, x * channel + c);
				}
			}
		}
	}
}

__PORT void clcnst::normalize(cv::Mat& input, cv::Mat& output) {
	cv::Mat* i_ptr = &input;
	int width = i_ptr->cols;
	int height = i_ptr->rows;
	int channel = i_ptr->channels();

	cv::Mat* o_ptr = &output;
	if (i_ptr != o_ptr) {
		*o_ptr = cv::Mat(height, width, CV_MAKETYPE(CV_32F, channel));
	}

	for (int c = 0; c < channel; c++) {
		float maxval = -100.0f;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				if (maxval < i_ptr->at<float>(y, x * channel + c)) {
					maxval = i_ptr->at<float>(y, x * channel + c);
				}
			}
		}

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				o_ptr->at<float>(y, x * channel + c) = i_ptr->at<float>(y, x * channel + c) - maxval;
			}
		}
	}
}

__PORT void clcnst::normalize(cv::Mat& input, cv::Mat& output, float lower, float upper) {
	cv::Mat* i_ptr = &input;
	int width = i_ptr->cols;
	int height = i_ptr->rows;
	int channel = i_ptr->channels();

	cv::Mat* o_ptr = &output;
	if (i_ptr != o_ptr) {
		*o_ptr = cv::Mat(height, width, CV_MAKETYPE(CV_32F, channel));
	}

	float minval = 100000.0f;
	float maxval = -100000.0f;
	for (int c = 0; c < channel; c++) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				float value = i_ptr->at<float>(y, x * channel + c);
				minval = min(minval, value);
				maxval = max(maxval, value);
			}
		}
	}

	float ratio = (upper - lower) / (maxval - minval);
	for (int c = 0; c < channel; c++) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				float value = i_ptr->at<float>(y, x * channel + c);
				o_ptr->at<float>(y, x * channel + c) = (value - minval) * ratio + lower;
			}
		}
	}
}

__PORT void clcnst::hef(cv::Mat& input, cv::Mat& output, float lower, float upper, float threshold) {
	cv::Mat* i_ptr = &input;
	int width = i_ptr->cols;
	int height = i_ptr->rows;
	int channel = i_ptr->channels();

	cv::Mat* o_ptr = &output;
	if (i_ptr != o_ptr) {
		*o_ptr = cv::Mat(height, width, CV_MAKETYPE(CV_32F, channel));
	}

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float r = sqrt((float)(x * x + y * y));
			double coeff = (1.0 - 1.0 / (1.0 + exp(r - threshold))) * (upper - lower) + lower;
			for (int c = 0; c < channel; c++) {
				o_ptr->at<float>(y, x * channel + c) = coeff * i_ptr->at<float>(y, x * channel + c);
			}
		}
	}
}

//blake algorithm
/*
#include <iostream>
#include <string>

#include <opencv2/imgcodecs.hpp>

//#include "C:\Users\Thaddeus\Downloads\ColorConstancy-master\ColorConstancy-master\clcnst\clcnst.h"

float threshold;
string ifname, ofname;

int main(int argc, char** argv) {
	// Load input file
	cout << "[BlakeAlgorithm] input file name: ";
	cin >> ifname;
	//ifname = "C:\Users\Thaddeus\Desktop\Mask_RCNN-master\Mask_RCNN-master\samples\resistor\test1-mask0.png";
	//cv::Mat img = cv::imread(ifname, CV_LOAD_IMAGE_COLOR);
	cv::Mat img = cv::imread(ifname,cv::IMREAD_COLOR);
	if (img.empty()) {
		cout << "Failed to load file \"" << ifname << "\"." << endl;
		return -1;
	}
	int width = img.cols;
	int height = img.rows;
	int channel = img.channels();
	img.convertTo(img, CV_32FC3, 1.0 / 255.0);

	// Obtain threshold value by keyborad interaction
	cout << "[BlakeAlgorithm] input threshold value (default = 0.10): ";
	cin >> threshold;

	// Compute logarithm of input image
	cv::Mat out;
	clcnst::logarithm(img, out);

	// Laplacian filter divided by thresholding
	cv::Mat laplace = cv::Mat::zeros(height, width, CV_32FC3);
	for (int c = 0; c < channel; c++) {
		// Compute gradient and thresholding
		cv::Mat grad = cv::Mat::zeros(height, width, CV_32FC2);
		for (int y = 0; y < height - 1; y++) {
			for (int x = 0; x < width - 1; x++) {
				float dx = out.at<float>(y, (x + 1) * channel + c) - out.at<float>(y, x * channel + c);
				float dy = out.at<float>(y + 1, x * channel + c) - out.at<float>(y, x * channel + c);
				if (fabs(dx) > threshold) {
					grad.at<float>(y, x * 2 + 0) = dx;
				}

				if (fabs(dy) > threshold) {
					grad.at<float>(y, x * 2 + 1) = dy;
				}
			}
		}

		// Compute gradient again
		for (int y = 1; y < height; y++) {
			for (int x = 1; x < width; x++) {
				float ddx = grad.at<float>(y, x * 2 + 0) - grad.at<float>(y, (x - 1) * 2 + 0);
				float ddy = grad.at<float>(y, x * 2 + 1) - grad.at<float>(y - 1, x * 2 + 1);
				laplace.at<float>(y, x * channel + c) = ddx + ddy;
			}
		}
	}

	// Gauss Seidel method
	clcnst::gauss_seidel(out, laplace, 20);

	// Normalization
	clcnst::normalize(out, out);

	// Compute exponential
	clcnst::exponential(out, out);

	// Display result
	cv::namedWindow("Input");
	cv::namedWindow("Output");
	cv::imshow("Input", img);
	cv::imshow("Output", out);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// Save output
	cout << "[BlakeAlgorithm] save as: ";
	cin >> ofname;
	out.convertTo(out, CV_8UC3, 255.0);
	cv::imwrite(ofname, out);

}
*/

//Faugeras
/*
#include <iostream>
#include <string>
//using namespace std;

//#include <opencv2/opencv.hpp>

//#include "../clcnst/clcnst.h"

int ns;
float sigma, scale;
string ifname, ofname;

void hef_faugeras(cv::Mat& input, cv::Mat& output) {
	cv::Mat* i_ptr = &input;
	int width = i_ptr->cols;
	int height = i_ptr->rows;
	int channel = i_ptr->channels();

	cv::Mat* o_ptr = &output;
	if (i_ptr != o_ptr) {
		*o_ptr = cv::Mat(height, width, CV_MAKETYPE(CV_32F, channel));
	}

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float r = sqrt((float)(x * x + y * y));
			double coeff = 1.5f - exp(-r / 5.0f);
			for (int c = 0; c < channel; c++) {
				o_ptr->at<float>(y, x * channel + c) = coeff * i_ptr->at<float>(y, x * channel + c);
			}
		}
	}
}

int main(int argc, char** argv) {
	// Load input image
	cout << "[FaugerasAlgorithm] input file name to load: ";
	cin >> ifname;
	cv::Mat img = cv::imread(ifname, cv::IMREAD_COLOR);
	if (img.empty()) {
		cout << "Failed to load file \"" << ifname << "\"." << endl;
		return -1;
	}
	int width = img.cols;
	int height = img.rows;
	int channel = img.channels();
	img.convertTo(img, CV_32FC3, 1.0 / 255.0);

	// Convert color space 
	cv::Mat hvs;
	cv::cvtColor(img, hvs, cv::COLOR_BGR2Lab);

	// Homomophic filtering
	vector<cv::Mat> chs, spc(channel, cv::Mat(height, width, CV_32FC1));
	cv::split(hvs, chs);

	for (int c = 1; c < channel; c++) {
		cv::dct(chs[c], spc[c]);
		hef_faugeras(spc[c], spc[c]);
		cv::idct(spc[c], chs[c]);
	}
	cv::Mat out;
	cv::merge(chs, out);

	// Recover color space
	cv::cvtColor(out, out, cv::COLOR_Lab2BGR);

	// Display result
	cv::namedWindow("Input");
	cv::namedWindow("Output");
	cv::imshow("Input", img);
	cv::imshow("Output", out);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// Save result
	cout << "[FaugerasAlgorithm] save as: ";
	cin >> ofname;
	out.convertTo(out, CV_8UC3, 255.0);
	cv::imwrite(ofname, out);
}
*/

//Faugeras with homomorphic filter
/*
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
//using namespace std;

//#include <opencv2/opencv.hpp>

//#include "../clcnst/clcnst.h"

string ifname, ofname, isp;
float lower, upper, threshold;

int main(int argc, char** argv) {
	// Load input image
	cout << "[HomomorphicFilter] input file name to load: ";
	cin >> ifname;

	cv::Mat img = cv::imread(ifname, cv::IMREAD_COLOR);
	if (img.empty()) {
		cout << "Failed to load file \"" << ifname << "\"." << endl;
		return -1;
	}
	int width = img.cols;
	int height = img.rows;
	int channel = img.channels();
	img.convertTo(img, CV_32FC3, 1.0 / 255.0);

	// Obtain parameters from command line arguments
	cout << "[HomomorphicFilter] you want to specify parameters? (y/n): ";
	cin >> isp;
	if (isp == "y") {
		cout << "  scale for  low frequency (default = 0.5): ";
		cin >> lower;
		cout << "  scale for high frequency (default = 2.0): ";
		cin >> upper;
		cout << "  threshold value for frequency domain (default = 7.5):";
		cin >> threshold;
	}
	else {
		lower = 0.5f;
		upper = 2.0f;
		threshold = 7.5f;
	}

	// Perform DFT, high emphasis filter and IDFT
	vector<cv::Mat> chs, spc(channel, cv::Mat(height, width, CV_32FC1));
	cv::split(img, chs);

	for (int c = 0; c < channel; c++) {
		cv::dct(chs[c], spc[c]);
		clcnst::hef(spc[c], spc[c], lower, upper, threshold);
		cv::idct(spc[c], chs[c]);
	}
	cv::Mat out;
	cv::merge(chs, out);

	// Display result
	cv::namedWindow("Input");
	cv::namedWindow("Output");
	cv::imshow("Input", img);
	cv::imshow("Output", out);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// Save result
	cout << "[HomomorphicFilter] save as: ";
	cin >> ofname;
	out.convertTo(out, CV_8UC3, 255.0);
	cv::imwrite(ofname, out);
}
*/

//horn
/*
#include <iostream>
#include <string>
//using namespace std;

//#include <opencv2/opencv.hpp>

//#include "../clcnst/clcnst.h"

float threshold;
string ifname, ofname;

int main(int argc, char** argv) {
	if (argc < 1) {
		cout << "usage: HornAlgorithm.exe" << endl;
		return -1;
	}

	cout << "[HornAlgorithm] input file name: ";
	cin >> ifname;

	cv::Mat img = cv::imread(ifname, cv::IMREAD_COLOR);
	if (img.empty()) {
		cout << "Failed to load file \"" << ifname << "\"." << endl;
		return -1;
	}
	int width = img.cols;
	int height = img.rows;
	int channel = img.channels();
	img.convertTo(img, CV_32FC3, 1.0 / 255.0);

	cout << "[HornAlgorithm] input threshold value (default = 0.05): ";
	cin >> threshold;

	cv::Mat out, laplace;

	
	clcnst::logarithm(img, out);

	
	clcnst::laplacian(out, laplace);

	
	clcnst::threshold(laplace, laplace, threshold);

	
	clcnst::gauss_seidel(out, laplace, 20);


	clcnst::normalize(out, out);

	
	clcnst::exponential(out, out);

	cv::namedWindow("Input");
	cv::namedWindow("Output");
	cv::imshow("Input", img);
	cv::imshow("Output", out);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// Save output
	cout << "[HornAlgorithm] save as: ";
	cin >> ofname;
	out.convertTo(out, CV_8UC3, 255.0);
	cv::imwrite(ofname, out);
}
*/

//moore
/*
#include <iostream>
#include <string>
//using namespace std;

//#include <opencv2/opencv.hpp>

//#include "../clcnst/clcnst.h"

float sigma;
string ifname, ofname, isex;

int main(int argc, char** argv) {
	// Load input image
	cout << "[MooreAlgorithm] input file name to load: ";
	cin >> ifname;

	cv::Mat img = cv::imread(ifname, cv::IMREAD_COLOR);
	if (img.empty()) {
		cout << "Failed to load file \"" << ifname << "\"." << endl;
		return -1;
	}
	int width = img.cols;
	int height = img.rows;
	int channel = img.channels();
	img.convertTo(img, CV_32FC3, 1.0 / 255.0);

	// Apply Gaussian filter
	cout << "[MooreAlgorithm] input sigma value for Gaussian: ";
	cin >> sigma;
	sigma = sigma * max(width, height);

	cv::Mat gauss;
	cv::GaussianBlur(img, gauss, cv::Size(0, 0), sigma);

	// Additional process for extended Moore
	cout << "[MooreAlgorithm] use extended algorithm? (y/n): ";
	cin >> isex;
	if (isex == "y") {
		cv::Mat gray;
		cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

		cv::Mat edge = cv::Mat::zeros(height, width, CV_32FC1);
		for (int y = 1; y < height - 1; y++) {
			for (int x = 1; x < width - 1; x++) {
				float dx = (gray.at<float>(y, x + 1) - gray.at<float>(y, x - 1)) / 2.0f;
				float dy = (gray.at<float>(y + 1, x) - gray.at<float>(y - 1, x)) / 2.0f;
				edge.at<float>(y, x) = sqrt(dx * dx + dy * dy);
			}
		}

		cv::GaussianBlur(edge, edge, cv::Size(0, 0), sigma);
		cv::namedWindow("Edge");
		cv::imshow("Edge", edge);

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				for (int c = 0; c < channel; c++) {
					gauss.at<float>(y, x * channel + c) *= edge.at<float>(y, x);
				}
			}
		}
	}

	// Subtraction
	cv::Mat out;
	cv::subtract(img, gauss, out);

	// Offset reflectance
	out.convertTo(out, CV_32FC3, 1.0, -1.0);

	// Normalization
	clcnst::normalize(out, out, 0.0f, 1.0f);

	// Display result
	cv::namedWindow("Input");
	cv::namedWindow("Output");
	cv::imshow("Input", img);
	cv::imshow("Output", out);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// Save output image
	cout << "[MooreAlgorithm] save as: ";
	cin >> ofname;
	out.convertTo(out, CV_8UC3, 255.0);
	cv::imwrite(ofname, out);
}
*/

//rahman
/*
#include <iostream>
#include <string>
//using namespace std;

//#include <opencv2/opencv.hpp>

//#include "../clcnst/clcnst.h"

int ns;
float sigma, scale;
string ifname, ofname, isp;

int main(int argc, char** argv) {
	// Load input image
	cout << "[RahmanAlgorithm] input file name to load: ";
	cin >> ifname;
	cv::Mat img = cv::imread(ifname, cv::IMREAD_COLOR);
	if (img.empty()) {
		cout << "Failed to load file \"" << ifname << "\"." << endl;
		return -1;
	}
	int width = img.cols;
	int height = img.rows;
	int channel = img.channels();
	img.convertTo(img, CV_32FC3, 1.0 / 255.0);

	// Obtain parameters by keyboard interaction
	cout << "[RahmanAlgorithm] you want to specify parameters? (y/n): ";
	cin >> isp;
	if (isp == "y") {
		cout << "  sigma = ";
		cin >> sigma;
		cout << "  number of sigmas = ";
		cin >> ns;
		cout << "  scales for sigmas = ";
		cin >> scale;
	}
	else {
		sigma = 1.0f;
		ns = 3;
		scale = 0.16f;
	}

	vector<float> sigmas = vector<float>(ns);
	sigmas[0] = sigma * (float)max(height, width);
	for (int i = 1; i < ns; i++) sigmas[i] = sigmas[i - 1] * scale;

	// Accumulate multiscale results of Moore's algorithm
	cv::Mat out, tmp, gauss;
	double weight = 0.0;
	out = cv::Mat(height, width, CV_32FC3);
	for (int i = 0; i < ns; i++) {
		// Apply Gaussian filter
		cv::GaussianBlur(img, gauss, cv::Size(0, 0), sigmas[i]);

		// Subtraction
		cv::subtract(img, gauss, tmp);

		// Offset reflectance
		tmp.convertTo(tmp, CV_32FC3, 1.0, -1.0);

		// Normalization
		clcnst::normalize(tmp, tmp, 0.0f, 1.0f);

		// Accumulate
		cv::scaleAdd(tmp, 1.0 / (i + 1), out, out);
		weight += 1.0 / (i + 1);
	}
	out.convertTo(out, CV_32FC3, 1.0 / weight);

	// Display result
	cv::namedWindow("Input");
	cv::namedWindow("Output");
	cv::imshow("Input", img);
	cv::imshow("Output", out);
	cv::waitKey(0);
	cv::destroyAllWindows();

	// Save output image
	cout << "[RahmanAlgorithm] save as: ";
	cin >> ofname;
	out.convertTo(out, CV_8UC3, 255.0);
	cv::imwrite(ofname, out);
}
*/