#include <iostream>
#include <stdio.h>
#include <opencv\cv.h>
#include <opencv\cxcore.h>
#include <opencv\highgui.h>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;
using namespace cv;

int main(int argc, char*argv[]) {
	Mat img = imread("34.jpg");
	if (!img.data) {
		cout << "Could not load image" << endl;
		waitKey(0);
		return -1;
	}
	//Gray image and initialization
	Mat gr;
	int his[256];
	int his2[256];

	for (int i = 0; i < 256; i++) {
		his[i] = 0;
		his2[i] = 0;
	}
	//Gray conversion
	cvtColor(img, gr, CV_BGR2GRAY);

	int h = gr.rows;
	int w = gr.cols;
	int val = 0;
	int max = 0;

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			val = gr.at<uchar>(i, j);
			his[val]++;
		}
	}
	for (int i = 0; i < 256; i++) {
		if (max < his[i]) {
			max = his[i];
		}
	}

	//Histogram for Original Image
	Mat him(301, 260, CV_8UC1, Scalar(255));
	int hist[256];
	double maxd = max;
	for (int i = 0; i < 256; i++) {
		hist[i] = cvRound(double(his[i] / maxd) * 300);
		Point pt1 = Point(i, 300 - hist[i]);
		Point pt2 = Point(i, 300);
		line(him, pt1, pt2, Scalar(0), 1, 8, 0);
	}
	//Laplacian Operator
	Mat Lap = gr.clone();
	int kernel = 3;
	int lapMask[3][3] = {
		{-1 , -1 , -1 },
		{-1 , 8 , -1 },
		{ -1 , -1 , -1 }
	};

	for (int i = kernel / 2; i < h - kernel / 2; i++) {
		for (int j = kernel / 2; j < w - kernel / 2; j++) {
			int sum = 0;
			for (int k = -kernel / 2; k <= kernel / 2; k++) {
				for (int l = -kernel / 2; l <= kernel / 2; l++) {
					int val = gr.at<uchar>(i + k, j + l);
					sum += cvRound(lapMask[k + (kernel / 2)][l + (kernel / 2)] * val);
				}
			}
			if (sum < 0) {
				sum = 0;
			}
			else if (sum > 255) {
				sum = 255;
			}

			Lap.at<uchar>(i, j) = sum;
		}
	}

	//Pixel Calculation for Smoothed image
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			val = Lap.at<uchar>(i, j);
			his2[val] = his2[val] + 1;
		}
	}
	int max2 = 0;
	for (int i = 0; i < 256; i++) {
		//cout << "Gray Level  " << i << " : " << his2[i] << endl;
		if (max2 < his2[i]) {
			max2 = his2[i];
		}
	}
	//Histogram of Smoothed Image
	Mat him2(301, 260, CV_8UC1, Scalar(255));
	int hist2[256];
	double maxd2 = max2;
	for (int i = 0; i <= 255; i++) {
		hist2[i] = cvRound(double(his2[i] / maxd2) * 300);
		Point pt1 = Point(i, 300 - hist2[i]);
		Point pt2 = Point(i, 300);
		line(him2, pt1, pt2, Scalar(0), 1, 8, 0);
	}

	imshow("Image:", gr);
	imshow("Histograam:", him);
	imshow("Laplasian Image", Lap);
	imshow("New Histogram:", him2);

	cvWaitKey(0);


	return 0;

}