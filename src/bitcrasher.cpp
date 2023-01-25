//g++ -o a main.cpp `pkg-config --cflags --libs opencv`

#include <cmath>
#include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "libbitcrasher.hpp"

using namespace cv;

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " in.pgm out.pbm" << std::endl;
    }
    else
    {
        Mat image = imread(argv[1], IMREAD_GRAYSCALE);
        int block_size = 64;

        /*resize to divide into 64x64 blocks*/
        int origheight = image.rows, origwidth = image.cols;
        int dwidth = (64 - (origwidth % 64)) % 64;
        int dheight = (64 - (origheight % 64)) % 64;
        copyMakeBorder(image, image, 0, dheight, 0, dwidth, BORDER_REPLICATE, Scalar::all(0));
        int height = image.rows, width = image.cols;

        /*create matrix path walk*/
        Mat order = zigzag(block_size);

        Mat DCT_1d = Mat::zeros(block_size * block_size, block_size * block_size, CV_64F);
        Mat temp, reshaped;
        int index;

        for (int q = 0; q < block_size; q++)
        {
            for (int p = 0; p < block_size; p++)
            {
                temp = DCT_function(p, q, block_size);
                temp = temp.t();
                reshaped = temp.reshape(0, block_size * block_size);
                index = order.at<int>(p, q);
                reshaped.col(0).copyTo(DCT_1d.col(index)); // first row necessary?
            }
        }

        int M = 10;
        Mat A = DCT_1d(Range::all(), Range(0, 10));// problem!!!!!

        Mat oneD_version = im2col(image, block_size);

        int total_blocks = image.rows * image.cols / (std::pow(block_size, 2));

        Mat Rec_image = Mat::zeros(image.size(), CV_64F);
        Mat err_image = Mat::zeros(image.size(), CV_64F);

        Mat one_D_rec = Mat::zeros(oneD_version.size(), CV_64F);
        Mat one_D_error = Mat::zeros(oneD_version.size(), CV_64F);

        Mat b = Mat::zeros((block_size * block_size), 1, CV_64F);

        Mat x = Mat::zeros(A.cols, 1, CV_64F);

        double lagrangian_param = 1, relaxation_param = 1;
        Block_data data;
        Mat R;
        {
            Mat Atr = Mat::zeros(A.cols, A.rows, CV_64F);
            transpose(A, Atr);
            Mat NA = Atr * A;
            Cholesky(NA, R);
        }
        for (int i = 0; i != total_blocks; ++i)
        {
            b = oneD_version.col(i);
            ADMM(A, x, b, R, lagrangian_param, relaxation_param, data, i, total_blocks);
            one_D_rec.col(i) = A * x;
            one_D_error.col(i) = b - A * x;
        }

        Rec_image = col2im(one_D_rec, block_size, height, width);
        err_image = col2im(one_D_error, block_size, height, width);

        /*threshold result and save*/
        Mat map = abs(err_image), result;
        map.convertTo(map, CV_16UC1);
        //threshold(map, result, thresh, maxval, THRESH_BINARY_INV);
        threshold(map, result, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);

        Mat ROI(result, Rect(0, 0, origwidth, origheight));
        Mat croppedImage;
        // Copy the data into new matrix
        ROI.copyTo(croppedImage);

        std::cout << "map size: " << croppedImage.rows << " " << croppedImage.cols << std::endl;

        std::vector<int> compression_params;
        compression_params.push_back(IMWRITE_PXM_BINARY);
        imwrite(argv[2], croppedImage, compression_params);
    }
}
