#include <cmath>
#include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "libbitcrasher.hpp"

using namespace cv;

Mat zigzag(int size)
{
    int num_dia, number, k;
    //double number_previous_elements, number;
    int x, y;
    Mat result = Mat::zeros(size, size, CV_32S);

    for (k = 0; k < size; k++)
    {
        number = k * (k + 1) / 2;
        if (k % 2 == 0)
        {
            for (y = 0; y < k + 1; y++)
            {
                x = k - y;
                result.at<int>(x, y) = number;
                number++;
            }
        }
        else
        {
            for (x = 0; x < k + 1; x++)
            {
                y = k - x;
                result.at<int>(x, y) = number;
                number++;
            }
        }
    }

    for (k = size; k < 2 * size - 1; k++)
    {
        num_dia = 2 * size - k - 1;
        if (k % 2 == 1)
        {
            for (y = size - 1; y > size - num_dia - 1; y--)
            {
                x = k - y;
                result.at<int>(x, y) = number;
                number++;
            }
        }
        else
        {
            for (x = size - 1; x > size - num_dia - 1; x--)
            {
                y = k - x;
                result.at<int>(x, y) = number;
                number++;
            }
        }
    }

    return result;
}

Mat DCT_function(int p, int q, int size)
{
    const double pi = std::acos(-1);
    double alpha_p, alpha_q, fs;
    int m, n;
    Mat result = Mat::zeros(size, size, CV_64F);

    fs = 0.5 /(double)size;
    for (m = 0; m < size; m++)
    {
        for (n = 0; n < size; n++)
        {
            result.at<double>(m, n) =
                std::cos(pi * (2 * m + 1) * p * fs) *
                std::cos(pi * (2 * n + 1) * q * fs);
        }
    }

    alpha_p = (p == 0) ? std::sqrt(2.0 * fs) : std::sqrt(4.0 * fs);
    alpha_q = (q == 0) ? std::sqrt(2.0 * fs) : std::sqrt(4.0 * fs);

    result *= (alpha_q * alpha_p);
    return result;
}

/*converts each block into column*/
Mat im2col(Mat &input, int block_size)
{
    int size = input.rows * input.cols, column = 0;
    Mat result = Mat::zeros(std::pow(block_size, 2), size / std::pow(block_size, 2),CV_64F);

    for (int x = 0; x != input.cols; x += block_size)
    {
        for (int y = 0; y != input.rows; y += block_size)
        {
            for (int i = 0; i != block_size; ++i)
            {
                input(Range(y, y + block_size),
                      Range(x + i, x + i + 1)).copyTo(result(Range(i * 64, i * 64 + block_size),
                                                      Range(column, column + 1)));
            }
            ++column;
        }
    }
    return result;
}

/*vice versa to im2col*/
Mat col2im(Mat &input, int block_size, int height, int width)
{
    int column = 0;
    Mat result = Mat::zeros(height, width, CV_64F);

    for (int x = 0; x != width; x += block_size)
    {
        for (int y = 0; y != height; y += block_size)
        {
            for (int i = 0; i != block_size; ++i)
            {
                input(Range(i * 64, i * 64 + block_size),
                      Range(column, column + 1)).copyTo(result(Range(y, y + block_size),
                                                        Range(x + i, x + i + 1)));
            }
            ++column;
        }
    }
    return result;
}

/* Calculates upper triangular matrix S, where A is a symmetrical matrix A=S'*S */
void Cholesky(Mat& A, Mat& S)   // change to return Mat!
{
    int dim = A.rows;
    S.create(dim, dim, CV_64F);

    int i, j, k;

    for( i = 0; i < dim; i++ )
    {
        for( j = 0; j < i; j++ )
            S.at<double>(i,j) = 0.;

        double sum = 0.;
        for( k = 0; k < i; k++ )
        {
            double val = S.at<double>(k,i);
            sum += val*val;
        }

        S.at<double>(i,i) = std::sqrt(std::max(A.at<double>(i,i) - sum, 0.));
        double ival = 1./S.at<double>(i, i);

        for( j = i + 1; j < dim; j++ )
        {
            sum = 0;
            for( k = 0; k < i; k++ )
                sum += S.at<double>(k, i) * S.at<double>(k, j);

            S.at<double>(i, j) = (A.at<double>(i, j) - sum)*ival;
        }
    }
}

Mat take_max(Mat a, double b)
{
    return max(0, a - b) - max(0, -a - b);
}

void ADMM(Mat A, Mat &x, Mat b, Mat R, double lagrangian_param, double relaxation_param, Block_data &data, int i, int total_blocks)
{
    Mat D, w, Rtr, AX;
    Mat Atr = Mat::zeros(A.cols, A.rows, CV_64F);
    double ABSTOL, RELTOL;
    int m, n;

    int iters = 200;
    ABSTOL = 1e-4;
    //ABSTOL = std::pow(10, -4);
    RELTOL = 1e-2;//change
    //RELTOL = std::pow(10, -2);
    m = A.rows;//check docs!
    n = A.cols;

    data.value.reserve(iters);
    data.primal_rest_norm.reserve(iters);
    data.dual_rest_norm.reserve(iters);
    data.tolerance_primal.reserve(iters);
    data.tolerance_dual.reserve(iters);

    Mat x1 = Mat::zeros(n, 1, CV_64F);
    Mat z = Mat::zeros(m, 1, CV_64F); //64??? check shape on reflect!!
    Mat u = Mat::zeros(m, 1, CV_64F);

    transpose(A, Atr);
    transpose(R, Rtr);
    for (int k = 0; k < iters; k++)
    {

        solve(Rtr, Atr * (b + z - u), x1);
        solve(R, x1, x);

        w = z;
        AX = A * x;
        D = relaxation_param * AX + (1. - relaxation_param) * (w + b);
        z = take_max(D - b + u, 1. / lagrangian_param);
        u = u + (D - z - b);

        data.value[k] = norm(z, NORM_L1, noArray());
        data.primal_rest_norm[k] = norm(AX - z - b, NORM_L2, noArray());

        data.dual_rest_norm[k] = norm(-lagrangian_param * Atr * (z - w), NORM_L2, noArray());
        data.tolerance_primal[k] = std::sqrt(m) * ABSTOL + RELTOL * std::max(norm(AX, NORM_L2, noArray()),
                                   std::max(norm(-z, NORM_L2, noArray()),
                                            norm(b, NORM_L2, noArray())));
        data.tolerance_dual[k] = std::sqrt(n) * ABSTOL + RELTOL * norm(lagrangian_param * Atr * u, NORM_L2, noArray());
        if (((k + 1) % 100) == 0)
            std::cout << "block " << i << "/" << total_blocks << "\t"
                      //<< "iter " << k << "/" << iters << "\t"
                      << data.primal_rest_norm[k] << "\t"
                      << data.tolerance_primal[k] << "\t"
                      << data.dual_rest_norm[k] << "\t"
                      << data.tolerance_dual[k] << "\t"
                      << data.value[k] << std::endl;

        if (data.primal_rest_norm[k] < data.tolerance_primal[k] &&
                data.dual_rest_norm[k] < data.tolerance_dual[k])
        {
            break;
        }
    }
}
