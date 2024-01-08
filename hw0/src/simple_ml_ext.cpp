#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <utility>

namespace py = pybind11;

void matmul(const float *A, const float *B, float *r, size_t x, size_t y, size_t z)
{
    for (int k = 0; k < y; k++)
    {
        for (int i = 0; i < x; i++) 
        {
            for (int j = 0; j < z; j++)
            {
                r[i * z + j] += A[i * y + k] * B[k * z + j];
            }
        }
    }
}

void mat_transpose(const float *X, float *r, size_t m, size_t n)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m ; j++)
            r[i * m + j] = X[j * n + i];
}
void softmax_regression_epoch_cpp(const float *X, const unsigned char *y, float *theta, size_t m, size_t n, size_t k, float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for (int i = 0; i < m; i += batch)
    {
        int this_batch = std::min(m - i, batch);
        // printf("this batch:")
        auto this_X = X + i * n;
        auto this_y = y + i;
        float *Z = new float[this_batch * k]();
        matmul(this_X, theta, Z, this_batch, n, k); // batch * n, n * k
        // Z -> exp_Z
        for (int i = 0; i < this_batch * k; i++)
            Z[i] = std::exp(Z[i]);
        // exp_Z -> normalized_Z
        for (int i = 0; i < this_batch; i++) 
        {
            float sum = 0;
            for (int j = 0; j < k; j++)
                sum += Z[i * k + j];
            for (int j = 0; j < k; j++)
                Z[i * k + j] /= sum;
        }
        // normalized_Z = Z / np.sum(Z, axis=1, keepdims=True)
        
        for (int i = 0; i < this_batch; i++)
            Z[i * k + this_y[i]]--;
        // one_hot = np.eye(theta.shape[1])[this_y] # m * k
        float *this_X_trans = new float[this_batch * n]();
        mat_transpose(this_X, this_X_trans, this_batch, n);
        float *grad = new float[n * k]();
        matmul(this_X_trans, Z, grad, n, this_batch, k);
        for (int i = 0; i < n * k; i++)
        {
            theta[i] -= (grad[i] / this_batch) * lr;
        }
        delete [] Z;
        delete [] grad;
        delete [] this_X_trans;
        //  / this_batch  # n * m, m * k
        // theta -= lr * grad
    }
    
    /// END YOUR CODE
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m)
{
    m.def(
        "softmax_regression_epoch_cpp",
        [](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch)
        {
            softmax_regression_epoch_cpp(
                static_cast<const float *>(X.request().ptr),
                static_cast<const unsigned char *>(y.request().ptr),
                static_cast<float *>(theta.request().ptr),
                X.request().shape[0],
                X.request().shape[1],
                theta.request().shape[1],
                lr,
                batch);
        },
        py::arg("X"), py::arg("y"), py::arg("theta"),
        py::arg("lr"), py::arg("batch"));
}
