"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    import gzip
    import struct
    images = []
    def read_header(f):
        int_bytes = 4
        # read image
        magic_num_buf = f.read(int_bytes)
        magic_num = struct.unpack('>i', magic_num_buf)[0]
        dim = magic_num & 0x000000ff
        int_buf = f.read(int_bytes * dim)
        pack_str = '>' + 'i' * dim
        dims = struct.unpack(pack_str, int_buf)
        return dims

    with gzip.open(image_filename, 'rb') as f:
        dims = read_header(f)
        for i in range(dims[0]):
            length = dims[1] * dims[2]
            buf = f.read(length)
            pack_str = 'B' * length
            r = list(struct.unpack(pack_str, buf))
            images.append(r)

    with gzip.open(label_filename, 'rb') as f:
        dims = read_header(f)
        length = dims[0]
        buf = f.read(length)
        pack_str = 'B' * length
        label = list(struct.unpack(pack_str, buf))
    import numpy as np 
    return np.array(images, dtype=np.float32) / 255, np.array(label, dtype=np.uint8)
    ### END YOUR CODE


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    exp_Z = ndl.exp(Z)
    sum_exp_Z = ndl.summation(exp_Z, axes=(1,))
    Z_y = Z * y_one_hot
    Z_y = ndl.summation(Z_y, axes=(1,))
    losses = ndl.log(sum_exp_Z) - Z_y
    return ndl.summation(losses) / losses.shape[0]
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    m = X.shape[0]
    for i in range(0, m, batch):
        this_batch = batch
        if m - i < batch:
            this_batch = m - i 
        this_X = X[i:(i + this_batch), :]
        this_y = y[i:(i + this_batch)]
        one_hot = np.eye(W2.shape[1])[this_y]
        X_batch = ndl.Tensor(this_X)
        y_batch = ndl.Tensor(one_hot) # one_hot y   
        Z1 = ndl.matmul(X_batch, W1) # m * d, 
        layer_1_output = ndl.relu(Z1)
        layer_2_output = ndl.matmul(layer_1_output, W2)
        loss = softmax_loss(layer_2_output, y_batch)
        loss.backward()
         # m * k
        
        
        W1_updated = W1.numpy()- lr * W1.grad.numpy() 
        W2_updated = W2.numpy()- lr * W2.grad.numpy()
        W1 = ndl.Tensor(W1_updated)
        W2 = ndl.Tensor(W2_updated)
    return W1, W2 
    # raise NotImplementedError()
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
