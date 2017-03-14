Here is a one-line description of each directory:

CNN/ - Contains functions for CNN training.
data/ - Contains the MNIST digit recognition data set.
log/ - Output log files go here.
tests/ - Contains sample code for training a CNN on the MNIST data set.
utils/ - Contains many utility functions.

Here are some key points:

(1) Check tests/test_example_CNN.m for a sample code for training a CNN.
You should only modify this file under normal circumstances.

(2) The training input data variable should be a N x M x T array for T images
of size N x M each. The training target data variable should be a C x T matrix
for a C-class classification problem. Each row of the target data matrix
should contain a 1-in-C bit vector of the true class label. For example,
if C = 5 and the true class label is 2, then the corresponding 1-in-C
bit vector is [0 1 0 0 0].

(3) Normalize the image pixels to [0,1]. The code tests/test_example_CNN.m
assumes that the un-normalized pixels are in [0,255].

(4) Start by tuning the noise variance on a logarithmic scale, e.g.
{1e-4, 1e-3, 1e-2, 1e-1}. Then tune the noise using a linear scale around
the best point.

(5) Fix the output neuron activations to 'softmax' for classification.
