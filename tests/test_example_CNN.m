% Path of toolbox directory
toolbox_dir='/home/kartik/research/dnn/software/nnsoft_DrTaber/';
addpath(genpath(toolbox_dir));

% Load the digit data
load mnist_uint8;

% Load the training and testing features and labels
train_x = double(reshape(train_x',28,28,60000))/255;
test_x = double(reshape(test_x',28,28,10000))/255;
train_y = double(train_y');
test_y = double(test_y');

% Pick the first 1000 training samples for example
trainsize = 1000;
train_x = train_x(:,:,1:trainsize);
train_y = train_y(:,1:trainsize);

rand('state',0)

% The CNN configuration
cnn.layers = {
    struct('type', 'i') % input layer
    struct('type', 'c', 'outputmaps', 3, 'kernelsize', 3) % convolution layer
    struct('type', 's', 'scale', 2) % sub-sampling layer
};
cnn.output = 'softmax';	% Output neuron activations - one of 'softmax' or 'sigm'

% Stochastic gradient descent parameters
opts.alpha = 1;	% Learning rate
opts.batchsize = 50;	% Size of one batch
opts.numepochs = 20;	% Number of iterations

% Noise parameters
opts.noise_type = 'blind';	% One of 'blind' or 'nem'
opts.noise_var = 1e-1;	% Noise variance
opts.noise_pdf = 'uniform';	% One of 'gaussian' or 'uniform'
opts.anneal_fact = 2;	% Noise annealing factor (positive real number, preferably 1 or 2)
opts.logfile = sprintf('../log/logfile.txt');	% Logfile

% Setup the CNN
cnn = cnnsetup(cnn, train_x, train_y);

% Train the CNN
cnn = cnntrain_nem(cnn, train_x, train_y, opts);

% Test the CNN
[er, bad] = cnntest(cnn, test_x, test_y);
fprintf('Test set classification error: %f\n',er);
