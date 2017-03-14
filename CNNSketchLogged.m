function err_rates = CNNSketchLogged(trials_sz, log_file)
% Path of toolbox directory
% trials_sz =  1e3:1e3:6e3;
toolbox_dir='./';
addpath(genpath(toolbox_dir));

% Load the digit data
load mnist_uint8;

% Load the training and testing features and labels
train_x = double(reshape(train_x',28,28,60000))/255;
test_x = double(reshape(test_x',28,28,10000))/255;
train_y = double(train_y');
test_y = double(test_y');
% rand('state',0)


% The CNN configuration
cnn.layers = {
    struct('type', 'i') % input layer
    struct('type', 'c', 'outputmaps', 3, 'kernelsize', 3) % convolution layer
    struct('type', 's', 'scale', 2) % sub-sampling layer
    };
cnn.output = 'softmax';	% Output neuron activations - one of 'softmax' or 'sigm'
% Stochastic gradient descent parameters
opts.alpha = 1;			% Learning rate
opts.batchsize = 50;	% Size of one batch
opts.numepochs = 20;	% Number of iterations
% Noise parameters
opts.noise_type = 'blind';	% One of 'blind' or 'nem'
opts.noise_var = 1e-1;	% Noise variance
opts.noise_pdf = 'uniform';	% One of 'gaussian' or 'uniform'
opts.anneal_fact = 2;	% Noise annealing factor (positive real number, preferably 1 or 2)
opts.maxMC = 1;
opts.logfile = sprintf(log_file);	% Logfile
%'log/logfile-nem-2.txt'

er=zeros(1, length(trials_sz));
%bad=zeros(1, length(trials_sz));
for j = 1:length(trials_sz)
    sketchsz = randsample(length(train_x), trials_sz(j))';
    trainX = train_x(:,:,sketchsz);
    trainY = train_y(:,sketchsz);
    % Setup the CNN
    cnn = cnnsetup(cnn, trainX, trainY);
    % Train the CNN
    cnn = cnntrain(cnn, trainX, trainY, opts);
    % Test the CNN
    er(j) = cnntest(cnn, test_x, test_y);
    fprintf('Test set classification error:  %f\n',...
        er(j)...
        );
end

%figure; plot(er);
err_rates = er;
return;
