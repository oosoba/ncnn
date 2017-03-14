function [loss] = cnneval(net, x, y)

net = cnnff(net, x);
net = cnnbp(net, y);
loss = net.L;
