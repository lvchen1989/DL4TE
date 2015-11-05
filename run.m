
% joint GaussianRBM training
%
% Based on code provided by Geoff Hinton and Ruslan Salakhutdinov.
% http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html
% and code provided by Yichuan (Charlie) Tang 
% http://www.cs.toronto.edu/~tang
% This program trains the joint RBM 
% You can set the maximum number of epochs for training
% and you can set the architecture of the multilayer net.

clear all
close all


maxepoch=2000; %max iteration times for RBM
numhid=100; %the number of hidden units

fprintf(1,'Pretraining a deep autoencoder. \n');
fprintf(1,'The Science paper used 50 epochs. This uses %3i \n', maxepoch);

mymakebatches;%construct batches
[numcases numdims numbatches]=size(batchdata);%100x200xn, each batch contains 100 cases.

%unsupervised learning for RBM, save the model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%GRBM training%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%this layer is GRBM, the visible units are Gaussian distribution, the hidden units are stochastic binary
fprintf(1,'Pretraining Layer 1 with GRBM: %d-%d \n',numdims,numhid);

GRBM_General;%train the Gaussian RBM

hidrecbiases=hb;%bias for hidden units 
vishid = vhW;% the weight of the layer
%save the trained model
save jointRBM vishid hidrecbiases vb;
%%%%%%%%%%%%%%%%%%%%%%%%%%GRBM training%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



clear all;


