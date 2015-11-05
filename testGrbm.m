% function output = testGrbm( input,vishid,vb,hidrecbiases)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
load testdata;% load test dataset
load jointRBM;% load jointRBM model

digitdata=testdata;
rand('state',sum(100*clock));
%scale the test data
for j=1:size(digitdata,2)% 200, each T-H case is 200 dimension, the dimension of both T and H is 100. 
    digitdata(:,j) = (digitdata(:,j) - mean(digitdata(:,j)))/std(digitdata(:,j));
end
%

n=1;
d=200;
Fstd=1;
Ones=ones(1,1);
vhW=vishid;
hb=hidrecbiases;

fid = fopen('test_err','w')
    
for j=1:size(digitdata,1)
    input = digitdata(j,:);
    
    j
    %get hidden units, positive phrase
    pos_hidprobs = 1./(1 + exp(-(input./Fstd)*vhW - Ones*hb));
    pos_hidstates = pos_hidprobs > rand( size(pos_hidprobs) );
    
    %get reconstructed visible units, negative phrase
    negdataprobs = pos_hidstates*vhW'.*Fstd+Ones*vb;
    negdata = negdataprobs + randn(n, d).*Fstd;
    
    
    ori1 = input(1:100);% original data
    ori2 = input(101:200);
    d1 = negdata(1:100);% reconstructed data
    d2 = negdata(101:200);
    
    s1 = normest(d1-ori1);
    s2 = normest(d2-ori2);
    
    fprintf(fid,'%d\t%f\t%f\n',j,s1,s2);
    
end
fclose(fid)

