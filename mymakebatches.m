
%%%%%%%%%%%%%%%%%%%%%%load training data%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
digitdata=[]; %save the data£¬will be cleared from the memory

%load the data

load mydata_batch; 

digitdata = [digitdata; mydata_batch];


%scale the data.
digitdata = single(digitdata);%235207x200, data contains 235207 cases, the visible units of the each case are 200
parfor j=1:size(digitdata,2)% 200
    digitdata(:,j) = (digitdata(:,j) - mean(digitdata(:,j)))/std(digitdata(:,j));
end

save scale_mybatch digitdata;


%set parameters for the batch£¬including the number of cases in each batch¡¢the number of batches etc.
totnum=size(digitdata,1);%the number of the dataset, 235207
fprintf(1, 'Size of the training dataset= %5d \n', totnum);

rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);%

numbatches=totnum/100;%the number of batches

numbatches = fix(numbatches);

fprintf(1, 'Size of the numbatches dataset= %5d \n', numbatches);

numdims  =  size(digitdata,2);%the dimension of each case.
batchsize = 100;%the number of cases in each batch
batchdata = zeros(batchsize, numdims, numbatches);

%construct data in batches
for b=1:numbatches
  batchdata(:,:,b) = digitdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);%
end;
clear digitdata;%clear the temporary variable


%clear the unused data
clear data;
clear randomorder;

%%% Reset random seeds 
rand('state',sum(100*clock)); 
randn('state',sum(100*clock)); 



