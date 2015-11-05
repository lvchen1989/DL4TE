
% GaussianRBM

% Based on code provided by Geoff Hinton and Ruslan Salakhutdinov.
% http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html
% and code provided by Yichuan (Charlie) Tang 
% http://www.cs.toronto.edu/~tang

% PURPOSE: Training Gaussian RBM with CD, learning visible nodes' variances as well as sparsity penalty


%input of the function£º
% 1¡¢batchdata : data batches
% 2¡¢params : the parameters

%output of the function£ºparams1~4 are parameters to train the GRBM
% 1¡¢vhW : weights between the visible units and hidden units
% 2¡¢vb  : bias on visible units
% 3¡¢hb  : bias on hidden units
% 4¡¢fvar: variance, the square of fstd
% 5¡¢errs: the reconstruction error for each interation
%function [vhW, vb, hb, fvar, errs] = GRBM_Standard(batchdata, params)

% n£ºthe number of cases in each batch
% d£ºthe dimension of each case
%nBatches±íÊ¾£ºthe number of batches
[n d nBatches]=size(batchdata);%n = 100  d = 200  nBatches


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%parameters%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
params.v_var = 1;                                     %variance 
params.STD = 0;                                       %whether to learn variance, 1 is true
params.nHidNodes = numhid;                            %the number of hidden units
params.std_rate = 0.00001;                            %learning rate for STD
params.maxepoch = maxepoch;                           %the interation times for GRBM training
params.epislonw = 0.001;                              %learning rate for weights
params.PreWts.vhW = 0.1*randn(d, params.nHidNodes);   %initialization for weights
params.PreWts.hb = zeros(1,params.nHidNodes);         %initialization for hidden units
params.PreWts.vb = zeros(1,d);                        %initialization for visible units
params.nCD = 1;                                       %the interation times for CD
params.init_final_momen_iter = 5;                     %the start initialization for momentum update
params.final_momen = 0.9;                             %final value for momentum
params.init_momen = 0.5;                              %initialization for momentum
params.wtcost = 0.0002;%0.0002;                       %weight to prevent overfitting
params.SPARSE = 1;                                    %whether to learn sparse parameter, 1 is true
params.sparse_p = 0.05;                               %
params.sparse_lambda = 0.002;                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%parameters%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%parameters initialization%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
assert(params.v_var > 0);%

fstd = ones(1,d)*sqrt(params.v_var);
params.v_var=[];

r = params.epislonw; %learning rate for weights£¬you can learn more details in A Practical Guide to Training Restricted Boltzmann Machines

%learning rate for STD
if params.STD == 1
    std_rate = linspace(0, params.std_rate, params.maxepoch);
    std_rate(:) = params.std_rate;
    std_rate(1:min(10, params.maxepoch/2)) = 0; %learning schedule for variances
    invfstdInc = zeros(1,d);
end;

%assert the dimension of each parameter is correct
assert( all(size(params.PreWts.vhW) == [d params.nHidNodes]));
assert( all(size(params.PreWts.hb) == [1 params.nHidNodes]));
assert( all(size(params.PreWts.vb) == [1 d]));

vhW = params.PreWts.vhW;%weight
vb = params.PreWts.vb;%bias on visible units
hb = params.PreWts.hb;%bias on hidden units


grbm_batchposhidprobs=zeros(n,params.nHidNodes,nBatches);
 
%the update value for the weights
vhWInc  = zeros( d, params.nHidNodes);%weights
hbInc   = zeros( 1, params.nHidNodes);%bias
vbInc   = zeros( 1, d);%bias

Ones = ones(n,1);%
errs =  zeros(1, params.maxepoch);%the reconstruction error for each interation


if params.SPARSE == 1
    q=zeros(1, params.nHidNodes); %keep track of average activations
end

fprintf('\rTraining Learning v_var Gaussian-Binary RBM %d-%d   epochs:%d r:%f',...
    d, params.nHidNodes, params.maxepoch, r);

%%%%%%%%%%%%%%%%%%%%%%%%%%%end parameter initialization%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%start GRBM training%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%start interation
for epoch = 1:params.maxepoch
  
%     
%     if rem(epoch, int32(params.maxepoch/20)) == 0 || epoch < 30
%         fprintf('\repoch %d',epoch);
%     end
    
    fprintf(1,'epoch %d\r',epoch);
    
    THerrs =  zeros(n*nBatches, 2);%init the reconstruction error 0,two parts T and H.
    
    errsum=0;%reconstruction error 0
    ptot = 0;
    %update the parameters for each batch
    for batch = 1:nBatches
        if epoch ==1
        fprintf(1,'epoch %d batch %d\r',epoch,batch); 
        end
       
        Fstd = Ones*fstd;
        
        %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data = batchdata(:,:,batch); %nxd, load the batch   
        pos_hidprobs = 1./(1 + exp(-(data./Fstd)*vhW - Ones*hb)); %p(h_j =1|data) , (100 * nhid)
        grbm_batchposhidprobs(:,:,batch)=pos_hidprobs;
        
        %compute the binary states for hidden units activations
        pos_hidstates = pos_hidprobs > rand( size(pos_hidprobs) );  
                
        pos_prods    = (data./Fstd)'* pos_hidprobs; % <vj  x  hj>data
        pos_hid_act  = sum(pos_hidprobs); 
        pos_vis_act  = sum(data)./(fstd.^2); 
               
        %%%%%%%%% END OF POSITIVE PHASE %%%%%%%%%
        
        %the times for CD
        for iterCD = 1:params.nCD
            
            %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            negdataprobs = pos_hidstates*vhW'.*Fstd+Ones*vb; 
            negdata = negdataprobs + randn(n, d).*Fstd; 
            neg_hidprobs = 1./(1 + exp(-(negdata./Fstd)*vhW - Ones*hb ));     %updating hidden nodes again
            pos_hidstates = neg_hidprobs > rand( size(neg_hidprobs) );       
            
        end %end CD iterations
       
        neg_prods  = (negdata./Fstd)'*neg_hidprobs;  % <vj  x  hj>model
        neg_hid_act = sum(neg_hidprobs);
        neg_vis_act = sum(negdata)./(fstd.^2);
        
        %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        Tdata = data(:,1:d/2);
        Hdata = data(:,d/2+1:d);
        Tnegdata = negdata(:,1:d/2);
        Hnegdata = negdata(:,d/2+1:d);
        
%         sqrt(sum((Tdata-Tnegdata).^2,2));
        
%         sqrt(sum((Hdata-Hnegdata).^2,2));
        
        THerrs((1+(batch-1)*n:batch*n),1)= sqrt(sum((Tdata-Tnegdata).^2,2));
        THerrs((1+(batch-1)*n:batch*n),2)= sqrt(sum((Hdata-Hnegdata).^2,2));
        
        
        errsum = errsum + sum(sum( (data-negdata).^2 ));%sum the reconstruction error
        
        
        if epoch > params.init_final_momen_iter,
            momentum=params.final_momen;
        else
            momentum=params.init_momen;
        end
        
        %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
       
        vhWInc = momentum*vhWInc + r/n*(pos_prods-neg_prods) - r*params.wtcost*vhW;
        vbInc = momentum*vbInc + (r/n)*(pos_vis_act-neg_vis_act);
        hbInc = momentum*hbInc + (r/n)*(pos_hid_act-neg_hid_act);
        
        
        if params.STD == 1
            invfstd_grad = sum(2*data.*(Ones*vb-data/2)./Fstd,1) + sum(data' .* (vhW*pos_hidprobs') ,2)';
            invfstd_grad = invfstd_grad - ( sum(2*negdata.*(Ones*vb-negdata/2)./Fstd,1) + ...
                                    sum( negdata'.*(vhW*neg_hidprobs') ,2 )' );
            %compute gradient                   
            invfstdInc = momentum*invfstdInc + std_rate(epoch)/n*invfstd_grad;
        end
        
       
        if params.SPARSE == 1 %nair's paper on 3D object recognition            
            %update q
            if batch==1 && epoch == 1
                q = mean(pos_hidprobs);
            else
                q_prev = q;
                q = 0.9*q_prev+0.1*mean(pos_hidprobs);
            end           
           
            p = params.sparse_p;
            grad = 0.1*params.sparse_lambda/n*sum(pos_hidprobs.*(1-pos_hidprobs)).*(p-q)./(q.*(1-q));
            gradW =0.1*params.sparse_lambda/n*(data'./Fstd'*(pos_hidprobs.*(1-pos_hidprobs))).*repmat((p-q)./(q.*(1-q)), d,1);
            
            
            hbInc = hbInc + r*grad;
            vhWInc = vhWInc + r*gradW;
        end
        
        ptot = ptot+mean(pos_hidprobs(:));
        
        %update weights
        vhW = vhW + vhWInc;
        vb = vb + vbInc;
        hb = hb + hbInc;    
        
        %
        if params.STD == 1
            invfstd = 1./fstd;
            invfstd =  invfstd + invfstdInc;
            fstd = 1./invfstd;
            fstd = max(fstd, 0.005); %have a lower bound!        
        end
        %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
    end

%     if rem(epoch, int32(params.maxepoch/20)) == 0 || epoch < 30
%         fprintf(1, ' p%1.2f  ',  ptot/nBatches );
%         %fprintf(1, ' error %6.2f  stdr:%.5f fstd(x,y): [%2.3f %2.3f] mm:%.2f ', errsum, std_rate(epoch), fstd(1), fstd(2), momentum);
%         fprintf(1, ' error %6.2f ', errsum);
%         fprintf(1, 'vh_W min %2.4f   max %2.4f ', min(min(vhW)), max(max(vhW)));
%     end
    if rem(epoch,10)==0
        hidrecbiases=hb;%bias on hidden units
        vishid = vhW;%
        name=strcat('jointRBM_mid',num2str(epoch));
        save name_60 vishid hidrecbiases vb;
        save THerrorsum_60 THerrs;
    end
    
    fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum); %the reconstruction error of all batches
    
    %record the error
    errs(epoch) = errsum;   
    
    save errorsum_60 errs;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%end GRBM training%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%clear the variables
clear params;
clear vhWInc;
clear vbInc;
clear hbInc;
clear Ones;
clear errs;
clear Fstd;
clear data;
clear pos_hidprobs;
clear pos_hidstates;
clear pos_prods;
clear pos_hid_act;
clear pos_vis_act;
clear negdataprobs;
clear negdata;
clear neg_hidprobs;
clear neg_hidstates;
clear neg_prods;
clear neg_hid_act;
clear neg_vis_act;

fvar = fstd.^2;

