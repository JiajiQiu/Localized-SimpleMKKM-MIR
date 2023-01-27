path = '';
addpath(genpath(path));


dataName = 'proteinFold';
load([path,'datasets\',dataName,'_Kmatrix'],'KH','Y');
numclass = length(unique(Y));
numker = size(KH,3);
num = size(KH,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
KH = kcenter(KH);
KH = knorm(KH);


options.seuildiffsigma=1e-5;        % stopping criterion for weight variation
%------------------------------------------------------
% Setting some numerical parameters
%------------------------------------------------------
options.goldensearch_deltmax=1e-1; % initial precision of golden section search
options.numericalprecision=1e-16;   % numerical precision weights below this value
% are set to zero
%------------------------------------------------------
% some algorithms paramaters
%------------------------------------------------------
options.firstbasevariable='first'; % tie breaking method for choosing the base
% variable in the reduced gradient method
options.nbitermax=500;             % maximal number of iteration
options.seuil=0;                   % forcing to zero weights lower than this
options.seuilitermax=10;           % value, for iterations lower than this one
options.miniter=0;                 % minimal number of iterations
options.threshold = 1e-4;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 MM=zeros(numker);
        for i=1:numker
            for j=1:numker
%                 MM(i,j)=trace(KHH(:,:,i)*KHH(:,:,j));
                 MM(i,j)=trace(KH(:,:,i)*KH(:,:,j));
                 MM(j,i)=MM(i,j);
            end
        end
lambda = 2.^[-15:1:10];

tic;
Sigma = ones(numker,1)/numker;
avgKer  = mycombFun(KH,Sigma);
tauset = [0.05:0.05:0.95];
res_mean = zeros(4,length(tauset));
res_std = zeros(4,length(tauset));
Sigma_ = zeros(numker,length(tauset));
for la=1:length(lambda)
    for it =1:length(tauset)
        numSel = round(tauset(it)*num);
        NS = genarateNeighborhood(avgKer,numSel);
        %%--Calculate Neighborhood--%%%%%%
        A = zeros(num);
        for i =1:num
            A(NS(:,i),NS(:,i)) = A(NS(:,i),NS(:,i))+1;
        end       
        [H_normalized,Sigma(:,it),obj] = localizedSimpleMKKM(KH,numclass,A,options,MM,lambda(la),Y);
        fprintf("===========\n")
        [res_mean((la-1)*5+1:(la-1)*5+4,it),res_std((la-1)*5+1:(la-1)*5+4,it,it)] = myNMIACCV2(H_normalized,Y,numclass);
        fprintf("%g,%g\n",la,it);
    end
end
timecost = toc;

save('answer.mat','res_mean','res_std','Sigma','timecost'); 

