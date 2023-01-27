function [Hstar,Sigma,obj] = localizedSimpleMKKM(KH,numclass,NN,option,MM,lambda,Y)

numker = size(KH,3);
Sigma = ones(numker,1)/numker;

% KHP = zeros(num,num,numker);
% for p = 1:numker
%     KHP(:,:,p) = myLocalKernel(KH,tau,p);
% end
% KH = KHP;
% clear KHP
%--------------------------------------------------------------------------------
% Options used in subroutines
%--------------------------------------------------------------------------------
if ~isfield(option,'goldensearch_deltmax')
    option.goldensearch_deltmax=5e-2;
end
if ~isfield(option,'goldensearchmax')
    optiongoldensearchmax=1e-8;
end
if ~isfield(option,'firstbasevariable')
    option.firstbasevariable='first';
end

nloop = 1;
loop = 1;
goldensearch_deltmaxinit = option.goldensearch_deltmax;
%%---
% MaxIter = 30;
% res_mean = zeros(4,MaxIter);
% res_std = zeros(4,MaxIter);
%-----------------------------------------
% Initializing Kernel K-means
%------------------------------------------
Kmatrix = sumKbeta(KH,Sigma.^2);
%% H = mylocalkernelkmeans(KC,A0,cluster_count);
Sigmaold  = Sigma;
[Hstar,obj1]= mylocalkernelkmeans(Kmatrix,NN,numclass,MM,Sigma,lambda);
obj(nloop) = obj1;
disp(obj1);
[a,b,c,d,e,f]=myNMIACCV2(Hstar,Y,numclass);
fprintf("%g,,%g,,%g,,%g,,%g,,%g\n",a(1),a(2),a(3),a(4),nloop,obj(nloop));
% [[grad] = localSimpleMKKMGrad(KH,NN,Hstar,Sigma)
[grad] = localSimpleMKKMGrad(KH,NN,Hstar,Sigma,MM,lambda);


%------------------------------------------------------------------------------%
% Update Main loop
%------------------------------------------------------------------------------%

while loop
    nloop = nloop+1;
    %-----------------------------------------
    % Update weigths Sigma
    %-----------------------------------------
    [Sigma,Hstar,obj(nloop)] = localSimpleMKKMupdate(KH,Sigmaold,grad,NN,obj(nloop-1),numclass,option,MM,lambda);
    % [res_mean(:,nloop),res_std(:,nloop)] = myNMIACCV2(Hstar,Y,numclass);
    %     %-------------------------------
    %     % Numerical cleaning
    %     %-------------------------------
    %    Sigma(find(abs(Sigma<option.numericalprecision)))=0;
    %    Sigma = Sigma/sum(Sigma);
    
    %-----------------------------------------------------------
    % Enhance accuracy of line search if necessary
    %-----------------------------------------------------------
    if max(abs(Sigma-Sigmaold))<option.numericalprecision &&...
            option.goldensearch_deltmax > optiongoldensearchmax
        option.goldensearch_deltmax=option.goldensearch_deltmax/10;
    elseif option.goldensearch_deltmax~=goldensearch_deltmaxinit
        option.goldensearch_deltmax*10;
    end
    
    [grad] = localSimpleMKKMGrad(KH,NN,Hstar,Sigma,MM,lambda);
    %----------------------------------------------------
    % check variation of Sigma conditions
    %----------------------------------------------------
        if  max(abs(Sigma-Sigmaold))<option.seuildiffsigma
            loop = 0;
            fprintf(1,'variation convergence criteria reached \n');
        end
        
    
%     if nloop>=MaxIter
%         loop = 0;
%     end
    
    %-----------------------------------------------------
    % Updating Variables
    %----------------------------------------------------
    Sigmaold  = Sigma;
    [a,b,c,d,e,f]=myNMIACCV2(Hstar,Y,numclass);
    fprintf("%g,,%g,,%g,,%g,,%g,,%g\n",a(1),a(2),a(3),a(4),nloop,obj(nloop));
end