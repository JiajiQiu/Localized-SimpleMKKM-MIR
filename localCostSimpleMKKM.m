function [cost,Hstar] = localCostSimpleMKKM(KH,StepSigma,DirSigma,NN,Sigma,numclass,MM,lambda)

global nbcall
nbcall=nbcall+1;
Sigma = Sigma+ StepSigma * DirSigma;
Kmatrix = sumKbeta(KH,(Sigma.*Sigma));
[Hstar,cost]= mylocalkernelkmeans(Kmatrix,NN,numclass,MM,Sigma,lambda);