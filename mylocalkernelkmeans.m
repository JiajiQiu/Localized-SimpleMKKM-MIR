function [H,obj]= mylocalkernelkmeans(K,A0,cluster_count,MM,Sigma,lambda)

opt.disp = 0;
K0 = A0.*K;
K0= (K0+K0')/2;
[H,~] = eigs(K0,cluster_count,'LA',opt);
temp=lambda/2*Sigma'*MM*Sigma;

obj = trace(H'*K0*H)+lambda/2*Sigma'*MM*Sigma;