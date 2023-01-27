function [grad] = localSimpleMKKMGrad(KH,NN,Hstar,Sigma,MM,lambda)

d=size(KH,3);
grad=zeros(d,1);
for k=1:d
     grad(k) = 2*Sigma(k)*trace(Hstar'*(KH(:,:,k).*NN)*Hstar);
%      disp(size(Sigma));
%      disp(size(MM(:,k))); 
     temp=lambda*sum(Sigma.*MM(:,k)); 
     
     grad(k)=grad(k)+temp;
end