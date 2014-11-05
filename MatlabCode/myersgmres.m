function [x]=myersgmres(A,b,n)
% Basic GMRES with residiual plot

q=b/norm(b);
q=q';
%r=zeros(10,1);
[m,~]=size(A);
Q(1:m,1)=q;
tic
for i = 1:n
    v=A*Q(:,i);
    for j=1:i
        h(j,i)=Q(:,j)'*v;
        v=v-h(j,i)*Q(:,j);
    end
    h(i+1,i)=norm(v);
    Q(:,i+1)=v/h(i+1,i);
    %Now we run GMRES minimize ||Hn*y-||b||*e1||
    e=eye(1,i+1);
    e=e';
    [P,L]=qr(h);
    y=L\(P'*norm(b)*e);
    x=Q(:,1:i)*y;
    r(i)=norm(A*x-b')/norm(b);
end
toc
plot([1:n],r([1:n]),'-');
xlabel('Iteration number');
ylabel('Norm of the residual Ax-b');
end