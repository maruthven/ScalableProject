%% Biconjugate Gradient stabilized method
% Aaron Myers Fall 2014 for Machine Learning Project

function [xf]=BiCGSTAB(A,b,k)
[m,n]=size(A);
x(:,1)=b;
r(:,1)=b-A*x(:,1);
rh(:,1) = rand(m,1);
while rh(:,1)'*r(:,1) ==0
	rh(:,1) = rand(m,1);
end

rho(1) = 1;
al = 1;
w(1) = 1;
v(:,1) = zeros(m,1);
p(:,1) = zeros(m,1);

r(:,2)= r(:,1);

i=2;
tic;
while norm(r(:,2))> k
	if i>2
		r(:,1) = r(:,2);
		rho(1)=rho(2);
		w(1)=w(2);
		v(:,1)=v(:,2);
		p(:,1)=p(:,2);
		x(:,1) = x(:,2);
	end
	rho(2) = rh(:,1)'*r(:,1);
	bet = rho(2)/rho(1) * al/w(1);
	p(:,2) = r(:,1) + bet*(p(:,1) - w(1)*v(:,1));
	v(:,2) = A*p(:,2);
	al = rho(2)/(rh(:,1)'*v(:,2));
	s = r(:,1) - al*v(:,2);
	t = A*s;
	w(2) = (t'*s) / (t'*t);
	x(:,2) = x(:,1) + al*p(:,2) + w(2)*s;
	r(:,2) = s - w(2)*t;
	i = 3;
end
x
xf = x(:,2);
toc

end
