
%inputs
iter = 15;
rho = 1.5;

% Splitting up the matrix to send to separate threads
N=2;
As=cell(1,N); % Cell to store separate A
bs=cell(1,N); % cell to store separate b
xs=cell(1,N); % cell to store separate x
us=cell(1,N); % cell to store separate x
[m,n]=size(A);
m2 = floor(m/N);
for i=1:N
    if i<N
        As{i} = A((i-1)*m2+1:i*m2,:);
        bs{i} = b((i-1)*m2+1:i*m2);
    else
        As{i} = A((i-1)*m2 +1:end,:);
        bs{i} = b((i-1)*m2 +1:end);
    end
    xs{i} = zeros(n,1);
    us{i} = zeros(n,1);
end

%Initializing z
z=zeros(n,1);


for i=1:iter
    xbar = zeros(n,1);
    ubar = zeros(n,1);
    % send each update to a different thread
    for j=1:N
        xs{j} = (As{j}'*As{j} + rho/2)\(As{j}'*bs{j} - (rho/2)*(-z-us{j}));
        xs{j} = xs{j}./norm(xs{j});
        xbar = xbar + xs{j};
        ubar = ubar + us{j};
    end
    xbar = xbar./N;
    ubar = ubar./N;
    z = xbar + ubar;
    %z=z./norm(z);
    
    for j=1:N
       us{j} =  us{j} + xs{j} - z;
    end
    
end

%output 
z