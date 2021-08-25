N = 8202;
f = 102.1;
Fs  = 8000;
Q = 68;

w = hamming(N);
seg = rand([N 1]);

%{
tic 
num1=0;
for n=1:N
    num1 = num1+ w(n)*seg(n)*exp( -1j*2*pi*f*(n-1)/Fs);
end
toc
%}

%{
tic 
xx = [0:N-1]';
e = exp( -1j * 2* pi * f *xx /Fs);
num2 = sum( w.*seg.*e);
toc
clear e xx
%}

%{
tic
xx = 0:N-1;
e = diag(exp( -1j * 2* pi * f *xx /Fs));
num = seg.'*e*w;
toc

clear e
%}
%{
tic
xx = 0:N-1;
e = exp( -1j * 2* pi * f/Fs *xx);
num3=0;
for n=1:N
    num3 = num3+ w(n)*seg(n)*e(n);
end
toc
%}

%{
tic

for k=1:500
    xx = (0:N-1)';
    omega = -1i * 2* pi *f /Fs;
    e = exp( omega * xx);
    num3 = sum( w.*seg.*e);
end
toc

clear e xx omega
%}

%{
tic

for k=1:1000
    xx = 0:N-1;
    omega = -1i * 2* pi *f /Fs;
    e = exp( omega * xx);
    
    wseg = w.*seg;
    num4 = e*wseg;
end
toc

clear e xx omega
%}

%{
tic
for k=1:1000
%     xx = 0:N-1;
%     omega = -1i * 2* pi *f /Fs;
    e = exp( -1j * 2* pi * f/Fs *(0:N-1));
    
%     wseg = ;
    num5 = e*(w.*seg);
end
toc
clear e 
%}

%{
tic
for k=1:1000


        e1 = exp( -1j * 2* pi * f/Fs *(0:Fs-1));

    num6 = [e1 e1(1: N-Fs )]*(w.*seg);
end
toc
clear e1 
%}

%{
x = 11;
tic
for k=1:1000
    val1  = exp( -1j * 2* pi *f/Fs * x);
end
toc

tic
for k=1:1000
    val2  = exp( -1j * 2* pi * f/Fs * (x+ 10*round(Fs/f) ));
end
toc
clear x
%}

tic
for  k =1:1000
    y1  = exp( -1j * 2*pi * Q/N * (0:N-1));
end
toc

%{
tic
for k =1:1000
    q = Q; n=N; i = 0;
    while ~rem(q,2) && ~rem(n,2)
        q = q/2; n = n/2; 
        i = i+1;
    end
    y_  = exp( -1j * 2*pi * Q/N * (0:n-1));
    
    y2 = repmat(y_, [1, 2^i]);
end
toc
%}

%{
tic
for k =1:1000
    q = Q; n=N; i = 0;
    while ~rem(q,2) && ~rem(n,2)
        q = q/2; n = n/2; 
        i = i+1;
    end
    e  = exp( -1j * 2*pi * Q/N * (0:n-1));
    
    x = w.*seg;
    z2 = 0;
%     Z = zeros( 2^i, N);
    L = length(e);
    for j = 1:(2^i)
        z2 = z2+ [zeros(1, L*(j-1)) e zeros(1, L*(2^i-j ))]* x/N;
%         Z(j, L*(j-1)+1: L*j) = e;
    end
%     z2 = ones(1, 2^i)*Z*x/N;
    
end
toc
clear e x
%}

tic
for k =1:1000
    q = Q; n=N; i = 0;
    while ~rem(q,2) && ~rem(n,2)
        q = q/2; n = n/2; 
        i = i+1;
    end
    e  = exp( -1j * 2*pi * Q/N * (0:n-1));
    
    if i>0
        z3 =0; x = w.*seg;
        L = length(e);
        for j = 1:(2^i)
            z3 = z3+ e* x( L*(j-1)+1: L*j )/N;
        end
    else
        z3 = e*(w.*seg)/N;
    end
end
toc
clear e

tic
for k =1:1000
    q = Q; n=N; i = 0;
    while ~rem(q,2) && ~rem(n,2)
        q = q/2; n = n/2; 
        i = i+1;
    end
    e  = exp( -1j * 2*pi * Q/N * (0:n-1));
    
    
    for j = 1:i
        e = [e e];
    end
    
    z4 = e*(w.*seg)/N;
end
toc
clear e

%{
tic
for k =1:1000
%     q = Q; n=N; i = 0;
    G = gcd(Q, N);
%     while ~rem(q,2) && ~rem(n,2)
%         q = q/2; n = n/2; 
%         i = i+1;
%     end
    e  = exp( -1j * 2*pi * Q/N * (0:(N/G-1)));
    
    
    z5 =0; x = w.*seg;
    L = length(e);
    for j = 1:G
        z5 = z5+ e* x( L*(j-1)+1: L*j )/N;
    end
end
toc
%}

%{
tic
for k =1:1000
    q = Q; n=N; %i = 0;
    for i=0:100
        if rem(q,2) || rem(n,2)
            break;
        end
        
        q = q/2; n = n/2; 
%         i = i+1;
    end
    y4  = exp( -1j * 2*pi * Q/N * (0:n-1));
    
    
    for j = 1:i
        y4 = [y4 y4];
    end
end
toc
%}

%{
tic

omega = -1j * 2* pi *f /Fs;
xx = (0:N-1)' * omega;
e = exp( xx);
num4 = sum( w.*seg.*e);
toc

clear e xx
%}

%{
tic
basis = -1j * 2* pi *f /Fs * (0:N-1)';
e = exp( basis);
num4 = sum( w.*seg.*e);
toc
%}

%{
tic
xx = (0:N-1)';
w1 = 0.54 - 0.46*cos(2*pi*xx);
toc

clear xx

tic
xx = (0:N-1)';
toc

clear xx 

tic
yy = 0:N-1;
xx = yy(:);
toc
%}

