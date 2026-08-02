function [B, Theta, Obj] = func_opt_TC_tridiag_MC(sRT_hat, sRI_hat, sIT_hat, YII)

NI = size(sIT_hat,1);
Y0 = 1/50;

im_YII = imag(YII);
re_YII_sqrt = sqrtm(real(YII));
re_YII_inv_sqrt = sqrtm(inv(real(YII)));

% Vectors alpha and beta
a_bar = 1i * (sIT_hat + exp(1i * angle(sRT_hat)) * sRI_hat');
b_bar = Y0 * (sIT_hat - exp(1i * angle(sRT_hat)) * sRI_hat');

a = re_YII_inv_sqrt * a_bar;
b = (1/Y0) * re_YII_sqrt * b_bar - im_YII * re_YII_inv_sqrt * a_bar;

% Matrix A
A = zeros(2*NI,3*NI);

% First N columns
A(1:NI,     1:NI)=diag(real(a));
A(NI+1:2*NI,1:NI)=diag(imag(a));

% Second N-1 columns
A(1:NI,     NI+1:2*NI)=diag(real([a(2:NI);0])) + diag(real(a(1:NI-1)),-1);
A(NI+1:2*NI,NI+1:2*NI)=diag(imag([a(2:NI);0])) + diag(imag(a(1:NI-1)),-1);
A = A(:,1:2*NI-1);

% Solve the linear system
x = A \ [real(b);imag(b)]; % Same as pinv(A) * [real(b);imag(b)] and inv(A'*A)*A' * [real(b);imag(b)]

% Matrix B
D1 = x(1:NI);
D2 = x(NI+1:end);
B = diag(D1) + diag(D2,-1) + diag(D2,+1);



Theta = (eye(NI) + 1i*50*B) \ (eye(NI) - 1i*50*B); % Scattering matrix

Obj = abs(sRT_hat + sRI_hat*Theta*sIT_hat) ^ 2; % Channel gain

end