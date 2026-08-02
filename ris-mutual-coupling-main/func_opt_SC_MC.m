function [XI,obj_trend] = func_opt_SC_MC(zRT,zRI,zIT,ZII)

nIter = 5000; % Max number of iterations 
eps = 1e-4;   % Threshold for convergence (1e-6)
delta = 30;   % Stepsize for the increment (6e-4)

NI = length(zIT);

% Initialization (Theorem 1 in https://ieeexplore.ieee.org/document/9360851)
XSS = real(ZII(1,1));
a = zRI.' .* zIT / (2*abs(XSS));
b = zRT - sum(a);
theta_init = angle(b)/2 - angle(a)/2 + pi/2;
XI = imag(diag(2*abs(XSS) ./ (1 + exp(1i*2*theta_init))) - diag(diag(ZII)));

D = zeros(NI);
b = zRT - zRI/(1i*XI + ZII)*zIT;
p = zRI/(1i*XI + ZII);
q = (1i*XI + ZII)\zIT;
obj = abs(b + p*(1i*imag(D))*q);

%XI_old = zeros(NI);
%D_old = zeros(NI);
%b_old = zRT - zRI/(1i*XI_old + ZII)*zIT;
%p_old = zRI/(1i*XI_old + ZII);
%q_old = (1i*XI_old + ZII)\zIT;
%obj_old = abs(b_old + p_old*(1i*imag(D_old))*q_old);
obj_old = abs(zRT - zRI/ZII*zIT);

obj_trend = nan(1,nIter);
iIter = 0;
while iIter < nIter && abs(obj-obj_old)/obj_old > eps
    iIter = iIter + 1;
    obj_old = obj;

    XI = XI + imag(D);
    b = zRT - zRI/(1i*XI + ZII)*zIT;
    p = zRI/(1i*XI + ZII);
    q = (1i*XI + ZII)\zIT;
    
    % Theorem 2 in https://ieeexplore.ieee.org/document/9360851
    d = delta * exp(1i * (angle(b) - angle(p.' .* q)));
    D = diag(d);
    obj = abs(b + p*(1i*imag(D))*q);
    
    obj_trend(iIter) = abs(zRT - zRI/(1i*XI+ZII)*zIT)^2;
end

%figure; plot(obj_trend);

end