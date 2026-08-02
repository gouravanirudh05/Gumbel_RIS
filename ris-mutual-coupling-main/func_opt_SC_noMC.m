function XI = func_opt_SC_noMC(zRT,zRI,zIT,ZII)

Z0 = 50;
NI = length(zIT);

[~,~,~,sRT,sRI,sIT] = func_Z_to_YS(zRT,zRI,zIT,ZII);
theta = angle(sRT) - angle(sRI.' .* sIT);
Theta = diag(exp(1i * theta));
XI = imag(2 *Z0 * inv(eye(NI)-Theta) - Z0 * eye(NI));

% Theorem 1 in https://ieeexplore.ieee.org/document/9360851 (same XI as obtained through diag(diag(ZII)))
%XSS = real(ZII(1,1));
%a = zRI.' .* zIT / (2*abs(XSS));
%b = zRT - sum(a);
%theta_init = angle(b)/2 - angle(a)/2 + pi/2;
%XI1 = imag(diag(2*abs(XSS) ./ (1 + exp(1i*2*theta_init))) - diag(diag(ZII)));

end