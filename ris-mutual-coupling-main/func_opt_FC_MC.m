function Theta = func_opt_FC_MC(sRT_hat, sRI_hat, sIT_hat)

NI = size(sIT_hat,1);

% Matrix A
RRI = sRI_hat' * sRI_hat;
RIT = sIT_hat * sIT_hat';
ARI = (RRI + RRI.') / 2;
AIT = (RIT + RIT.') / 2;
A = ARI - AIT;

% Eigenvalue decomposition of A
[U,Delta] = eig(A);
delta = flip(diag(Delta)); % Order delta in decreasing order

% Compute matrix T distinguishing three cases
T = zeros(NI);

if NI == 2

    T = [sqrt(1/2), sqrt(1/2);
         sqrt(1/2), -sqrt(1/2)];

elseif NI == 3

    T = [sqrt(-delta(3)/(delta(1)-delta(3))), sqrt(delta(1)/(2*(delta(1)-delta(3)))), -sqrt(delta(1)/(2*(delta(1)-delta(3))));
         0, sqrt(1/2), sqrt(1/2);
         sqrt(delta(1)/(delta(1)-delta(3))), -sqrt(-delta(3)/(2*(delta(1)-delta(3)))), sqrt(-delta(3)/(2*(delta(1)-delta(3))))];

else

    T(1,1) = sqrt(-delta(NI-1) / (delta(1) - delta(NI-1)));
    T(NI-1,1) = sqrt(delta(1) / (delta(1) - delta(NI-1)));
    T(2,2) = sqrt(-delta(NI) / (delta(2) - delta(NI)));
    T(NI,2) = sqrt(delta(2) / (delta(2) - delta(NI)));
    T(1,3) = sqrt(1/2) * T(NI-1,1);
    T(2,3) = sqrt(1/2) * T(NI,2);
    T(NI-1,3) = -sqrt(1/2) * T(1,1);
    T(NI,3) = -sqrt(1/2) * T(2,2);
    T(1,4) = sqrt(1/2) * T(NI-1,1);
    T(2,4) = -sqrt(1/2) * T(NI,2);
    T(NI-1,4) = -sqrt(1/2) * T(1,1);
    T(NI,4) = sqrt(1/2) * T(2,2);

    T(3:NI-2,5:NI) = eye(NI-4);

end

% Compute matrices V, D, and Theta
V = flip(U,2) * T; % Order U according to delta
hRI_bar = sRI_hat * V;
hIT_bar = V' * sIT_hat;
theta = - angle(hRI_bar) - angle(hIT_bar.');
d = exp(1i * theta);
D = diag(d);
Theta_tmp = V * D * V';

Theta = exp(1i * angle(sRT_hat)) * Theta_tmp;

end