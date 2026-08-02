function ZII = func_gen_ZII(NI_ind,lambda,d)

% Input (dipoles parallel to y axis):
% NI_ind(1): Number of RIS elements along x
% NI_ind(2): Number of RIS elements along y
% lambda: Wavelength
% d: Inter-element distance

% Parameters
l = lambda/4; % Length of dipoles (lambda/32)
%r = lambda/500; % Radius of dipoles (lambda/500)
eta0 = 377; % Impedance of free space
k0 = 2*pi/lambda; % Wavenumber

% Place the NI RIS elements in the xy plane
NI = NI_ind(1)*NI_ind(2);
loc_xy = zeros(NI,2);
for nx = 1 : NI_ind(1)
    for ny = 1 : NI_ind(2)
        ind = (ny-1)*NI_ind(1) + nx;
        loc_xy(ind,:) = [nx-1,ny-1]*d;
    end
end

% Compute the entries of ZII
ZII = zeros(NI,NI);
for q = 1 : NI
    for p = 1 : NI
        if q == p
            %d_x = r^2;
            ZII(q,p) = 50;
        else
            d_x = (loc_xy(q,1) - loc_xy(p,1))^2;
            fun = @(y2,y1) cost(y1,y2,d_x,loc_xy(q,2),loc_xy(p,2));
            ZII(q,p) = integral2(fun,loc_xy(q,2)-l/2,loc_xy(q,2)+l/2,...
                                     loc_xy(p,2)-l/2,loc_xy(p,2)+l/2);
        end
    end
end

function obj = cost(y1,y2,d_x,y_q,y_p)
    d_qp = sqrt(d_x + (y2-y1).^2);
    obj = 1j*eta0/(4*pi*k0)...
          .*((y2-y1).^2./d_qp.^2 .* (3./d_qp.^2 + 1j*3*k0./d_qp - k0^2) - (1j*k0 + 1./d_qp)./d_qp + k0.^2)...
          .*exp(-1j*k0.*d_qp)./d_qp...
          .*sin(k0*(l/2-abs(y1-y_p)))./sin(k0*l/2)...
          .*sin(k0*(l/2-abs(y2-y_q)))./sin(k0*l/2);
end

end