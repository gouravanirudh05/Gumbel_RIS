clear; clc;

% Parameters
x = 1; %Lambda / inter-element distance (1,2,3 or 4)

NIs = 16:16:128;
NI_ind1 = 8;
NI_ind2 = NIs / NI_ind1;

f = 28e9; % Frequency
c = 3e8; % Speed of light
lambda = c / f; % Wavelength
d = lambda / x; % Inter-element distance

ZIIs = cell(size(NI_ind2));
for iNI = 1:length(NI_ind2)
    NI_ind = [NI_ind1,NI_ind2(iNI)];
    ZIIs{iNI} = func_gen_ZII(NI_ind,lambda,d);
end

save(['ZIIs-d-lambda-',num2str(x)]);