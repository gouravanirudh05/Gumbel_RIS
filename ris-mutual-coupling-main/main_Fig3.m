clear; clc;
rng(3);
tic;

% Parameters
nMonte = 2e3; %2e3
ds = [0,1,2,3,4]; % Lambda / inter-element distance (0 = no mutual coupling)
NIs = 16:16:128; % Number of RIS elements
gain = 1e-16;
Z0 = 50;
Y0 = 1 / Z0;

% Load mutual coupling matrices
ZIIs_list = cell(size(ds));
for id = 1:length(ds)
    d = ds(id);
    if d ~= 0
        ZIIs_struct = load(['ZIIs-d-lambda-',num2str(ds(id)),'.mat'],'ZIIs');
        ZIIs_list{id} = ZIIs_struct.ZIIs;
    end
end

% Main loop
P_FC = nan(nMonte,length(ds),length(NIs));
P_TC = nan(nMonte,length(ds),length(NIs));
P_UB = nan(nMonte,length(ds),length(NIs));
for iMonte = 1:nMonte
    if mod(iMonte,100) == 0, fprintf(['iMonte: ',num2str(iMonte),'\n']); end

    for id = 1:length(ds)
        d = ds(id);
        if d ~= 0, ZIIs = ZIIs_list{id}; end

        for iNI = 1:length(NIs)
    
            NI = NIs(iNI);
            if d ~= 0, ZII = ZIIs{iNI}; else, ZII = Z0 * eye(NI); end

            % Generate channels and useful matrices
            zRT = 0;
            zRI = 2 * Z0 * sqrt(1/2) * (randn(1,NI) + 1i * randn(1,NI)); %sqrt(1/2) * (randn(1,NI) + 1i * randn(1,NI)) or exp(1i * 2 * pi * rand(1,NI))
            zIT = 2 * Z0 * sqrt(1/2) * (randn(NI,1) + 1i * randn(NI,1)); %sqrt(1/2) * (randn(NI,1) + 1i * randn(NI,1)) or exp(1i * 2 * pi * rand(NI,1))

            [yRT,yRI,yIT,sRT,sRI,sIT] = func_Z_to_YS(zRT,zRI,zIT,ZII);

            re_ZII_inv = inv(real(ZII));
            re_ZII_sqrt = sqrtm(real(ZII));
            re_ZII_inv_sqrt = sqrtm(re_ZII_inv);
    
            YII = inv(ZII);
            re_YII_inv = inv(real(YII));
            re_YII_inv_sqrt = sqrtm(re_YII_inv);

            % Optimize fully-connected RIS
            zRI_bar_FC = zRI * re_ZII_inv_sqrt * sqrt(Z0);
            zIT_bar_FC = sqrt(Z0) * re_ZII_inv_sqrt * zIT;

            sRI_bar_FC = zRI_bar_FC / (2*Z0);
            sIT_bar_FC = zIT_bar_FC / (2*Z0);
            sRT_bar_FC = (zRT - zRI_bar_FC * zIT_bar_FC / (2*Z0)) / (2*Z0);

            sRI_hat_FC = sRI_bar_FC / norm(sRI_bar_FC);
            sIT_hat_FC = sIT_bar_FC / norm(sIT_bar_FC);
            sRT_hat_FC = sRT_bar_FC / (norm(sRI_bar_FC) * norm(sIT_bar_FC));

            Theta_bar = func_opt_FC_MC(sRT_hat_FC, sRI_hat_FC, sIT_hat_FC);

            XI_bar = real(-1i * (2 *Z0 * inv(eye(NI)-Theta_bar) - Z0 * eye(NI)));

            XI = re_ZII_sqrt * XI_bar * re_ZII_sqrt / Z0 - imag(ZII);

            P_FC(iMonte,id,iNI) = gain / (4*Z0^2) * abs(zRT - zRI / (1i * XI + ZII) * zIT) ^ 2;

            % Optimize tree-connected RIS
            yRI_bar_TC = yRI * re_YII_inv_sqrt * sqrt(Y0);
            yIT_bar_TC = sqrt(Y0) * re_YII_inv_sqrt * yIT;

            sRI_bar_TC = - yRI_bar_TC / (2*Y0);
            sIT_bar_TC = - yIT_bar_TC / (2*Y0);
            sRT_bar_TC = - (yRT - yRI_bar_TC * yIT_bar_TC / (2*Y0)) / (2*Y0);

            sRI_hat_TC = sRI_bar_TC / norm(sRI_bar_TC);
            sIT_hat_TC = sIT_bar_TC / norm(sIT_bar_TC);
            sRT_hat_TC = sRT_bar_TC / (norm(sRI_bar_TC) * norm(sIT_bar_TC));
            
            [BI, ~, ~] = func_opt_TC_tridiag_MC(sRT_hat_TC, sRI_hat_TC, sIT_hat_TC, YII); % func_opt_TC_tridiag_MC or func_opt_TC_arrow_MC
            
            P_TC(iMonte,id,iNI) = gain / (4*Y0^2) * abs(yRT - yRI / (1i * BI + YII) * yIT) ^ 2;
    
            % Compute theoretical upper bound
            if d ~= 0
                P_UB(iMonte,id,iNI) = gain / (4*Y0^2) * (abs(yRT-yRI*re_YII_inv*yIT/2) + norm(yRI*re_YII_inv_sqrt) * norm(re_YII_inv_sqrt*yIT) / 2) ^ 2;
                % or gain / (4*Z0^2) * (abs(zRT-zRI*re_ZII_inv*zIT/2) + norm(zRI*re_ZII_inv_sqrt) * norm(re_ZII_inv_sqrt*zIT) / 2) ^ 2;
            else
                P_UB(iMonte,id,iNI) = gain / (4*Y0^2) * (abs(yRT-yRI*yIT/(2*Y0)) + norm(yRI) * norm(yIT) / (2*Y0)) ^ 2;
                % or gain / (4*Z0^2) * (abs(zRT-zRI*zIT/(2*Z0)) + norm(zRI) * norm(zIT) / (2*Z0)) ^ 2;
            end
        end
    end
end

P_FC_av = squeeze(mean(P_FC));
P_TC_av = squeeze(mean(P_TC));
P_UB_av = squeeze(mean(P_UB));

toc;

%% Plot
figure('defaultaxesfontsize',11)
LineW = 2; MarkS = 8;
hold on;
plot(NIs,10*log10(P_FC_av(5,:)),'-','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','Fully-conn.')
plot(NIs,10*log10(P_FC_av(4,:)),'-','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','Fully-conn.')
plot(NIs,10*log10(P_FC_av(3,:)),'-','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','Fully-conn.')
%plot(NIs,10*log10(P_FC_av(2,:)),'-','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','Fully-conn.')
plot(NIs,10*log10(P_FC_av(1,:)),'-','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','Fully-conn.')

plot(NIs,10*log10(P_TC_av(5,:)),'--','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','Tree-conn.')
plot(NIs,10*log10(P_TC_av(4,:)),'--','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','Tree-conn.')
plot(NIs,10*log10(P_TC_av(3,:)),'--','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','Tree-conn.')
%plot(NIs,10*log10(P_TC_av(2,:)),'--','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','Tree-conn.')
plot(NIs,10*log10(P_TC_av(1,:)),'--','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','Tree-conn.')

plot(NIs,10*log10(P_UB_av(5,:)),'kp','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','UB,{\it d = \lambda/4}')
plot(NIs,10*log10(P_UB_av(4,:)),'k^','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','UB,{\it d = \lambda/3}')
plot(NIs,10*log10(P_UB_av(3,:)),'ks','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','UB,{\it d = \lambda/2}')
%plot(NIs,10*log10(P_UB_av(2,:)),'kd','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','UB,{\it d = \lambda}')
plot(NIs,10*log10(P_UB_av(1,:)),'ko','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','UB, no MC')

%plot(NIs,10*log10(gain*(NIs.^2+NIs+sqrt(pi*NIs).*NIs)),'-k','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','UB, no MC')
grid on; box on;
set(gca,'GridLineStyle',':','GridAlpha',0.8,'LineWidth',1.5);
xlabel('Number of RIS elements');
ylabel('Channel gain (dB)')
legend('location','northwest','numColumns',3,'FontSize',11);
ax = gca;
ax.XTick = 16:16:128;
ax.XLim = [64 128];
%ax.YTick =
ax.YLim = [-123 -115];
set(gcf, 'Color', [1,1,1]);
set(gca, 'LineWidth',1.5);