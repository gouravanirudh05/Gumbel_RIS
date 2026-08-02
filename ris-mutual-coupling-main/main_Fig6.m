clear; clc;
rng(3);
tic;

% Parameters
nMonte = 2e3; %2e3
ds = [0,1,2,3,4]; % Lambda / inter-element distance (0 = no mutual coupling)
NIs = [128]; % Number of RIS elements
gain = 1e-16;
Z0 = 50;

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
P_BDRIS_aw = nan(nMonte,length(ds),length(NIs));
P_BDRIS_FC_un = nan(nMonte,length(ds),length(NIs));
P_BDRIS_TC_un = nan(nMonte,length(ds),length(NIs));
P_DRIS_aw = nan(nMonte,length(ds),length(NIs));
P_DRIS_un = nan(nMonte,length(ds),length(NIs));
for id = 1:length(ds)
    d = ds(id);
    if d ~= 0, ZIIs = ZIIs_list{id}; end

    for iNI = 1:length(NIs)
        NI = NIs(iNI);
        if d ~= 0, ZII = ZIIs{NI/16}; else, ZII = Z0 * eye(NI); end
        
        re_ZII_inv = inv(real(ZII));
        re_ZII_inv_sqrt = sqrtm(re_ZII_inv);

        P_BDRIS_aw_tmp = nan(nMonte,1);
        P_BDRIS_FC_un_tmp = nan(nMonte,1);
        P_BDRIS_TC_un_tmp = nan(nMonte,1);
        P_DRIS_aw_tmp = nan(nMonte,1);
        P_DRIS_un_tmp = nan(nMonte,1);
        parfor iMonte = 1:nMonte

            % Generate channels and useful matrices
            zRT = 0;
            zRI = 2 * Z0 * sqrt(1/2) * (randn(1,NI) + 1i * randn(1,NI));
            zIT = 2 * Z0 * sqrt(1/2) * (randn(NI,1) + 1i * randn(NI,1));

            % Optimize BD-RIS MC unaware
            [yRT,yRI,yIT,sRT,sRI,sIT] = func_Z_to_YS(zRT,zRI,zIT,diag(diag(ZII))); % ZII or diag(diag(ZII))
    
            sRI_hat = sRI / norm(sRI);
            sIT_hat = sIT / norm(sIT);
            sRT_hat = sRT / (norm(sRI) * norm(sIT));
    
            % Optimize fully-connected RIS MC unaware
            Theta = func_opt_FC_MC(sRT_hat, sRI_hat, sIT_hat);
    
            XI = real(-1i * (2 *Z0 * inv(eye(NI)-Theta) - Z0 * eye(NI)));
    
            P_BDRIS_FC_un_tmp(iMonte) = gain / (4*Z0^2) * abs(zRT - zRI / (1i * XI + ZII) * zIT) ^ 2;
    
            % Optimize tree-connected RIS MC unaware
            [BI, ~, ~] = func_opt_TC_tridiag_noMC(sRT_hat, sRI_hat, sIT_hat); % func_opt_TC_tridiag_noMC or func_opt_TC_arrow_noMC
    
            P_BDRIS_TC_un_tmp(iMonte) = gain / (4*Z0^2) * abs(zRT - zRI / (-1i * inv(BI) + ZII) * zIT) ^ 2;

            % Optimize D-RIS MC unaware
            XI_un = func_opt_SC_noMC(zRT, zRI, zIT, diag(diag(ZII))); % ZII or diag(diag(ZII))
            P_DRIS_un_tmp(iMonte) = gain / (4*Z0^2) * abs(zRT - zRI / (1i*XI_un + ZII) * zIT) ^ 2;

            % BD-RIS and D-RIS MC aware
            if d ~= 0
                P_BDRIS_aw_tmp(iMonte) = gain / (4*Z0^2) * (abs(zRT-zRI*re_ZII_inv*zIT/2) + norm(zRI*re_ZII_inv_sqrt) * norm(re_ZII_inv_sqrt*zIT) / 2) ^ 2;
                [XI,obj_trend] = func_opt_SC_MC(zRT, zRI, zIT, ZII);
                P_DRIS_aw_tmp(iMonte) = gain / (4*Z0^2) * abs(zRT - zRI / (1i*XI + ZII) * zIT) ^ 2;
            else
                P_BDRIS_aw_tmp(iMonte) = gain / (4*Z0^2) * (abs(zRT-zRI*zIT/(2*Z0)) + norm(zRI) * norm(zIT) / (2*Z0)) ^ 2;
                P_DRIS_aw_tmp(iMonte) = gain / (16*Z0^4) * (abs(zRT-zRI*zIT) + abs(zRI) * abs(zIT)) ^ 2;
            end
        end
        P_BDRIS_aw(:,id,iNI) = P_BDRIS_aw_tmp;
        P_BDRIS_FC_un(:,id,iNI) = P_BDRIS_FC_un_tmp;
        P_BDRIS_TC_un(:,id,iNI) = P_BDRIS_TC_un_tmp;
        P_DRIS_aw(:,id,iNI) = P_DRIS_aw_tmp;
        P_DRIS_un(:,id,iNI) = P_DRIS_un_tmp;
    end
end

P_BDRIS_aw_av = squeeze(mean(P_BDRIS_aw));
P_BDRIS_FC_un_av = squeeze(mean(P_BDRIS_FC_un));
P_BDRIS_TC_un_av = squeeze(mean(P_BDRIS_TC_un));
P_DRIS_aw_av = squeeze(mean(P_DRIS_aw));
P_DRIS_un_av = squeeze(mean(P_DRIS_un));

toc;

%% Plot
figure('defaultaxesfontsize',11)
LineW = 2; MarkS = 8;
hold on;
plot(ds,10*log10(P_BDRIS_aw_av),'-p','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','BD-RIS, MC aware')
plot(ds,10*log10(P_BDRIS_FC_un_av),':s','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','BD-RIS (FC), MC unaware')
plot(ds,10*log10(P_BDRIS_TC_un_av),'--d','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','BD-RIS (TC), MC unaware')
plot(ds,10*log10(P_DRIS_aw_av),'-h','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','D-RIS, MC aware')
plot(ds,10*log10(P_DRIS_un_av),'-.o','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','D-RIS, MC unaware')
grid on; box on;
set(gca,'GridLineStyle',':','GridAlpha',0.8,'LineWidth',1.5);
xlabel('Inter-element distance');
ylabel('Channel gain (dB)')
legend('location','southwest','numColumns',1,'FontSize',11);
ax = gca;
ax.XTick = 0:1:4;
ax.XTickLabel = {'no MC','{\it\lambda}','{\it\lambda/2}','{\it\lambda/3}','{\it\lambda/4}'};
ax.YTick = -128:-115;
ax.YLim = [-122.8 -115.8]; % NIs=64: [-128.6 -121.6], NIs=128: [-122.8 -115.8]
set(gcf, 'Color', [1,1,1]);
set(gca, 'LineWidth',1.5);