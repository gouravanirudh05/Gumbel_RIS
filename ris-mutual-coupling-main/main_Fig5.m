clear; clc;
rng(3);
tic;

% Parameters
nMonte = 1e4; %1e4
ds = [0,1,2,3,4]; % Lambda / inter-element distance (0 = no mutual coupling)
NIs = 16:16:128; % Number of RIS elements
Z0 = 50;
gainRI = 4*Z0^2*1e-8;
gainIT = 4*Z0^2*1e-8;

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
P_UB = nan(nMonte,length(ds),length(NIs));
P_SL = nan(length(ds),length(NIs));
for id = 1:length(ds)
    d = ds(id);
    if d ~= 0, ZIIs = ZIIs_list{id}; end

    for iNI = 1:length(NIs)
        NI = NIs(iNI);
        if d ~= 0, ZII = ZIIs{iNI}; else, ZII = Z0 * eye(NI); end

        re_ZII_inv = inv(real(ZII));
        re_ZII_inv_sqrt = sqrtm(re_ZII_inv);

        for iMonte = 1:nMonte

            % Generate channels
            zRI = sqrt(gainRI/2) * (randn(1,NI) + 1i * randn(1,NI)); %sqrt(1/2) * (randn(1,NI) + 1i * randn(1,NI)) or exp(1i * 2 * pi * rand(1,NI))
            zIT = sqrt(gainIT/2) * (randn(NI,1) + 1i * randn(NI,1)); %sqrt(1/2) * (randn(NI,1) + 1i * randn(NI,1)) or exp(1i * 2 * pi * rand(NI,1))

            % Compute upper bound
            if d ~= 0
                P_UB(iMonte,id,iNI) = 1 / (16*Z0^2) * (abs(zRI*re_ZII_inv*zIT) + norm(zRI*re_ZII_inv_sqrt) * norm(re_ZII_inv_sqrt*zIT)) ^ 2;
            else
                P_UB(iMonte,id,iNI) = 1 / (16*Z0^4) * (abs(zRI*zIT) + norm(zRI) * norm(zIT)) ^ 2;
            end
        end

        % Compute scaling laws
        if d ~= 0
            P_SL(id,iNI) = gainRI * gainIT / (16*Z0^2) * (trace(re_ZII_inv^2) + trace(re_ZII_inv)^2 + sqrt(pi*trace(re_ZII_inv^2))*trace(re_ZII_inv));
        else
            P_SL(id,iNI) = gainRI * gainIT / (16*Z0^4) * (NI + NI^2 + sqrt(pi*NI)*NI);
        end

    end
end

P_UB_av = squeeze(mean(P_UB));

toc;

%% Plot
figure('defaultaxesfontsize',11)
LineW = 2; MarkS = 8;
hold on;
plot(NIs,10*log10(P_UB_av(5,:)),'-p','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','UB')
plot(NIs,10*log10(P_UB_av(4,:)),'-^','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','UB')
plot(NIs,10*log10(P_UB_av(3,:)),'-s','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','UB')
%plot(NIs,10*log10(P_UB_av(2,:)),'-d','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','UB')
plot(NIs,10*log10(P_UB_av(1,:)),'-o','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','UB')

plot(NIs,10*log10(P_SL(5,:)),'--h','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','Scaling law,{\it d = \lambda/4}')
plot(NIs,10*log10(P_SL(4,:)),'--v','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','Scaling law,{\it d = \lambda/3}')
plot(NIs,10*log10(P_SL(3,:)),'--d','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','Scaling law,{\it d = \lambda/2}')
%plot(NIs,10*log10(P_SL(2,:)),'--s','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','Scaling law,{\it d = \lambda}')
plot(NIs,10*log10(P_SL(1,:)),'--x','linewidth',LineW,'MarkerSize',MarkS,'DisplayName','Scaling law, no MC')
grid on; box on;
set(gca,'GridLineStyle',':','GridAlpha',0.8,'LineWidth',1.5);
xlabel('Number of RIS elements');
ylabel('Channel gain (dB)')
legend('location','northwest','numColumns',2,'FontSize',11);
ax = gca;
ax.XTick = 16:16:128;
ax.XLim = [64 128];
%ax.YTick =
ax.YLim = [-123 -115];
set(gcf, 'Color', [1,1,1]);
set(gca, 'LineWidth',1.5);