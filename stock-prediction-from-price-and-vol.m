% TODO
% How to choose best net size?
% How to choose best time series size?
% How to sustain the free run 

%% Read data from file
priceall = csvread('FTSE100_20121203_20171203.dat');
tr = priceall(1:ceil(length(priceall)*0.75));
trvol = priceall(1:ceil(length(priceall)*0.75),2)';
ts = priceall(ceil((length(priceall)*0.75)+1):length(priceall));
tsvol = priceall(ceil((length(priceall)*0.75)+1):length(priceall),2)';

%% standardise
m_tr = mean(tr);
m_trvol = mean(trvol);
m_ts = mean(ts);
m_tsvol = mean(tsvol);
sd_tr = std(tr);
sd_trvol = std(trvol);
sd_ts = std(ts);
sd_tsvol = std(tsvol);

tr = (tr-m_tr)/sd_tr;
trvol = (trvol-m_trvol)/sd_trvol;
ts = (ts-m_tr)/sd_tr;

tsvol = (tsvol-m_trvol)/sd_trvol;

%% construct matrix
Ntr = length(tr);
p = 21;
p_feature = p-1;
row_num_tr = Ntr-p+1;

Nts = length(ts);
row_num_ts = Nts-p+1;

%% Prepare Train and Test data
% Prepare Train data
i = 1;
Xtemp = zeros(row_num_tr,p);
Xtemp1 = zeros(row_num_tr,p);
while i <= row_num_tr
    Xtemp(i,:) = tr(i:(p-1)+i);
    Xtemp1(i,:) = trvol(i:(p-1)+i);
    i = i+1 ;
end
% feature = first 20 columns
Xtr = [Xtemp(:,1:p_feature) Xtemp1(:,1:p_feature)];
% target = last column
ftr = Xtemp(:,p);
volftr = Xtemp1(:,p);

% Prepare Test data
i = 1;
Xtemp = zeros(row_num_ts,p);
Xtemp1 = zeros(row_num_ts,p);
while i <= row_num_ts
    Xtemp(i,:) = ts(i:(p-1)+i);
    Xtemp1(i,:) = tsvol(i:(p-1)+i);
    i = i+1 ;
end
Xts = [Xtemp(:,1:p_feature) Xtemp1(:,1:p_feature)];
fts = Xtemp(:,p);
volfts = Xtemp1(:,p);

%% Train Neural Network, target price
[trainInd,valInd,testInd] = dividetrain(3000);
[net] = feedforwardnet(20);
net.divideFcn = 'divideblock';
% net.divideParam.trainRatio = 1;
% net.divideParam.valRatio = 0;
% net.divideParam.testRatio = 0;
[net] = train(net, Xtr', ftr');

% fhtr = net(Xtr');
fhts = net(Xts');

%% Train Neural Network, target vol
[volnet] = feedforwardnet(20);
volnet.divideFcn = 'divideblock';
% volnet.divideParam.trainRatio = 1;
% volnet.divideParam.valRatio = 0;
% volnet.divideParam.testRatio = 0;
[volnet] = train(volnet, Xtr', volftr');

% fhtr = net(Xtr');
% volfhts = volnet(Xts');


% Train Linear, target price
gama = 0.5;
cvx_begin quiet
    variable w( p_feature + p_feature );
    % ||Ya - f||^2 + gama||a||^2
    minimize( norm(Xtr*w-ftr) + gama*norm(w,1) );
cvx_end

% Train Linear, target vol
gama = 0.5;
cvx_begin quiet
    variable volw( p_feature + p_feature );
    % ||Ya - f||^2 + gama||a||^2
    minimize( norm(Xtr*volw-volftr) + gama*norm(volw,1) );
cvx_end

lrfhts = Xts*w;
% vollrfhts = Xts*volw;

% figure(), clf
% hold on, grid on;
% plot(ftr, 'DisplayName', 'Actual');
% plot(fhtr, 'DisplayName', 'ANN One-day-ahead');
% hold off;
% xlabel('Time series', 'FontSize', 12)
% ylabel('Values', 'FontSize', 12)
% title('Train set', 'FontSize', 14)
% legend('show');

figure(1), clf
hold on, grid on;
plot(fts, 'DisplayName', 'Actual');
plot(lrfhts, 'DisplayName', 'LR One-day-ahead');
plot(fhts, 'DisplayName', 'ANN One-day-ahead');
% yL = get(gca,'YLim');
% line([row_num_ts row_num_ts],yL,'Color','r');

hold off;
xlabel('Time series', 'FontSize', 12)
ylabel('Values', 'FontSize', 12)
title('Test set', 'FontSize', 14)
legend('show');


% figure(), clf
% hold on, grid on;
% plot(ftr,fhtr,'bx');
% hold off;
% xlabel('Actual', 'FontSize', 12)
% ylabel('Predicted', 'FontSize', 12)
% title('Train set: Actual vs Predicted', 'FontSize', 14)
% legend('show');

figure(2), clf
hold on, grid on;
plot(fts,fhts,'rx');
hold off;
xlabel('Actual', 'FontSize', 12)
ylabel('ANN One-day-ahead', 'FontSize', 12)
title('Test set: Actual vs ANN One-day-ahead', 'FontSize', 14)
legend('show');

figure(3), clf
hold on, grid on;
plot(fts,lrfhts,'rx');
hold off;
xlabel('Actual', 'FontSize', 12)
ylabel('LR One-day-ahead', 'FontSize', 12)
title('Test set: Actual vs LR One-day-ahead', 'FontSize', 14)
legend('show');

%% Free running mode

% % Train Linear
% gama = 0.5;
% cvx_begin quiet
%     variable w( p_feature );
%     % ||Ya - f||^2 + gama||a||^2
%     minimize( norm(Xtr*w-ftr) + gama*norm(w,1) );
% cvx_end

% %% Predict train data 
% Xstart = Xtr(1, 1:p_feature);
% ann_tr_predict = Xstart;
% lr_tr_predict = Xstart;
% 
% for i=1:Ntr-p_feature
%     xnext = net(ann_tr_predict(i:i+p_feature-1)');
%     ann_tr_predict = [ann_tr_predict xnext];
%     xnext = lr_tr_predict(i:i+p_feature-1)*w;
%     lr_tr_predict = [lr_tr_predict xnext];
% end

%% Predict test data start with first 20 days of actual data to predict
% future price with free running mode
Xstart = Xts(1, 1:p_feature);
ann_ts_predict = Xstart;
lr_ts_predict = Xstart;

volXstart = tsvol(1, 1:p_feature);
volann_ts_predict = volXstart;
vollr_ts_predict = volXstart;

predict_ahead = 0;
for i=1:Nts-p_feature + predict_ahead
    %ANN
    xnext = net([ann_ts_predict(i:i+p_feature-1)'; volann_ts_predict(i:i+p_feature-1)']);
    volxnext = volnet([ann_ts_predict(i:i+p_feature-1)'; volann_ts_predict(i:i+p_feature-1)']);
    
    ann_ts_predict = [ann_ts_predict xnext];
    volann_ts_predict = [volann_ts_predict volxnext];
    
    %LR
    xnext = [lr_ts_predict(i:i+p_feature-1) vollr_ts_predict(i:i+p_feature-1)]*w;
    volxnext = [lr_ts_predict(i:i+p_feature-1) vollr_ts_predict(i:i+p_feature-1)]*volw;
    
    lr_ts_predict = [lr_ts_predict xnext];
    vollr_ts_predict = [vollr_ts_predict xnext];
end
% cut out first 20 days 
% ann_tr_predict = ann_tr_predict(p:Ntr);
% lr_tr_predict = lr_tr_predict(p:Ntr);

ann_ts_predict = ann_ts_predict(p:Nts+ predict_ahead);
lr_ts_predict = lr_ts_predict(p:Nts+ predict_ahead);

%% Plotting 
% % Train set - append to compare result
% figure(1)
% hold on, grid on;
% plot(ann_tr_predict, 'DisplayName', 'ANN_Freerun');
% plot(lr_tr_predict, 'DisplayName', 'Linear_Freerun');
% hold off;
% legend('show');

% Testset - append the plot on test set to compare result
figure(1)
hold on, grid on;
plot(ann_ts_predict, 'DisplayName', 'ANN_Freerun');
plot(lr_ts_predict, 'DisplayName', 'Linear_Freerun');
hold off;
legend('show');

%% Plot comparing accuracy
% figure(5), clf
% hold on, grid on;
% plot(ftr,ann_tr_predict,'rx');
% hold off;
% xlabel('Actual', 'FontSize', 12)
% ylabel('ANN_Freerun_Predicted', 'FontSize', 12)
% title('Train set Freerun: Actual vs Predicted', 'FontSize', 14)
% legend('show');
% 
% figure(6), clf
% hold on, grid on;
% plot(fts,ann_ts_predict,'rx');
% hold off;
% xlabel('Actual', 'FontSize', 12)
% ylabel('ANN_Freerun_Predicted', 'FontSize', 12)
% title('Test set Freerun: Actual vs Predicted', 'FontSize', 14)
% legend('show');







