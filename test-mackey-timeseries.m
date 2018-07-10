% mackeyglass();
tr = X(1:1500);
Ntr = 1500;
p = 10;
p_feature = p-1;
row_num_tr = Ntr-p+1;
%Xtr = zeros(row_num_tr,p_feature);
% ftr = zeros(row_num_tr,1);

ts = X(1501:sample_n);
Nts = 500;
row_num_ts = Nts-p+1;
%Xts = zeros(row_num_ts,p_feature);
% fts = zeros(row_num_ts,1);

%% Prepare Train and Test data
% Prepare Train data
i = 1;
Xtemp = zeros(row_num_tr,p);
while i <= row_num_tr
    Xtemp(i,:) = tr(i:(p-1)+i);
    i = i+1 ;
end
% feature = first 19 columns
Xtr = Xtemp(:,1:p_feature);
% target = last column
ftr = Xtemp(:,p);

% Prepare Test data
i = 1;
Xtemp = zeros(row_num_ts,p);
while i <= row_num_ts
    Xtemp(i,:) = ts(i:(p-1)+i);
    i = i+1 ;
end
Xts = Xtemp(:,1:p_feature);
fts = Xtemp(:,p);

%% Linear Regression to predict 1 data ahead

% Training part
gama = 0.5;
cvx_begin quiet
    variable w( p_feature );
    % ||Ya - f||^2 + gama||a||^2
    minimize( norm(Xtr*w-ftr) + gama*norm(w,1) );
cvx_end

% % Predict with train data
% fhtr = Xtr*w;
% figure(1), clf
% plot(ftr, 'bx', 'LineWidth', 1,'DisplayName','Actual'),
% hold on,
% plot(fhtr, 'rx', 'LineWidth', 1,'DisplayName','Predicted')
% hold off
% xlabel('Time series', 'FontSize', 12)
% ylabel('Values', 'FontSize', 12)
% title('Regression: Train set - Actual vs Predicted values over time series', 'FontSize', 14)

% % Predict with test data
% fhts = Xts*w;
% figure(2), clf
% plot(ts,'LineWidth', 1,'DisplayName','Actual'),
% hold on,
% plot([ts(1:p_feature);fhts],'LineWidth', 1,'DisplayName','Regression Predicted')
% hold off
% xlabel('Time series', 'FontSize', 12)
% ylabel('Values', 'FontSize', 12)
% title('Test set: Actual vs Regression and ANN predicted over time', 'FontSize', 14)
% 
% figure(3), clf
% plot(fts, fhts, 'x')
% xlabel('Actual', 'FontSize', 12)
% ylabel('Predicted', 'FontSize', 12)
% title('Test set: Actual vs Regression Predicted values', 'FontSize', 14)

%% ANN training for prediction
[net] = feedforwardnet(20);
[net] = train(net, Xtr', ftr');

% ann_fhtr = net(Xtr');
ann_fhts = net(Xts');

% % Predict with train data
% figure(4), clf
% plot(ftr, 'bx', 'LineWidth', 1,'DisplayName','Actual'),
% hold on,
% plot(ann_fhtr, 'rx', 'LineWidth', 1,'DisplayName','Predicted')
% hold off
% xlabel('Time series', 'FontSize', 12)
% ylabel('Values', 'FontSize', 12)
% title('ANN: Train set- Actual vs Predicted values over time', 'FontSize', 14)

% % Predict with test data
% figure(2), hold on;
% % plot(fts, 'bx', 'LineWidth', 1,'DisplayName','Actual'),
% % hold on,
% plot([ts(1:p_feature)' ann_fhts], 'LineWidth', 1,'DisplayName','ANN Predicted')
% hold off
% % xlabel('Time series', 'FontSize', 12)
% % ylabel('Values', 'FontSize', 12)
% legend('show');

% figure(6), clf
% plot(fts, ann_fhts, 'x')
% xlabel('Actual', 'FontSize', 12)
% ylabel('Predicted', 'FontSize', 12)
% legend('show');
% title('Test set: Actual vs ANN Predicted values', 'FontSize', 14)

%% Free running mode
Xstart = Xts(1, 1:p_feature);
Yhann = Xstart;
Yhlr = Xstart;

for i=1:Nts-p_feature
    xnext = net(Yhann(i:i+p_feature-1)');
    Yhann = [Yhann xnext];
    xnext = Yhlr(i:i+p_feature-1)*w;
    Yhlr = [Yhlr xnext];
end

figure(2)
hold on
plot(Yhann, 'LineWidth', 1,'DisplayName','ANN Free-running 60') ;
plot(Yhlr, 'LineWidth', 1,'DisplayName','Regression Free-running 60') ;
hold off
xlabel('Time', 'FontSize', 12)
ylabel('Values', 'FontSize', 12)
legend('show');
title('Free running Mode ANN vs Linear Regression', 'FontSize', 14)



% figure(7), clf
% plot(fts, Yhlr, 'x')
% xlabel('Actual', 'FontSize', 12)
% ylabel('Free-running Regression', 'FontSize', 12)
% legend('show');
% title('Test set: Actual vs Free-running Regression Prediction', 'FontSize', 14)
% axis ([0.4 1.3 0.4 1.3]);
% 
% figure(8), clf
% plot(fts, Yhann, 'x')
% xlabel('Actual', 'FontSize', 12)
% ylabel('Free-running ANN', 'FontSize', 12)
% legend('show');
% title('Test set: Actual vs Free-running ANN Prediction', 'FontSize', 14)
% axis ([-1.5 3.5 -1.5 3.5]);









