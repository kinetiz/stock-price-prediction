%%Setup
N = 100;

m1 = [0; 3];
C1 = [2 1;1 2];

m2 = [2; 1];
C2 = [1 0;0 1];

numGrid = 50;
xRange = linspace(-6.0,6.0,numGrid);
yRange = linspace(-6.0,6.0,numGrid);

P1 = zeros(numGrid, numGrid);
P2 = P1;
Post1 = P1;
Postxy = [];
xx = [];
yy = [];

for i=1:numGrid
    for j=1:numGrid
        x = [yRange(j) xRange(i)]';
        P1(i,j) = mvnpdf(x', m1', C1);
        P2(i,j) = mvnpdf(x', m2', C2);
        Post1(i,j) = 1 / (1 + exp( (-1/2) *( ((x-m2)' * (C2^-1) * (x-m2)) - ((x-m1)' * (C1^-1) * (x-m1) ))));
        Postxy = [Postxy Post1(i,j)];
        xx = [xx xRange(i)];
        yy = [yy yRange(j)];
    end
end
Pmax = max(max([P1 P2]));
figure(1), clf,
contour(xRange, yRange, P1, [0.1*Pmax 0.5*Pmax  0.8*Pmax ], 'LineWidth', 2,'DisplayName','d1');
hold on
 plot(m1(1), m1(2), '*', 'LineWidth', 4,'DisplayName','d2');
contour(xRange, yRange, P2, [0.1*Pmax  0.5*Pmax  0.8*Pmax], 'LineWidth', 2,'DisplayName','d3');
 plot(m2(1), m2(2), '*', 'LineWidth', 4,'DisplayName','d4');
contour(xRange, yRange, Post1, [0.5 0.5], 'r','LineWidth', 2,'DisplayName','Baye''s boundary');
grid on;
% colorbar

xlabel('X1', 'FontSize', 12)
ylabel('X2', 'FontSize', 12)

% 3d post
figure(2)
scatter3(xx(:),yy(:),Post1(:),'xb');

xlabel('Feature1', 'FontSize', 12)
ylabel('Feature2', 'FontSize', 12)
zlabel('Posterior Probability', 'FontSize', 12)

%% Plot sampled data on contour
X1 = mvnrnd(m1, C1, N);
X2 = mvnrnd(m2, C2, N);
figure(1)
hold on;
sample_size = 100;
sx1 = datasample(X1,sample_size);
sx2 = datasample(X2,sample_size);
plot(sx1(:,1),sx1(:,2),'bo','DisplayName','Class1 Sample');
plot(sx2(:,1),sx2(:,2),'ro','DisplayName','Class2 Sample');
grid on;
 
axis([-6 6 -6 6]);
title('Posterior Baye''s vs ANN boundary', 'FontSize', 14);
xlabel('Feature1', 'FontSize', 12)
ylabel('Feature2', 'FontSize', 12)

%% 1) neuron network
X = [X1;X2];
W=[ones(length(X1),1);zeros(length(X2),1)];

[netsmall] = feedforwardnet(10);
[netsmall] = train(netsmall, X', W');

[netbig] = feedforwardnet(30);
[netbig] = train(netbig, X', W');

% [net] = feedforwardnet(20);
% [net] = train(net, X', W');
% 
Pnetbig =zeros(numGrid, numGrid);
% Pnet =zeros(numGrid, numGrid);
Pnetsmall =zeros(numGrid, numGrid);
for i=1:numGrid
    for j=1:numGrid
        x = [yRange(j) xRange(i)]';
        [Pnetsmall(i,j)] = netsmall(x);
        [Pnetbig(i,j)] = netbig(x);
%         [Pnet(i,j)] = net(x);
    end
end
figure(1),
hold on,
contour(xRange, yRange, Pnetbig, [0.5 0.5],'m', 'LineWidth', 2,'DisplayName','Net30');
% contour(xRange, yRange, Pnet, [0.5 0.5],'g', 'LineWidth', 2,'DisplayName','Net20');
contour(xRange, yRange, Pnetsmall, [0.5 0.5],'c', 'LineWidth', 2,'DisplayName','Net10');

legend('show')
hold off



% % 
% % %% Generate unseen data
% % X1 = mvnrnd(m1, C1, 1000);
% % X2 = mvnrnd(m2, C2, 1000);
% % figure(1)
% % hold on;
% % sample_size = 100;
% % sx1 = datasample(X1,sample_size);
% % sx2 = datasample(X2,sample_size);
% % plot(sx1(:,1),sx1(:,2),'bx','DisplayName','Class1 Unseen data');
% % plot(sx2(:,1),sx2(:,2),'rx','DisplayName','Class2 Unseen data');
% % grid on;
% %  

