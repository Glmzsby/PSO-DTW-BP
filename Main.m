clc;
clear;
warning off;

%% 直接启动PSO-BP神经网络GUI界面
disp('正在启动PSO-BP神经网络GUI界面...');
PSO_BP_GUI;

%% 以下代码已移至GUI界面中，点击"开始训练"按钮后执行
% 以下为原始训练代码，现已注释
%{

% 选取前376个样本作为训练集，后96个样本作为测试集，即（1：376），和（377：end）
Train_InPut = Data(1:1500,InPut_num); % 训练输入
Train_OutPut = Data(1:1500,OutPut_num); % 训练输出
Test_InPut = Data(1501:end,InPut_num); % 测试输入
Test_OutPut = Data(1501:end,OutPut_num); % 测试输出

%% 数据归一化
% 将数据归一化到0-1之间
Temp = [Train_OutPut;Test_OutPut];
[~, Ps] = mapminmax(Temp',0,1); 
% 归一化训练输入值
Sc = size(Train_InPut);
Temp = reshape(Train_InPut,[1,Sc(1)*Sc(2)]);
Temp = mapminmax('apply',Temp,Ps);
Train_InPut = reshape(Temp,[Sc(1),Sc(2)])';
% 归一化测试输入值
Sc = size(Test_InPut);
Temp = reshape(Test_InPut,[1,Sc(1)*Sc(2)]);
Temp = mapminmax('apply',Temp,Ps);
Test_InPut = reshape(Temp,[Sc(1),Sc(2)])';
% 归一化训练输出值
Train_OutPut = mapminmax('apply',Train_OutPut',Ps);
% 归一化测试输出值
Test_OutPut = mapminmax('apply',Test_OutPut',Ps);

%% BP网络设置
Hiddennum = 10; % BP神经网络的隐含层节点数设置，建议至少大于等于输入的个数+1，此处8个输入，隐含层节点数设置为10
InPut_num = size(Train_InPut,1); % 网络输入个数
OutPut_num = size(Train_OutPut,1); % 网络输出个数

net = newff(Train_InPut,Train_OutPut,Hiddennum); % 构建网络

net.trainParam.epochs = 1000; % 训练次数
net.trainParam.goal = 1e-6; % 目标误差
net.trainParam.lr = 0.01; % 学习率
net.trainParam.showWindow = 0; % 关闭窗口

Optimize_num = InPut_num*Hiddennum+Hiddennum+Hiddennum*OutPut_num+OutPut_num; % 确定总的需要优化的参数个数

%% 粒子群算法(PSO)设置，
c1 = 3; % 学习因子
c2 = 3; % 学习因子
maxgen = 50; % 种群更新次数  
sizepop = 5; % 种群规模
Vmax = 1.0; % 最大速度
Vmin = -1.0; % 最小速度
popmax = 1.0; % 最大边界
popmin = -1.0; % 最小边界

%% PSO优化开始

% 生成初始化种群
for i = 1 : sizepop
    pop(i,:) = rands(1,Optimize_num); % 初始化种群
    V(i,:) = rands(1,Optimize_num); % 初始化速度
    fitness(i) = Optimize_Object(pop(i,:),Hiddennum,net,Train_InPut,Train_OutPut);
end
% 确定初始化的最优值
[fitnesszbest, bestindex] = min(fitness);
zbest = pop(bestindex,:); % 全局最佳
gbest = pop; % 个体最佳
fitnessgbest = fitness; % 个体最佳适应度值
BestFit = fitnesszbest; % 全局最佳适应度值

% PSO迭代寻优开始
h=waitbar(0,'正在进行迭代寻优计算，请稍等！（请进度完成后再关闭本窗口）');
for i = 1 : maxgen
    for j = 1 : sizepop
        % 速度更新
        V(j,:) = V(j,:) + c1 * rand * (gbest(j,:) - pop(j,:)) + c2 * rand * (zbest - pop(j,:));
        V(j,(V(j,:) > Vmax)) = Vmax;
        V(j,(V(j,:) < Vmin)) = Vmin;
        % 种群更新
        pop(j,:) = pop(j,:) + 0.2 * V(j,:);
        pop(j,(pop(j,:) > popmax)) = popmax;
        pop(j,(pop(j,:) < popmin)) = popmin;
        % 自适应变异
        pos = unidrnd(Optimize_num);
        if rand > 0.85
            pop(j, pos) = rands(1, 1);
        end
        % 适应度值
        fitness(j) = Optimize_Object(pop(j,:),Hiddennum,net,Train_InPut,Train_OutPut);
    end
    % 个体最优更新
    if fitness(j) < fitnessgbest(j)
        gbest(j,:) = pop(j,:);
        fitnessgbest(j) = fitness(j);
    end
    % 群体最优更新
    if fitness(j) < fitnesszbest
        zbest = pop(j,:);
        fitnesszbest = fitness(j);
    end
    BestFit = [BestFit, fitnesszbest];
    disp(['Iteration：' num2str(i) ' || Best_Object：' num2str(BestFit(i))]);
    waitbar(i/maxgen,h);
end

%% 提取最优结果与赋值
w1 = zbest(1 : InPut_num * Hiddennum);
B1 = zbest(InPut_num * Hiddennum + 1 : InPut_num * Hiddennum + Hiddennum);
w2 = zbest(InPut_num * Hiddennum + Hiddennum + 1 : InPut_num * Hiddennum ...
    + Hiddennum + Hiddennum * OutPut_num);
B2 = zbest(InPut_num * Hiddennum + Hiddennum + Hiddennum * OutPut_num + 1 : ...
    InPut_num * Hiddennum + Hiddennum + Hiddennum * OutPut_num + OutPut_num);
% 对网络进行最优赋值
net.Iw{1,1} = reshape(w1,Hiddennum,InPut_num);
net.Lw{2,1} = reshape(w2,OutPut_num,Hiddennum);
net.b{1} = reshape(B1,Hiddennum,1);
net.b{2} = B2';
%% 开始正式训练网络
net.trainParam.showWindow = 1; % 打开窗口
net = train(net,Train_InPut,Train_OutPut); % 网络训练

%% 网络预测输出测试
T_sim_Train = sim(net,Train_InPut); % 输出训练集预测值
T_sim_Test = sim(net,Test_InPut); % 输出测试集预测值
T_sim_Train = mapminmax('reverse',T_sim_Train,Ps); % 训练输出预测值反归一化
T_sim_Test = mapminmax('reverse',T_sim_Test,Ps); % 测试输出预测值反归一化

Train_OutPut = mapminmax('reverse',Train_OutPut,Ps);
Test_OutPut = mapminmax('reverse',Test_OutPut,Ps);

%% 误差值评价值输出
% RMSE
R1 = 1 - norm(Train_OutPut - T_sim_Train)^2 / norm(T_sim_Train - mean(T_sim_Train))^2;
R2 = 1 - norm(Test_OutPut  - T_sim_Test)^2 / norm(Test_OutPut  - mean(Test_OutPut ))^2;
disp(['训练集数据的R2为：', num2str(R1)]);
disp(['测试集数据的R2为：', num2str(R2)]);

% MAE
mae1 = sum(abs(T_sim_Train - Train_OutPut), 2)' ./ length(Train_OutPut);
mae2 = sum(abs(T_sim_Test - Test_OutPut ), 2)' ./ length(Test_OutPut);
disp(['训练集数据的MAE为：', num2str(mae1)]);
disp(['测试集数据的MAE为：', num2str(mae2)]);

% MBE
mbe1 = sum(T_sim_Train - Train_OutPut, 2)' ./ length(Train_OutPut);
mbe2 = sum(T_sim_Test - Test_OutPut , 2)' ./ length(Test_OutPut);
disp(['训练集数据的MBE为：', num2str(mbe1)]);
disp(['测试集数据的MBE为：', num2str(mbe2)]);

%% 可视化结果

% 图1：PSO迭代进化曲线
figure(1);
semilogy(BestFit,'LineWidth',2);
xlabel('迭代次数');
ylabel('误差的变化');
legend('误差曲线');
title('进化过程');
grid on;

% 图2：训练集预测效果
figure(2)
subplot(2,1,1);
plot(Train_OutPut,'r-*');
hold on
plot(T_sim_Train,'b:o');
grid on
hold off
legend('真实值','预测值');
xlabel('样本编号');
ylabel('数值');
string = {'训练集预测结果对比(PSO-BP)'; ['RMSE=' num2str(R1)]};
title(string);

subplot(2,1,2);
stem(Train_OutPut - T_sim_Train,'k--o');
legend('误差');
xlabel('样本编号');
ylabel('数值');
title('训练集预测结果误差情况');

% 图3：测试集预测效果
figure(3)
subplot(2,1,1);
plot(Test_OutPut,'r-*');
hold on
plot(T_sim_Test,'b:o');
grid on
hold off
legend('真实值','预测值');
xlabel('样本编号');
ylabel('数值');
string = {'测试集预测结果对比(PSO-BP)'; ['RMSE=' num2str(R2)]};
title(string);

subplot(2,1,2);
stem(Test_OutPut - T_sim_Test,'k--o');
legend('误差');
xlabel('样本编号');
ylabel('数值');
title('测试集预测结果误差情况');

% 图4 训练集散点图
sz = 25;
c = 'b';

figure(4)
scatter(Train_OutPut, T_sim_Train, sz, c);
hold on
plot(xlim, ylim, '--k');
xlabel('训练集真实值');
ylabel('训练集预测值');
xlim([min(Train_OutPut) max(Train_OutPut)]);
ylim([min(T_sim_Train) max(T_sim_Train)]);
hold off
grid on 
title('训练集预测值 vs. 训练集真实值');

% 图5 训测试集散点图

c = 'r';
figure(5);
scatter(Test_OutPut, T_sim_Test, sz, c);
hold on
plot(xlim, ylim, '--k');
xlabel('测试集真实值');
ylabel('测试集预测值');
xlim([min(Test_OutPut) max(Test_OutPut)]);
ylim([min(T_sim_Test) max(T_sim_Test)]);
hold off
grid on 
title('测试集预测值 vs. 测试集真实值');
%% 与原始未优化过的网络进行对比

% 构建并训练原始BP
Train_OutPut = mapminmax('apply',Train_OutPut,Ps); % 训练输出归一化
Test_OutPut = mapminmax('apply',Test_OutPut,Ps); % 训练输入归一化
net_original = newff(Train_InPut,Train_OutPut,Hiddennum); % 构建网络
net_original.trainParam.showWindow = 0; % 关闭窗口
net_original = train(net_original,Train_InPut,Train_OutPut); % 网络训练
T_sim_Train_1 = sim(net_original,Train_InPut); % 输出训练集预测值
T_sim_Test_2 = sim(net_original,Test_InPut); % 输出测试集预测值
T_sim_Train_1 = mapminmax('reverse',T_sim_Train_1,Ps); % 训练输出预测值反归一化
T_sim_Test_2 = mapminmax('reverse',T_sim_Test_2,Ps); % 测试输出预测值反归一化
Train_OutPut = mapminmax('reverse',Train_OutPut,Ps); % 反归一化
Test_OutPut = mapminmax('reverse',Test_OutPut,Ps); % 反归一化

% 图6 PSO-BP与BP对比
figure(6)
subplot(2,1,1);
plot(Train_OutPut,'r-*');
hold on
plot(T_sim_Train,'b:o');
plot(T_sim_Train_1,'k--*');
grid on
hold off
legend('真实值','PSO-BP预测值','未优化BP预测值');
xlabel('训练集样本编号');
ylabel('数值');
title('训练集预测结果对比(PSO-BP Vs BP)');


subplot(2,1,2);
plot(Test_OutPut,'r-*');
hold on
plot(T_sim_Test,'b:o');
plot(T_sim_Test_2,'k--*');
grid on
hold off
legend('真实值','PSO-BP预测值','未优化BP预测值');
xlabel('测试集样本编号');
ylabel('数值');
title('测试集预测结果对比(PSO-BP Vs BP)');

%}

%% GUI界面已启动
disp('请在GUI界面中点击"开始训练"按钮开始训练过程。');
