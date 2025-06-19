function calculate_metrics()
% 计算MSE和MAE评价指标并保存到Excel文件

% 读取数据
Data = table2array(readtable("数据集.xlsx"));
train_num = 1500;

% 划分训练集和测试集
InPut_num = 1:1:8;
OutPut_num = 9;
Train_InPut = Data(1:train_num,InPut_num);
Train_OutPut = Data(1:train_num,OutPut_num);
Test_InPut = Data(train_num+1:end,InPut_num);
Test_OutPut = Data(train_num+1:end,OutPut_num);

% 数据归一化
Temp = [Train_OutPut;Test_OutPut];
[~, Ps] = mapminmax(Temp',0,1);
Sc = size(Train_InPut);
Temp = reshape(Train_InPut,[1,Sc(1)*Sc(2)]);
Temp = mapminmax('apply',Temp,Ps);
Train_InPut = reshape(Temp,[Sc(1),Sc(2)])';
Sc = size(Test_InPut);
Temp = reshape(Test_InPut,[1,Sc(1)*Sc(2)]);
Temp = mapminmax('apply',Temp,Ps);
Test_InPut = reshape(Temp,[Sc(1),Sc(2)])';
Train_OutPut = mapminmax('apply',Train_OutPut',Ps);
Test_OutPut = mapminmax('apply',Test_OutPut',Ps);

% 训练原始BP网络
Hiddennum = 10;
net_bp = newff(Train_InPut,Train_OutPut,Hiddennum);
net_bp.trainParam.showWindow = 0;
net_bp = train(net_bp,Train_InPut,Train_OutPut);

% BP网络预测
bp_train_pred = sim(net_bp,Train_InPut);
bp_test_pred = sim(net_bp,Test_InPut);

% 反归一化
bp_train_pred = mapminmax('reverse',bp_train_pred,Ps);
bp_test_pred = mapminmax('reverse',bp_test_pred,Ps);
Train_OutPut = mapminmax('reverse',Train_OutPut,Ps);
Test_OutPut = mapminmax('reverse',Test_OutPut,Ps);

% 训练PSO优化的BP网络
net_pso = PSO_BP_GUI();

% PSO-BP网络预测
pso_train_pred = sim(net_pso,Train_InPut);
pso_test_pred = sim(net_pso,Test_InPut);

% 反归一化
pso_train_pred = mapminmax('reverse',pso_train_pred,Ps);
pso_test_pred = mapminmax('reverse',pso_test_pred,Ps);

% 计算评价指标
% BP网络
bp_train_mse = mean((Train_OutPut - bp_train_pred).^2);
bp_train_mae = mean(abs(Train_OutPut - bp_train_pred));
bp_test_mse = mean((Test_OutPut - bp_test_pred).^2);
bp_test_mae = mean(abs(Test_OutPut - bp_test_pred));

% PSO-BP网络
pso_train_mse = mean((Train_OutPut - pso_train_pred).^2);
pso_train_mae = mean(abs(Train_OutPut - pso_train_pred));
pso_test_mse = mean((Test_OutPut - pso_test_pred).^2);
pso_test_mae = mean(abs(Test_OutPut - pso_test_pred));

% 读取现有的Excel文件
try
    existing_data = readtable('评价指标.xlsx');
    last_row = height(existing_data);
    if last_row >= 2  % 跳过标题行
        start_row = last_row + 1;
    else
        start_row = 3;  % 从第3行开始写入
    end
catch
    start_row = 3;  % 如果文件不存在或为空，从第3行开始写入
end

% 准备新数据
new_data = {
    bp_train_mse, bp_train_mae, pso_train_mse, pso_train_mae;
    bp_test_mse, bp_test_mae, pso_test_mse, pso_test_mae
};

% 写入Excel文件
filename = '评价指标.xlsx';
writematrix(new_data(1,:), filename, 'Sheet', 1, 'Range', 'B3:E3');
writematrix(new_data(2,:), filename, 'Sheet', 1, 'Range', 'B4:E4');


fprintf('评价指标已保存到Excel文件中。\n');
end