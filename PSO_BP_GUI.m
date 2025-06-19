function PSO_BP_GUI
% PSO_BP_GUI - 粒子群优化BP神经网络的GUI界面

% 创建主窗口
fig = figure('Name', 'PSO-BP神经网络优化系统', 'Position', [100, 100, 900, 600], ...
    'NumberTitle', 'off', 'MenuBar', 'none', 'Resize', 'on');

% 创建面板 - 参数设置区域
param_panel = uipanel('Title', '参数设置', 'Position', [0.02, 0.55, 0.3, 0.43]);

% 创建参数输入控件
uicontrol(param_panel, 'Style', 'text', 'String', '隐含层节点数:', ...
    'Position', [10, 180, 100, 20], 'HorizontalAlignment', 'left');
uicontrol(param_panel, 'Style', 'edit', 'String', '10', ...
    'Position', [120, 180, 100, 20], 'Tag', 'hiddennum');

uicontrol(param_panel, 'Style', 'text', 'String', '训练次数:', ...
    'Position', [10, 150, 100, 20], 'HorizontalAlignment', 'left');
uicontrol(param_panel, 'Style', 'edit', 'String', '1000', ...
    'Position', [120, 150, 100, 20], 'Tag', 'epochs');

uicontrol(param_panel, 'Style', 'text', 'String', '目标误差:', ...
    'Position', [10, 120, 100, 20], 'HorizontalAlignment', 'left');
uicontrol(param_panel, 'Style', 'edit', 'String', '1e-6', ...
    'Position', [120, 120, 100, 20], 'Tag', 'goal');

uicontrol(param_panel, 'Style', 'text', 'String', '学习率:', ...
    'Position', [10, 90, 100, 20], 'HorizontalAlignment', 'left');
uicontrol(param_panel, 'Style', 'edit', 'String', '0.01', ...
    'Position', [120, 90, 100, 20], 'Tag', 'lr');

uicontrol(param_panel, 'Style', 'text', 'String', '种群规模:', ...
    'Position', [10, 60, 100, 20], 'HorizontalAlignment', 'left');
uicontrol(param_panel, 'Style', 'edit', 'String', '15', ...
    'Position', [120, 60, 100, 20], 'Tag', 'sizepop');

uicontrol(param_panel, 'Style', 'text', 'String', '迭代次数:', ...
    'Position', [10, 30, 100, 20], 'HorizontalAlignment', 'left');
uicontrol(param_panel, 'Style', 'edit', 'String', '50', ...
    'Position', [120, 30, 100, 20], 'Tag', 'maxgen');

% 创建面板 - 数据设置区域
data_panel = uipanel('Title', '数据设置', 'Position', [0.02, 0.3, 0.3, 0.23]);

% 创建数据设置控件
uicontrol(data_panel, 'Style', 'text', 'String', '数据文件:', ...
    'Position', [10, 90, 100, 20], 'HorizontalAlignment', 'left');
uicontrol(data_panel, 'Style', 'edit', 'String', 'L2输出.xlsx', ...
    'Position', [120, 90, 100, 20], 'Tag', 'datafile');

uicontrol(data_panel, 'Style', 'text', 'String', '输入特征列:', ...
    'Position', [10, 60, 100, 20], 'HorizontalAlignment', 'left');
uicontrol(data_panel, 'Style', 'edit', 'String', '1:8', ...
    'Position', [120, 60, 100, 20], 'Tag', 'input_cols');

uicontrol(data_panel, 'Style', 'text', 'String', '输出特征列:', ...
    'Position', [10, 30, 100, 20], 'HorizontalAlignment', 'left');
uicontrol(data_panel, 'Style', 'edit', 'String', '9', ...
    'Position', [120, 30, 100, 20], 'Tag', 'output_cols');

% 创建面板 - 训练测试比例设置
train_panel = uipanel('Title', '训练测试设置', 'Position', [0.02, 0.1, 0.3, 0.25]);

% 添加滚动条
scroll_panel = uipanel('Parent', train_panel, 'Position', [0, 0, 1, 1], 'Units', 'normalized');
set(scroll_panel, 'Scrollable', 'on');

% 创建训练测试比例控件
uicontrol(scroll_panel, 'Style', 'text', 'String', '训练起始行:', ...
    'Position', [10, 100, 100, 20], 'HorizontalAlignment', 'left');
uicontrol(scroll_panel, 'Style', 'edit', 'String', '1', ...
    'Position', [120, 100, 100, 20], 'Tag', 'train_start');

uicontrol(scroll_panel, 'Style', 'text', 'String', '训练结束行:', ...
    'Position', [10, 70, 100, 20], 'HorizontalAlignment', 'left');
uicontrol(scroll_panel, 'Style', 'edit', 'String', '1000', ...
    'Position', [120, 70, 100, 20], 'Tag', 'train_end');

uicontrol(scroll_panel, 'Style', 'text', 'String', '预测数量:', ...
    'Position', [10, 40, 100, 20], 'HorizontalAlignment', 'left');
uicontrol(scroll_panel, 'Style', 'edit', 'String', '200', ...
    'Position', [120, 40, 100, 20], 'Tag', 'predict_num');

% 创建按钮
uicontrol(scroll_panel, 'Style', 'pushbutton', 'String', '开始训练', ...
    'Position', [60, 10, 100, 30], 'Callback', @startTraining);

% 创建面板 - 结果显示区域
result_panel = uipanel('Title', '结果输出', 'Position', [0.34, 0.1, 0.64, 0.88]);

% 创建选项卡面板
tabgroup = uitabgroup(result_panel, 'Position', [0.02, 0.02, 0.96, 0.96]);

% 创建选项卡 - 文本输出
tab1 = uitab(tabgroup, 'Title', '文本输出');
output_text = uicontrol(tab1, 'Style', 'edit', 'Max', 2, 'Min', 0, ...
    'HorizontalAlignment', 'left', 'Position', [10, 10, 550, 450], ...
    'Tag', 'output_text', 'Enable', 'inactive');

% 创建选项卡 - 进化曲线
tab2 = uitab(tabgroup, 'Title', '进化曲线');
axes('Parent', tab2, 'Position', [0.1, 0.1, 0.8, 0.7], 'Tag', 'evolution_axes');

% 创建选项卡 - 训练集预测
tab3 = uitab(tabgroup, 'Title', '训练集预测');
axes('Parent', tab3, 'Position', [0.1, 0.55, 0.8, 0.4], 'Tag', 'train_pred_axes');
axes('Parent', tab3, 'Position', [0.1, 0.1, 0.8, 0.35], 'Tag', 'train_error_axes');

% 创建选项卡 - 测试集预测
tab4 = uitab(tabgroup, 'Title', '测试集预测');
axes('Parent', tab4, 'Position', [0.1, 0.55, 0.8, 0.4], 'Tag', 'test_pred_axes');
axes('Parent', tab4, 'Position', [0.1, 0.1, 0.8, 0.35], 'Tag', 'test_error_axes');

% 创建选项卡 - 训练集散点图
tab5 = uitab(tabgroup, 'Title', '训练集散点图');
axes('Parent', tab5, 'Position', [0.1, 0.1, 0.8, 0.8], 'Tag', 'train_scatter_axes');

% 创建选项卡 - 测试集散点图
tab6 = uitab(tabgroup, 'Title', '测试集散点图');
axes('Parent', tab6, 'Position', [0.1, 0.1, 0.8, 0.8], 'Tag', 'test_scatter_axes');

% 创建选项卡 - PSO-BP与BP训练对比
tab7 = uitab(tabgroup, 'Title', 'PSO-BP与BP训练对比');
axes('Parent', tab7, 'Position', [0.1, 0.1, 0.8, 0.8], 'Tag', 'compare_train_axes');

% 创建选项卡 - PSO-BP与BP测试对比
tab8 = uitab(tabgroup, 'Title', 'PSO-BP与BP测试对比');
axes('Parent', tab8, 'Position', [0.1, 0.1, 0.8, 0.8], 'Tag', 'compare_test_axes');

% 在每个结果选项卡添加保存图片按钮（使用normalized单位，确保显示）
uicontrol('Parent', tab2, 'Style', 'pushbutton', 'String', '保存图片', ...
    'Units', 'normalized', 'Position', [0.8, 0.05, 0.15, 0.08], ...
    'Callback', @(~,~) saveCurrentAxes('evolution_axes', '进化曲线'));
uicontrol('Parent', tab3, 'Style', 'pushbutton', 'String', '保存图片', ...
    'Units', 'normalized', 'Position', [0.8, 0.05, 0.15, 0.08], ...
    'Callback', @(~,~) saveCurrentAxes('train_pred_axes', '训练集预测'));
uicontrol('Parent', tab3, 'Style', 'pushbutton', 'String', '保存误差图', ...
    'Units', 'normalized', 'Position', [0.8, 0.15, 0.15, 0.08], ...
    'Callback', @(~,~) saveCurrentAxes('train_error_axes', '训练集误差图'));
uicontrol('Parent', tab4, 'Style', 'pushbutton', 'String', '保存图片', ...
    'Units', 'normalized', 'Position', [0.8, 0.05, 0.15, 0.08], ...
    'Callback', @(~,~) saveCurrentAxes('test_pred_axes', '测试集预测'));
uicontrol('Parent', tab4, 'Style', 'pushbutton', 'String', '保存误差图', ...
    'Units', 'normalized', 'Position', [0.8, 0.15, 0.15, 0.08], ...
    'Callback', @(~,~) saveCurrentAxes('test_error_axes', '测试集误差图'));
uicontrol('Parent', tab5, 'Style', 'pushbutton', 'String', '保存图片', ...
    'Units', 'normalized', 'Position', [0.8, 0.05, 0.15, 0.08], ...
    'Callback', @(~,~) saveCurrentAxes('train_scatter_axes', '训练集散点图'));
uicontrol('Parent', tab6, 'Style', 'pushbutton', 'String', '保存图片', ...
    'Units', 'normalized', 'Position', [0.8, 0.05, 0.15, 0.08], ...
    'Callback', @(~,~) saveCurrentAxes('test_scatter_axes', '测试集散点图'));
uicontrol('Parent', tab7, 'Style', 'pushbutton', 'String', '保存图片', ...
    'Units', 'normalized', 'Position', [0.8, 0.05, 0.15, 0.08], ...
    'Callback', @(~,~) saveCurrentAxes('compare_train_axes', 'PSO-BP与BP训练对比'));
uicontrol('Parent', tab8, 'Style', 'pushbutton', 'String', '保存图片', ...
    'Units', 'normalized', 'Position', [0.8, 0.05, 0.15, 0.08], ...
    'Callback', @(~,~) saveCurrentAxes('compare_test_axes', 'PSO-BP与BP测试对比'));

% 创建状态栏
status_text = uicontrol(fig, 'Style', 'text', 'String', '就绪', ...
    'Position', [10, 5, 880, 20], 'HorizontalAlignment', 'left', ...
    'Tag', 'status_text');

% 开始训练回调函数
function startTraining(~, ~)
    % 获取参数值
    hiddennum = str2double(get(findobj(fig, 'Tag', 'hiddennum'), 'String'));
    epochs = str2double(get(findobj(fig, 'Tag', 'epochs'), 'String'));
    goal = str2double(get(findobj(fig, 'Tag', 'goal'), 'String'));
    lr = str2double(get(findobj(fig, 'Tag', 'lr'), 'String'));
    sizepop = str2double(get(findobj(fig, 'Tag', 'sizepop'), 'String'));
    maxgen = str2double(get(findobj(fig, 'Tag', 'maxgen'), 'String'));
    
    datafile = get(findobj(fig, 'Tag', 'datafile'), 'String');
    input_cols = str2num(get(findobj(fig, 'Tag', 'input_cols'), 'String')); %#ok<ST2NM>
    output_cols = str2num(get(findobj(fig, 'Tag', 'output_cols'), 'String')); %#ok<ST2NM>
    train_start = str2double(get(findobj(fig, 'Tag', 'train_start'), 'String'));
    train_end = str2double(get(findobj(fig, 'Tag', 'train_end'), 'String'));
    predict_num = str2double(get(findobj(fig, 'Tag', 'predict_num'), 'String'));
    train_num = train_end - train_start + 1;
    
    % 更新状态
    set(findobj(fig, 'Tag', 'status_text'), 'String', '正在加载数据...');
    drawnow;
    
    % 清空输出文本
    set(findobj(fig, 'Tag', 'output_text'), 'String', '');
    
    % 记录输出信息的函数
    function appendOutput(text)
        current = get(findobj(fig, 'Tag', 'output_text'), 'String');
        if isempty(current)
            set(findobj(fig, 'Tag', 'output_text'), 'String', text);
        else
            set(findobj(fig, 'Tag', 'output_text'), 'String', [current, '\n', text]);
        end
        drawnow;
    end
    
    try
        % 导入数据
        appendOutput('正在导入数据...');
        Data = table2array(readtable(datafile));
        
        % 划分训练集和测试集
        Train_InPut = Data(1:train_num, input_cols);
        Train_OutPut = Data(1:train_num, output_cols);
        Test_InPut = Data(train_num+1:train_num+predict_num, input_cols);
        Test_OutPut = Data(train_num+1:train_num+predict_num, output_cols);
        
        appendOutput(['数据加载完成，共' num2str(size(Data,1)) '个样本']);
        appendOutput(['训练集：' num2str(size(Train_InPut,1)) '个样本']);
        appendOutput(['测试集：' num2str(size(Test_InPut,1)) '个样本']);
        
        % 数据归一化
        appendOutput('正在进行数据归一化...');
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
        
        % BP网络设置
        appendOutput('正在配置BP神经网络...');
        InPut_num = size(Train_InPut,1);
        OutPut_num = size(Train_OutPut,1);
        
        net = newff(Train_InPut,Train_OutPut,hiddennum);
        
        net.trainParam.epochs = epochs;
        net.trainParam.goal = goal;
        net.trainParam.lr = lr;
        net.trainParam.showWindow = 0;
        
        Optimize_num = InPut_num*hiddennum+hiddennum+hiddennum*OutPut_num+OutPut_num;
        
        % 粒子群算法(PSO)设置
        appendOutput('正在配置PSO算法参数...');
        c1 = 3;
        c2 = 3;
        Vmax = 1.0;
        Vmin = -1.0;
        popmax = 1.0;
        popmin = -1.0;
        
        % PSO优化开始
        appendOutput('开始PSO优化过程...');
        
        % 生成初始化种群
        pop = zeros(sizepop, Optimize_num);
        V = zeros(sizepop, Optimize_num);
        fitness = zeros(1, sizepop);
        
        for i = 1 : sizepop
            pop(i,:) = rands(1,Optimize_num);
            V(i,:) = rands(1,Optimize_num);
            fitness(i) = Optimize_Object(pop(i,:),hiddennum,net,Train_InPut,Train_OutPut);
        end
        
        % 确定初始化的最优值
        [fitnesszbest, bestindex] = min(fitness);
        zbest = pop(bestindex,:);
        gbest = pop;
        fitnessgbest = fitness;
        BestFit = fitnesszbest;
        
        % 创建进度条
        h = waitbar(0, '正在进行迭代寻优计算，请稍等！');
        
        % PSO迭代寻优开始
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
                fitness(j) = Optimize_Object(pop(j,:),hiddennum,net,Train_InPut,Train_OutPut);
                
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
            end
            
            BestFit(i+1) = fitnesszbest;
            appendOutput(['迭代次数：' num2str(i) ' || 最佳适应度值：' num2str(BestFit(i+1))]);
            waitbar(i/maxgen, h);
            
            % 更新进化曲线图
            axes_handle = findobj(fig, 'Tag', 'evolution_axes');
            axes(axes_handle);
            semilogy(0:i, BestFit(1:i+1), 'LineWidth', 2);
            xlabel('迭代次数');
            ylabel('误差的变化');
            legend('误差曲线');
            title('进化过程');
            grid on;
            drawnow;
        end
        
        % 关闭进度条
        close(h);
        
        % 提取最优结果与赋值
        appendOutput('PSO优化完成，正在配置最优网络...');
        w1 = zbest(1 : InPut_num * hiddennum);
        B1 = zbest(InPut_num * hiddennum + 1 : InPut_num * hiddennum + hiddennum);
        w2 = zbest(InPut_num * hiddennum + hiddennum + 1 : InPut_num * hiddennum + hiddennum + hiddennum * OutPut_num);
        B2 = zbest(InPut_num * hiddennum + hiddennum + hiddennum * OutPut_num + 1 : InPut_num * hiddennum + hiddennum + hiddennum * OutPut_num + OutPut_num);
        
        % 对网络进行最优赋值
        net.Iw{1,1} = reshape(w1, hiddennum, InPut_num);
        net.Lw{2,1} = reshape(w2, OutPut_num, hiddennum);
        net.b{1} = reshape(B1, hiddennum, 1);
        net.b{2} = B2';
        
        % 开始正式训练网络
        appendOutput('开始训练优化后的BP神经网络...');
        net.trainParam.showWindow = 0; % 不显示训练窗口
        net = train(net, Train_InPut, Train_OutPut); % 网络训练
        
        % 网络预测输出测试
        appendOutput('正在进行网络预测...');
        T_sim_Train = sim(net, Train_InPut); % 输出训练集预测值
        T_sim_Test = sim(net, Test_InPut); % 输出测试集预测值
        
        % 反归一化
        T_sim_Train = mapminmax('reverse', T_sim_Train, Ps); % 训练输出预测值反归一化
        T_sim_Test = mapminmax('reverse', T_sim_Test, Ps); % 测试输出预测值反归一化
        Train_OutPut_Original = mapminmax('reverse', Train_OutPut, Ps);
        Test_OutPut_Original = mapminmax('reverse', Test_OutPut, Ps);
        
        % 误差值评价值输出
        appendOutput('计算评价指标...');
        % RMSE
        R1 = 1 - norm(Train_OutPut_Original - T_sim_Train)^2 / norm(T_sim_Train - mean(T_sim_Train))^2;
        R2 = 1 - norm(Test_OutPut_Original - T_sim_Test)^2 / norm(Test_OutPut_Original - mean(Test_OutPut_Original))^2;
        appendOutput(['训练集数据的R2为：', num2str(R1)]);
        appendOutput(['测试集数据的R2为：', num2str(R2)]);
        
        % MAE
        mae1 = sum(abs(T_sim_Train - Train_OutPut_Original), 2)' ./ length(Train_OutPut_Original);
        mae2 = sum(abs(T_sim_Test - Test_OutPut_Original), 2)' ./ length(Test_OutPut_Original);
        appendOutput(['训练集数据的MAE为：', num2str(mae1)]);
        appendOutput(['测试集数据的MAE为：', num2str(mae2)]);
        
        % MBE
        mbe1 = sum(T_sim_Train - Train_OutPut_Original, 2)' ./ length(Train_OutPut_Original);
        mbe2 = sum(T_sim_Test - Test_OutPut_Original, 2)' ./ length(Test_OutPut_Original);
        appendOutput(['训练集数据的MBE为：', num2str(mbe1)]);
        appendOutput(['测试集数据的MBE为：', num2str(mbe2)]);
        
        % 可视化结果 - 训练集预测效果
        appendOutput('生成可视化结果...');
        
        % 训练集预测图
        axes_handle = findobj(fig, 'Tag', 'train_pred_axes');
        axes(axes_handle);
        plot(Train_OutPut_Original, 'r-*');
        hold on;
        plot(T_sim_Train, 'b:o');
        grid on;
        hold off;
        legend('真实值', '预测值');
        xlabel('样本编号');
        ylabel('数值');
        string = {'训练集预测结果对比(PSO-BP)'; ['R2=' num2str(R1)]};
        title(string);
        
        % 训练集误差图
        axes_handle = findobj(fig, 'Tag', 'train_error_axes');
        axes(axes_handle);
        stem(Train_OutPut_Original - T_sim_Train, 'k--o');
        legend('误差');
        xlabel('样本编号');
        ylabel('数值');
        title('训练集预测结果误差情况');
        
        % 测试集预测图
        axes_handle = findobj(fig, 'Tag', 'test_pred_axes');
        axes(axes_handle);
        test_len = min([predict_num, length(Test_OutPut_Original), length(T_sim_Test)]);
        test_samples = (train_end+1):(train_end+test_len);
        plot(test_samples, Test_OutPut_Original(1:test_len), 'r-*');
        hold on;
        plot(test_samples, T_sim_Test(1:test_len), 'b:o');
        grid on;
        hold off;
        legend('真实值', '预测值');
        xlabel('样本编号');
        ylabel('数值');
        string = {'测试集预测结果对比(PSO-BP)'; ['R2=' num2str(R2)]};
        title(string);
        
        % 测试集误差图
        axes_handle = findobj(fig, 'Tag', 'test_error_axes');
        axes(axes_handle);
        stem(Test_OutPut_Original(1:test_len) - T_sim_Test(1:test_len), 'k--o');
        legend('误差');
        xlabel('样本编号');
        ylabel('数值');
        title('测试集预测结果误差情况');
        
        % 训练集散点图
        axes_handle = findobj(fig, 'Tag', 'train_scatter_axes');
        axes(axes_handle);
        sz = 25;
        c = 'b';
        scatter(Train_OutPut_Original, T_sim_Train, sz, c);
        hold on;
        plot(xlim, ylim, '--k');
        xlabel('训练集真实值');
        ylabel('训练集预测值');
        xlim([min(Train_OutPut_Original) max(Train_OutPut_Original)]);
        ylim([min(T_sim_Train) max(T_sim_Train)]);
        hold off;
        grid on;
        title('训练集预测值 vs. 训练集真实值');
        
        % 测试集散点图
        axes_handle = findobj(fig, 'Tag', 'test_scatter_axes');
        axes(axes_handle);
        c = 'r';
        scatter(Test_OutPut_Original(1:test_len), T_sim_Test(1:test_len), sz, c);
        hold on;
        plot(xlim, ylim, '--k');
        xlabel('测试集真实值');
        ylabel('测试集预测值');
        xlim([min(Test_OutPut_Original(1:test_len)) max(Test_OutPut_Original(1:test_len))]);
        ylim([min(T_sim_Test(1:test_len)) max(T_sim_Test(1:test_len))]);
        hold off;
        grid on;
        title('测试集预测值 vs. 测试集真实值');
        
        % 构建并训练原始BP网络进行对比
        appendOutput('正在构建未优化的BP网络进行对比...');
        Train_OutPut_Norm = mapminmax('apply', Train_OutPut_Original, Ps);
        Test_OutPut_Norm = mapminmax('apply', Test_OutPut_Original, Ps);
        net_original = newff(Train_InPut, Train_OutPut_Norm, hiddennum);
        net_original.trainParam.showWindow = 0;
        net_original = train(net_original, Train_InPut, Train_OutPut_Norm);
        T_sim_Train_1 = sim(net_original, Train_InPut);
        T_sim_Test_2 = sim(net_original, Test_InPut);
        T_sim_Train_1 = mapminmax('reverse', T_sim_Train_1, Ps);
        T_sim_Test_2 = mapminmax('reverse', T_sim_Test_2, Ps);
        
        % PSO-BP与BP对比图 - 训练集
        axes_handle = findobj(fig, 'Tag', 'compare_train_axes');
        axes(axes_handle);
        % 创建样本编号数组作为x轴
        train_samples = train_start:train_end;
        plot(train_samples, Train_OutPut_Original, 'r-');
        hold on;
        plot(train_samples, T_sim_Train, 'b-');
        plot(train_samples, T_sim_Train_1, 'k-');
        grid on;
        hold off;
        legend('真实值', 'PSO-BP预测值', '未优化BP预测值');
        xlabel('训练集样本编号');
        ylabel('数值');
        title('训练集预测结果对比(PSO-BP Vs BP)');
        xlim([train_start train_end]);
        
        % PSO-BP与BP对比图 - 测试集
        axes_handle = findobj(fig, 'Tag', 'compare_test_axes');
        axes(axes_handle);
        test_len2 = min([predict_num, length(Test_OutPut_Original), length(T_sim_Test), length(T_sim_Test_2)]);
        test_samples2 = (train_end+1):(train_end+test_len2);
        plot(test_samples2, Test_OutPut_Original(1:test_len2), 'r-');
        hold on;
        plot(test_samples2, T_sim_Test(1:test_len2), 'b-');
        plot(test_samples2, T_sim_Test_2(1:test_len2), 'k-');
        grid on;
        hold off;
        legend('真实值', 'PSO-BP预测值', '未优化BP预测值');
        xlabel('测试集样本编号');
        ylabel('数值');
        title('测试集预测结果对比(PSO-BP Vs BP)');
        xlim([train_end+1 train_end+test_len2]);
        
        % 计算并输出R2、MAE、MBE、MSE到Excel
        n = test_len2;
        y_true = Test_OutPut_Original(1:n);
        y_pso = T_sim_Test(1:n);
        y_bp = T_sim_Test_2(1:n);
        % PSO-BP指标
        R2_pso = 1 - sum((y_true - y_pso).^2) / sum((y_true - mean(y_true)).^2);
        MAE_pso = mean(abs(y_true - y_pso));
        MBE_pso = mean(y_pso - y_true);
        MSE_pso = mean((y_true - y_pso).^2);
        % BP指标
        R2_bp = 1 - sum((y_true - y_bp).^2) / sum((y_true - mean(y_true)).^2);
        MAE_bp = mean(abs(y_true - y_bp));
        MBE_bp = mean(y_bp - y_true);
        MSE_bp = mean((y_true - y_bp).^2);
        % 写入Excel
        header = {'模型','R2','MAE','MBE','MSE'};
        data = {'PSO-BP', R2_pso, MAE_pso, MBE_pso, MSE_pso; 'BP', R2_bp, MAE_bp, MBE_bp, MSE_bp};
        xlswrite('zb.xlsx', header, 1, 'A1');
        xlswrite('zb.xlsx', data, 1, 'A2');
        
        % 更新状态
        set(findobj(fig, 'Tag', 'status_text'), 'String', '训练完成');
        appendOutput('所有操作已完成！');
        
    catch e
        % 错误处理
        set(findobj(fig, 'Tag', 'status_text'), 'String', ['错误: ' e.message]);
        appendOutput(['发生错误: ' e.message]);
        if exist('h', 'var') && ishandle(h)
            close(h);
        end
    end
end

% 添加保存图片的函数，确保保存内容完整
function saveCurrentAxes(axes_tag, title_str)
    % 获取当前tab对象
    tabgroup_handle = findobj(fig, 'Type', 'uitabgroup');
    selected_tab = tabgroup_handle.SelectedTab;
    % 获取当前tab下所有axes
    axes_list = findobj(selected_tab, 'Type', 'axes');
    % 优先找tag匹配的axes
    axes_handle = [];
    for k = 1:length(axes_list)
        if strcmp(get(axes_list(k), 'Tag'), axes_tag)
            axes_handle = axes_list(k);
            break;
        end
    end
    % 如果没找到tag匹配的，直接用第一个axes
    if isempty(axes_handle) && ~isempty(axes_list)
        axes_handle = axes_list(1);
    end
    if isempty(axes_handle)
        msgbox('未找到要保存的图像！', '错误', 'error');
        return;
    end
    % 创建新的figure并复制axes内容
    temp_fig = figure('Visible', 'off');
    temp_axes = copyobj(axes_handle, temp_fig);
    set(temp_axes, 'Units', 'normalized', 'Position', [0.13, 0.11, 0.775, 0.815]);
    
    % 获取图例对象
        legend_obj = findobj(axes_handle, 'Type', 'Legend');
    if ~isempty(legend_obj)
        % 复制图例到新图形
        legend_entries = get(legend_obj, 'String');
        legend_handles = get(legend_obj, 'Children');
        legend(temp_axes, legend_handles, legend_entries, 'Location', 'best');
    end
    
    title(temp_axes, title_str);
    [filename, pathname] = uiputfile({'*.png', 'PNG图片(*.png)'; '*.jpg', 'JPG图片(*.jpg)'; '*.fig', 'MATLAB图片(*.fig)'}, ['保存' title_str '图片']);
    if isequal(filename,0) || isequal(pathname,0)
        close(temp_fig);
        return;
    end
    saveas(temp_fig, fullfile(pathname, filename));
    close(temp_fig);
    msgbox(['图片已成功保存到: ' fullfile(pathname, filename)], '保存成功', 'help');
end

end