% 从 Excel 文件读取数据（第一列）
data = xlsread('数据集.xlsx', 'A:A');  % 读取第一列数据
data = data(~isnan(data));  % 去除可能的 NaN 值（空单元格）

% L2 正则化降噪（直接对原始数据）
lambda = 0.5;  % 正则化参数（可调整）
denoised_data = fminunc(@(x) norm(x - data)^2 + lambda * norm(diff(x))^2, data);

% 可视化结果
figure;
plot(data, 'b-', 'LineWidth', 1.5, 'DisplayName', '原始数据');
hold on;
plot(denoised_data, 'g-', 'LineWidth', 2, 'DisplayName', '降噪后数据');
hold off;
legend('show');
xlabel('数据点索引');
ylabel('数值');
title('L2 正则化降噪效果对比');
grid on;

% 读取现有Excel文件的表头
[~, headers] = xlsread('L2输出.xlsx', 'A1:I1');

% 准备输出数据
output_data = cell(length(denoised_data), 9);

% 填充第一列（完整数据）
for i = 1:length(denoised_data)
    output_data{i,1} = denoised_data(i);
end

% 填充其余列（依次删除前面的数据）
for col = 2:9
    start_idx = col - 1;
    for row = 1:(length(denoised_data)-start_idx)
        output_data{row,col} = denoised_data(row+start_idx);
    end
end

% 写入Excel文件
xlswrite('L2输出.xlsx', headers, 1, 'A1');  % 写入表头
xlswrite('L2输出.xlsx', output_data, 1, 'A2');  % 写入数据