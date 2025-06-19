function dist = dtw_distance(x, y)
% DTW_DISTANCE - 计算两个时间序列之间的DTW距离
% 输入:
%   x - 第一个时间序列
%   y - 第二个时间序列
% 输出:
%   dist - DTW距离

% 获取序列长度
n = length(x);
m = length(y);

% 初始化DTW矩阵
D = zeros(n+1, m+1);
D(1,:) = inf;
D(:,1) = inf;
D(1,1) = 0;

% 计算DTW矩阵
for i = 2:n+1
    for j = 2:m+1
        cost = abs(x(i-1) - y(j-1));
        D(i,j) = cost + min([D(i-1,j), D(i,j-1), D(i-1,j-1)]);
    end
end

% 返回DTW距离
dist = D(n+1,m+1);
end