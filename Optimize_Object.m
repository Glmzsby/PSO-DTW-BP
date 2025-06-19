function [error] = Optimize_Object(pop, Hiddennum, net, Train_InPut, Train_OutPut)
inputnum  = size(Train_InPut,1);  % 输入层节点数
outputnum = size(Train_OutPut,1);  % 输出层节点数

%%  提取权值和阈值
w1 = pop(1 : inputnum * Hiddennum);
B1 = pop(inputnum * Hiddennum + 1 : inputnum * Hiddennum + Hiddennum);
w2 = pop(inputnum * Hiddennum + Hiddennum + 1 : ...
    inputnum * Hiddennum + Hiddennum + Hiddennum * outputnum);
B2 = pop(inputnum * Hiddennum + Hiddennum + Hiddennum * outputnum + 1 : ...
    inputnum * Hiddennum + Hiddennum + Hiddennum * outputnum + outputnum);
 
%%  网络赋值
net.Iw{1, 1} = reshape(w1, Hiddennum, inputnum );
net.Lw{2, 1} = reshape(w2, outputnum, Hiddennum);
net.b{1} = reshape(B1, Hiddennum, 1);
net.b{2} = B2';

%%  网络训练
net = train(net,Train_InPut,Train_OutPut);

%%  仿真测试
t_sim = sim(net,Train_InPut);

%%  适应度值
% 使用DTW距离作为适应度值
error = 0;
for i = 1:size(Train_OutPut,2)
    error = error + dtw_distance(t_sim(:,i), Train_OutPut(:,i));
end

end

