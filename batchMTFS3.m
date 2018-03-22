%
% An Embarassingly Simple Approach to Visual Domain Adaptation
% IEEE Transactions on Image Processing, 2018
% Hao Lu, Chunhua Shen, Zhiguo Cao, Yang Xiao, Anton van den Hengel

clear; close all; clc

addpath('liblinear-2.1/matlab');

% set seed
rng('default')
  
domain = {
  'Gucheng2014'    % 1
  'Taian2010_1'    % 2
  'Taian2010_2'    % 3
  'Taian2011_1'    % 4
  'Taian2011_2'    % 5
  'Taian2012_1'    % 6
  'Taian2012_2'    % 7
  'Zhengzhou2010'  % 8
  'Zhengzhou2011'  % 9
  'Zhengzhou2012'  % 10
};

ex_setting{1} = {domain{10}, domain{7}};
ex_setting{2} = {domain{1}, domain{7}};
ex_setting{3} = {domain{4}, domain{9}};
ex_setting{4} = {domain{9}, domain{7}};
ex_setting{5} = {domain{4}, domain{10}};
ex_setting{6} = {domain{1}, domain{4}};

NUM_EX = length(ex_setting);

% parameter initialization
opt = paramInit;

if opt.nclasses ~= 3, error('A wrong dataset is chosen!'); end
  
tTotal = tic;
acc = cell(3, NUM_EX+1);
for i = 1:NUM_EX
  opt.sourcedir = ex_setting{i}{1};
  opt.targetdir = ex_setting{i}{2};
  
  annotations = genAnnotations(opt);
  
  [meanAcc, stdAcc] = trainTestDA(opt, annotations);
  
  acc{1, i} = {meanAcc.na, stdAcc.na};
  acc{2, i} = {meanAcc.ldada, stdAcc.ldada};
  acc{3, i} = {meanAcc.oracle, stdAcc.oracle};
  
  % print results
  print_on_screen(acc, NUM_EX, annotations, i)
end

% print results
print_on_screen(acc, NUM_EX)

elapsedTime  = toc(tTotal);
fprintf('overall time elapsed is %f\n', elapsedTime)
