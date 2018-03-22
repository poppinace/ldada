%
% An Embarassingly Simple Approach to Visual Domain Adaptation
% IEEE Transactions on Image Processing, 2018
% Hao Lu, Chunhua Shen, Zhiguo Cao, Yang Xiao, Anton van den Hengel

clear; close all; clc

addpath('liblinear-2.1/matlab');

% set seed
rng('default')
  
domain = {'banjaluka', 'remotesensing', 'ucmerced-landuse'};

ex_setting{1} = {domain{1}, domain{2}};
ex_setting{2} = {domain{2}, domain{1}};
ex_setting{3} = {domain{1}, domain{3}};
ex_setting{4} = {domain{3}, domain{1}};
ex_setting{5} = {domain{2}, domain{3}};
ex_setting{6} = {domain{3}, domain{2}};

NUM_EX = length(ex_setting);

% parameter initialization
opt = paramInit;

if opt.nclasses ~= 5, error('A wrong dataset is chosen!'); end
  
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
