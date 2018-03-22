function [meanAcc, stdAcc] = trainTestDA(opt, annotations)
%
% An Embarassingly Simple Approach to Visual Domain Adaptation
% IEEE Transactions on Image Processing, 2018
% Hao Lu, Chunhua Shen, Zhiguo Cao, Yang Xiao, Anton van den Hengel

dataset = opt.dataset;
featureType = opt.featureType;
modelType = opt.modelType;

sourceName = annotations.prm.sourceName; 
targetName = annotations.prm.targetName;

nclasses = opt.nclasses;
ntrials = opt.ntrials;
if opt.nclasstrain < 0
  ntrials = 1;
end

cachedir = opt.cachedir;
modeldir = opt.modeldir;

imPathSource = fullfile( ...
  opt.datasetdir, ...
  opt.sourcedir, ...
  opt.imagedir ...
);
imPathTarget = fullfile( ...
  opt.datasetdir, ...
  opt.targetdir, ...
  opt.imagedir ...
);

EN_NA     = 1;
EN_LDADA  = 1;
  
% load pretrained model
switch modelType
  case 'VGG-M'
    % net = load(fullfile(modeldir, 'imagenet-vgg-m'));
  case 'VGG-VD-16'
    % net = load(fullfile(modeldir, 'imagenet-vgg-verydeep-16'));
  otherwise
    net = 0;
    fprintf('CNN model is not loaded\n')
end

% extract features on the source domain
try
  source = load( ...
    fullfile( ...
      cachedir, ...
      [dataset '_' sourceName '_' modelType '_features_' featureType] ...
    ) ...
  );
catch
  categories = annotations.prm.categories;
  imagenames = annotations.imagenames.source;
  features = cell(1, nclasses);
  labels = cell(1,nclasses);
  tic;
  for i = 1:nclasses
    cls = categories{i};
    imNameCls = imagenames{i};
    nimages = length(imNameCls);
    feats = cell(1, nimages);
    for j = 1:nimages
      if toc > 5
        fprintf('%s: %s %s: %d/%d\n', cls, dataset, sourceName, j, nimages);
        drawnow;
        tic;
      end
      im3u = imread(fullfile(imPathSource, cls, imNameCls{j}));
      feats{j} = extractConvActivations(net, im3u, opt);
    end
    features{i} = feats;
    labels{i} = i * ones(nimages, 1);
  end
  save( ...
    fullfile( ...
      cachedir, ...
      [dataset '_' sourceName '_' modelType '_features_' featureType] ...
    ), ...
    'features', ...
    'labels' ...
  );
  source = load( ...
    fullfile( ...
      cachedir, ...
      [dataset '_' sourceName '_' modelType '_features_' featureType] ...
    ) ...
  );
end

% extract features on the target domain
try
  target = load( ...
    fullfile( ...
      cachedir, ...
      [dataset '_' targetName '_' modelType '_features_' featureType] ...
    ) ...
  );
catch
  categories = annotations.prm.categories;
  imagenames = annotations.imagenames.target;
  features = cell(1, nclasses);
  labels = cell(1,nclasses);
  tic;
  for i = 1:nclasses
    cls = categories{i};
    imNameCls = imagenames{i};
    nimages = length(imNameCls);
    feats = cell(1, nimages);
    for j = 1:nimages
      if toc > 5
        fprintf('%s: %s %s: %d/%d\n', cls, dataset, targetName, j, nimages);
        drawnow;
        tic;
      end
      im3u = imread(fullfile(imPathTarget, cls, imNameCls{j}));
      feats{j} = extractConvActivations(net, im3u, opt);
    end
    features{i} = feats;
    labels{i} = i * ones(nimages, 1);
  end
  save( ...
    fullfile( ...
      cachedir, ...
      [dataset '_' targetName '_' modelType '_features_' featureType] ...
     ), ...
     'features', ...
     'labels' ...
  );
  target = load( ...
    fullfile( ...
      cachedir, ...
      [dataset '_' targetName '_' modelType '_features_' featureType] ...
    ) ...
  );
end

clear net

% parse features
for i = 1:nclasses
  source.features{i} = cat(2, source.features{i}{:});
  target.features{i} = cat(2, target.features{i}{:});
end

% train & test
idxSourceTrials = annotations.train.source;
idxTestTargetTrials = annotations.test.target;

acc_na = zeros(1, ntrials);
acc_ldada = zeros(1, ntrials);
acc_oracle = zeros(1, ntrials);

tic;
for i = 1:ntrials
  fprintf('.')
  idxSource = idxSourceTrials{i};
  idxTargetTest = idxTestTargetTrials{i};
  % prase training and test data from both domain
  feats_source = [];
  feats_target = [];
  labels_source = [];
  labels_target = [];
  for j = 1:nclasses
    % index
    idxClassSource = idxSource{j};
    idxClassTargetTest = idxTargetTest{j};
    % features
      feats_source = cat( ...
        2, ...
        feats_source, ...
        source.features{j}(:, idxClassSource) ...
      );
      feats_target = cat( ...
        2, ...
        feats_target, ...
        target.features{j}(:, idxClassTargetTest) ...
      );
    % labels
    labels_source = cat( ...
      1, ...
      labels_source, ...
      source.labels{j}(idxClassSource) ...
    );
    labels_target = cat( ...
      1, ...
      labels_target, ...
      target.labels{j}(idxClassTargetTest) ...
    );
  end
  
  S.Xs = feats_source;
  S.Ys = labels_source;
  T.Xt = feats_target;
  T.Yt = labels_target;

% ----------------------------------------------
% baseline no alignment
% ----------------------------------------------
if EN_NA
  [Xs, Xt, Ys, Yt] = hl_na(S, T, opt);
  C = learnPredictSVM(Xs, Xt, Ys, Yt);
  acc_na(i) = normAcc(Yt, C);
end

% ----------------------------------------------
% LDA-inspired Domain Adaptation
% ----------------------------------------------
if EN_LDADA
  [Xs, Xt, Ys, Yt, acc_oracle(i)] = hl_ldada(S, T, opt);
  [~, C] = max(Xt, [], 1); C = C';
  acc_ldada(i) = normAcc(Yt, C);
end

end

time = toc;
fprintf('\nAverage time for each trial is %4.2f\n', time / ntrials)

meanAcc.na = mean(acc_na); stdAcc.na = std(acc_na);
meanAcc.ldada = mean(acc_ldada); stdAcc.ldada = std(acc_ldada);
meanAcc.oracle = mean(acc_oracle); stdAcc.oracle = std(acc_oracle);

