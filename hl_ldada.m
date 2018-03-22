function [Xs, Xt, Ys, Yt, acc_oracle] = hl_ldada(S, T, opt)
%HL_LDADA implements LDA-inspired Domain Adaptation (LDADA) appearing in
% An Embarassingly Simple Approach to Visual Domain Adaptation
% IEEE Transactions on Image Processing, 2018
% Hao Lu, Chunhua Shen, Zhiguo Cao, Yang Xiao, Anton van den Hengel
%

ldada = opt.ldada;

if ~isfield(ldada, 'maxiter'), ldada.maxiter = 10; end
if ~isfield(ldada, 'predictor'), ldada.predictor = 'ldada'; end
if ~isfield(ldada, 'verbose'), ldada.verbose = false; end

Xs = S.Xs;
Xt = T.Xt;
Ys = S.Ys;
Yt = T.Yt;

% mean subtraction
meanXs = mean(Xs, 2);
meanXt = mean(Xt, 2);
Xs = bsxfun(@minus, Xs, meanXs);
Xt = bsxfun(@minus, Xt, meanXt);

% L2 normalization
Xs = bsxfun(@times, Xs, 1./max(1e-12, sqrt(sum(Xs.^2))));
Xt = bsxfun(@times, Xt, 1./max(1e-12, sqrt(sum(Xt.^2))));

% get the lower bound and higher bound
acc_oracle = 0;
if ldada.verbose
  acc_na = normAcc(Yt, learnPredictSVM(Xs, Xt, Ys, Yt));
  W = learning_csproj(Xs, Xt, Ys, Yt);
  acc_oracle = normAcc(Yt, learn_predict_labels(W'*Xs, W'*Xt, Ys, Yt, ldada));
  fprintf('\nLower  bound, acc_na     = %.1f\n', acc_na)
  fprintf('Higher bound, acc_oracle = %.1f\n', acc_oracle)
end

% pseudo set setting
P.Xt = []; 
P.Yt = [];
P.id = [];
P.predictor = ldada.predictor;
P.nclasses = opt.nclasses;

% init W
W = eye(size(Xs, 1));
Wp = ones(size(Xs, 1), opt.nclasses);

accp = 101;
for t = 1:ldada.maxiter
 if t > 1, Wp = W; accp = P.acc_cv; end
  % update Yt given W
  P = labeling_target(Xs, Xt, Ys, Yt, W, P);
  % update W given Yt
  W = learning_csproj(Xs, P.Xt, Ys, P.Yt);
  % evaluation
  [~, C] = max(W' * P.Xt, [], 1);
  purity = normAcc(Yt(P.id), C') / 100;
  deltaW = norm(W - Wp, 'fro') / norm(Wp, 'fro');
  deltaAcc = abs(P.acc_cv - accp) / accp;
  if ldada.verbose
    acc_ldada = normAcc(Yt, learn_predict_labels(W'*Xs, W'*Xt, Ys, Yt, ldada));
    fprintf(['Iteration %d, acc_cspro = %.1f, acc_cv = %.4f, purity = %.4f, ' ...
     'deltaW = %.2e, deltaAcc = %.2e\n'], t, acc_ldada, P.acc_cv, purity, deltaW, deltaAcc)
  end
  if deltaW < 1e-3 || deltaAcc < 1e-3, break; end
end

Xs = W' * Xs;
Xt = W' * Xt;

end

function W = learning_csproj(Xs, Xt, Ys, Yt)

[d, ~] = size(Xt);

if size(Ys, 2) ~= 1, Cs = find(sum(Ys) ~= 0); else, Cs = unique(Ys)'; end
if size(Yt, 2) ~= 1, Ct = find(sum(Yt) ~= 0); else, Ct = unique(Yt)'; end
if numel(Cs) < numel(Ct), C = Cs; else, C = Ct; end

if size(Ys, 2) == 1, idxsi = bsxfun(@eq, Ys, C); else, idxsi = Ys; end
if size(Yt, 2) == 1, idxti = bsxfun(@eq, Yt, C); else, idxti = Yt; end

Ms = bsxfun(@rdivide, Xs * idxsi, sum(idxsi)+1e-10);
Mt = bsxfun(@rdivide, Xt * idxti, sum(idxti)+1e-10);
Ms_ = bsxfun(@rdivide, Xs * ~idxsi, sum(~idxsi)+1e-10);
Mt_ = bsxfun(@rdivide, Xt * ~idxti, sum(~idxti)+1e-10);

Sb = (Ms + Mt - Ms_ - Mt_) / 2;
v = sum(Ms .* Sb);
u = sum(Ms.^2) .* sum(Mt.^2);

W = zeros(d, max(numel(Cs), numel(Ct)));
W(:, C) = bsxfun(@times, v ./ u, Mt);

end

function P = labeling_target(Xs, Xt, Ys, Yt, W, P)

model = train(Ys, sparse(double(Xs'*W)), '-s 5 -c 1 -B 1 -q');
score = model.w * [W'*Xt; ones(1, size(Xt, 2))];

th = mean(score, 2)';
[confidence, C] = max(score, [], 1);
idxpos = confidence > th(C);

P.id = idxpos;
P.Xt = Xt(:, idxpos);
P.Yt = C(idxpos)';

% ciricular validation
P.acc_cv = 0;
Wt = learning_csproj(P.Xt, Xs, P.Yt, Ys);
P.acc_cv = normAcc(Ys, learn_predict_labels(Wt'*P.Xt, Wt'*Xs, P.Yt, Ys, P));

end

function C = learn_predict_labels(Xs, Xt, Ys, Yt, opt)

switch opt.predictor
  case 'ldada'
    [~, C] = max(Xt, [], 1); C = C';
  case 'svm'
    C = learnPredictSVM(Xs, Xt, Ys, Yt);
  otherwise
    error('unsupported predictor option')
end

end
