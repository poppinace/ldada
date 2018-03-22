function featPool = extractConvActivations(net, im3u, opt)

if ndims(im3u) ~= 3, im3u = cat(3, im3u, im3u, im3u); end

% image preprocessing
im3f = single(im3u);
r = net.meta.normalization.imageSize(1:2); % fixed normalization
im3f = imresize(im3f, r);
im3f = bsxfun(@minus, im3f, net.meta.normalization.averageImage(1, 1, :));

% feedforward CNN model
if isstruct(net)
  featCNN = vl_simplenn(net, im3f);
else
  net.eval({'data', im3f});
end
  
switch opt.modelType
  case 'VGG-M'
    featPool = squeeze(featCNN(18).x);
  case 'VGG-VD-16'
    featPool = squeeze(featCNN(36).x);
    error('unsupported CNN model')
end
    
end