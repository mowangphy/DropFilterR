function [net, info] = cnn_cifar_dropfilter_plus(varargin)
% CNN_CIFAR   Demonstrates MatConvNet on CIFAR-10
%    The demo includes two standard model: LeNet and Network in
%    Network (NIN). Use the 'modelType' option to choose one.

run(fullfile(fileparts(mfilename('fullpath')), '../../matlab/vl_setupnn.m')) ;

opts.modelType = 'lenet' ;
[opts, varargin] = vl_argparse(opts, varargin) ;
opts.train.batchSize = 100 ;
opts.train.numEpochs = 300 ;

switch opts.modelType
  case 'lenet'
    lr_schedule = zeros(1, opts.train.numEpochs);
	lr_init = 0.06;
	for i = 1:opts.train.numEpochs
		if (mod(i, 25) == 0)
			lr_init = lr_init / 2;% 1.5
		end
		lr_schedule(i) = lr_init;
	end
	opts.train.learningRate = lr_schedule;
	
    opts.train.weightDecay = 0.0005 ;
  case 'nin'
    opts.train.learningRate = [0.5*ones(1,30) 0.1*ones(1,10) 0.02*ones(1,10)] ;
    opts.train.weightDecay = 0.0005 ;
  otherwise
    error('Unknown model type %s', opts.modelType) ;
end

% opts.expDir = fullfile('data', sprintf('cifar-%s', opts.modelType)) ;
opts.expDir = fullfile('data','cifar-baseline-dropfilterplus') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile('data','cifar') ;
% opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.dataClass = 0; % RGB: 0; YUV: 1
opts.whitenData = false ; %data have been whitened
opts.contrastNormalization = false ;

if (opts.dataClass == 0)
	opts.imdbPath = '/eecs/research/asr/hengyue/DFT_DNN/Data/cifar/imdb_cifar.mat';
else
	opts.imdbPath = '/eecs/research/asr/hengyue/dataset/imdb_YUV_normalized.mat';
end

opts.train.continue = false ;
opts.train.gpus = 1 ;
opts.train.expDir = opts.expDir ;

opts.train.momentum = 0.9 ;
opts.train.isDropInput = 0;
opts.train.isDropFilter = 1;
opts.train.dropFilterRate_init = 0.06;
opts.train.dropFilterRate_final = 0.30;
opts.train.isDifferentRate = 0;
opts.train.learningRate_ReLU = 0.20;
opts.train.lrReLU_decay = 0.998;
opts.train.dropInputRate = 0.0;
opts.train.isDataAug = 0;
opts = vl_argparse(opts, varargin) ; 

% --------------------------------------------------------------------
%                                               Prepare data and model
% --------------------------------------------------------------------

switch opts.modelType
  case 'lenet', net = cnn_cifar_init_2016_dropfilter_plus(opts) ;
  % case 'lenet', net = cnn_cifar_init(opts) ;
  case 'nin',   net = cnn_cifar_init_nin(opts) ;
end



if exist(opts.imdbPath, 'file')
  load(opts.imdbPath) ;
  
  if (opts.dataClass == 0)
	dataMean = mean(imdb.images.data(:,:,:,1:50000), 4);
	imdb.images.data = bsxfun(@minus, imdb.images.data, dataMean);
  else
	imdb.images.labels = imdb.images.label;
  end
  
  
  if opts.whitenData
	%imdb.images.data = permute(imdb.images.data,[2 1 3 4]) ;
	z = reshape(imdb.images.data,[],60000) ;
	% W = z(:,imdb.images.set == 1)*z(:,imdb.images.set == 1)'/60000 ;
	% [V,D] = eig(W) ;
	% the scale is selected to approximately preserve the norm of W
	% d2 = diag(D) ;
	% en = sqrt(mean(d2)) ;
	% z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
	
	sigma = z * transpose(z) / size(z, 2);%sigma is a n-by-n matrix
	%%perform SVD
	[U,S,V] = svd(sigma);
	disp('Image processing using ZCAwhitening');
    epsilon = 0.01;
    z = U * diag(1./sqrt(diag(S) + epsilon)) * U' * z;
	imdb.images.data = reshape(z, 32, 32, 3, []) ;
  end
  
else
  imdb = getCifarImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
% imdb.images.data = bsxfun(@minus, imdb.images.data, mean(imdb.images.data,4)) ;
% k = imdb.images.data(:, :, 1, 1);
% maxSample = max(max(k))
% imdb.images.data = imdb.images.data * 255;

imdb.images.data = single(imdb.images.data);

fprintf('2016 new structure, 20190628 speed up, best lrrate baseline, 0.06-0.30 \n'); 
[net, info] = cnn_train_dropfilter_plus(net, imdb, @getBatch, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;

save info_baseline.mat info;
	
[valError, idx] = min(info.val.error(1, :));
fprintf('Minimum validation error is %.6g, number of epoch is %d \n', valError, idx);


% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
%if rand > 0.5, im=fliplr(im) ; end

% --------------------------------------------------------------------
function imdb = getCifarImdb(opts)
% --------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
unpackPath = fullfile(opts.dataDir, 'cifar-10-batches-mat');
files = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:5, 'UniformOutput', false) ...
  {'test_batch.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1, 5), 3]);

if any(cellfun(@(fn) ~exist(fn, 'file'), files))
  url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz' ;
  fprintf('downloading %s\n', url) ;
  untar(url, opts.dataDir) ;
end

data = cell(1, numel(files));
labels = cell(1, numel(files));
sets = cell(1, numel(files));
for fi = 1:numel(files)
  fd = load(files{fi}) ;
  data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
  labels{fi} = fd.labels' + 1; % Index from 1
  sets{fi} = repmat(file_set(fi), size(labels{fi}));
end

set = cat(2, sets{:});
data = single(cat(4, data{:}));

% remove mean in any case
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean);

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng

if opts.contrastNormalization
  z = reshape(data,[],60000) ;
  z = bsxfun(@minus, z, mean(z,1)) ;
  n = std(z,0,1) ;
  z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
  data = reshape(z, 32, 32, 3, []) ;
end

if opts.whitenData
  z = reshape(data,[],60000) ;
  W = z(:,set == 1)*z(:,set == 1)'/60000 ;
  [V,D] = eig(W) ;
  % the scale is selected to approximately preserve the norm of W
  d2 = diag(D) ;
  en = sqrt(mean(d2)) ;
  z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
  data = reshape(z, 32, 32, 3, []) ;
end

clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

imdb.images.data = data ;
imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.label_names;
