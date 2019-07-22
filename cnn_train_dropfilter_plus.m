function [net, info] = cnn_train_dropkernel(net, imdb, getBatch, varargin)
% CNN_TRAIN   Demonstrates training a CNN
%    CNN_TRAIN() is an example learner implementing stochastic
%    gradient descent with momentum to train a CNN. It can be used
%    with different datasets and tasks by providing a suitable
%    getBatch function.
%
%    The function automatically restarts after each training epoch by
%    checkpointing.
%
%    The function supports training on CPU or on one or more GPUs
%    (specify the list of GPU IDs in the `gpus` option). Multi-GPU
%    support is relatively primitive but sufficient to obtain a
%    noticable speedup.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.batchSize = 100 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.numEpochs = 300 ;
opts.gpus = 2 ; % which GPU devices to use (none, one, or more)
opts.learningRate = 0.001 ;
opts.continue = false ;
opts.expDir = fullfile('data','exp') ;
opts.conserveMemory = false ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.prefetch = false ;
opts.weightDecay = 0.004 ;
opts.learningRate_ReLU = 0.10;
opts.lrReLU_decay = 0.998;
opts.momentum = 0.95 ;
opts.momentum_ReLU = 0.90;
opts.errorFunction = 'multiclass' ;
opts.errorLabels = {} ;
opts.plotDiagnostics = false ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.isDropInput = 0;
opts.isDropFilter = 1;
opts.dropFilterRate_init = 0.05;
opts.dropFilterRate_final = 0.25;
opts.dropFilterRate = 0.25;
opts.isDifferentRate = 0;
opts.dropInputRate = 0.0;
opts.isDataAug = 0;
opts.AGDLayer = 0;

opts = vl_argparse(opts, varargin) ;


if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

evaluateMode = isempty(opts.train) ;

if ~evaluateMode
  for i=1:numel(net.layers)
    if isfield(net.layers{i}, 'weights')
      J = numel(net.layers{i}.weights) ;
      for j=1:J
        net.layers{i}.momentum{j} = zeros(size(net.layers{i}.weights{j}), 'single') ;
      end
      if ~isfield(net.layers{i}, 'learningRate')
        net.layers{i}.learningRate = ones(1, J, 'single') ;
      end
      if ~isfield(net.layers{i}, 'weightDecay')
        net.layers{i}.weightDecay = ones(1, J, 'single') ;
      end
    end
    % Legacy code: will be removed
    if isfield(net.layers{i}, 'filters')
      net.layers{i}.momentum{1} = zeros(size(net.layers{i}.filters), 'single') ;
      net.layers{i}.momentum{2} = zeros(size(net.layers{i}.biases), 'single') ;
      if ~isfield(net.layers{i}, 'learningRate')
        net.layers{i}.learningRate = ones(1, 2, 'single') ;
      end
      if ~isfield(net.layers{i}, 'weightDecay')
        net.layers{i}.weightDecay = single([1 0]) ;
      end
    end
	
	if ( strcmp(net.layers{i}.type, 'softrelu') ) 
		net.layers{i}.momentum{1} = zeros(size(net.layers{i}.alpha), 'single') ;
		net.layers{i}.momentum{2} = zeros(size(net.layers{i}.beta), 'single') ;	
	end
  end
end

% setup GPUs
numGpus = numel(opts.gpus) ;
if numGpus > 1
  if isempty(gcp('nocreate')),
    parpool('local',numGpus) ;
    spmd, gpuDevice(opts.gpus(labindex)), end
  end
elseif numGpus == 1
  gpuDevice(opts.gpus)
end
if exist(opts.memoryMapFile), delete(opts.memoryMapFile) ; end

% setup error calculation function
if isstr(opts.errorFunction)
  switch opts.errorFunction
    case 'none'
      opts.errorFunction = @error_none ;
    case 'multiclass'
      opts.errorFunction = @error_multiclass ;
      if isempty(opts.errorLabels), opts.errorLabels = {'top1e', 'top5e'} ; end
    case 'binary'
      opts.errorFunction = @error_binary ;
      if isempty(opts.errorLabels), opts.errorLabels = {'bine'} ; end
    otherwise
      error('Uknown error function ''%s''', opts.errorFunction) ;
  end
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------
opts.dropFilterRate = opts.dropFilterRate_init;
for epoch=1:opts.numEpochs
  learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;

  % fast-forward to last checkpoint
  modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d-AGD%d.mat', ep, opts.AGDLayer));
  modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;
  if opts.continue
    if exist(modelPath(epoch),'file')
      if epoch == opts.numEpochs
        load(modelPath(epoch), 'net', 'info') ;
      end
      continue ;
    end
    if epoch > 1
      fprintf('resuming by loading epoch %d\n', epoch-1) ;
      load(modelPath(epoch-1), 'net', 'info') ;
    end
  end

  % move CNN to GPU as needed
  if numGpus == 1
    net = vl_simplenn_move(net, 'gpu') ;
  elseif numGpus > 1
    spmd(numGpus)
      net_ = vl_simplenn_move(net, 'gpu') ;
    end
  end

  % train one epoch and validate
  train = opts.train(randperm(numel(opts.train))) ; % shuffle
  val = opts.val ;
  if (epoch <= 100)&&(opts.dropFilterRate <= 0.3)
	opts.dropFilterRate = opts.dropFilterRate * 1.01;
	% opts.isDropFilter = 0;
  else
	opts.dropFilterRate = opts.dropFilterRate_final;
  end
  
  if numGpus <= 1
	[net,stats.train] = process_epoch(opts, getBatch, epoch, train, learningRate, imdb, net) ;
    [~,stats.val] = process_epoch(opts, getBatch, epoch, val, 0, imdb, net) ;
  else
    spmd(numGpus)
      [net_, stats_train_] = process_epoch(opts, getBatch, epoch, train, learningRate, imdb, net_) ;
      [~, stats_val_] = process_epoch(opts, getBatch, epoch, val, 0, imdb, net_) ;
    end
    stats.train = sum([stats_train_{:}],2) ;
    stats.val = sum([stats_val_{:}],2) ;
  end
  
  opts.learningRate_ReLU = opts.learningRate_ReLU * opts.lrReLU_decay;

  % save
  if evaluateMode, sets = {'val'} ; else sets = {'train', 'val'} ; end
  for f = sets
    f = char(f) ;
    n = numel(eval(f)) ;
    info.(f).speed(epoch) = n / stats.(f)(1) ;
    info.(f).objective(epoch) = stats.(f)(2) / n ;
    info.(f).error(:,epoch) = stats.(f)(3:end) / n ;
  end
  if numGpus > 1
    spmd(numGpus)
      net_ = vl_simplenn_move(net_, 'cpu') ;
    end
    net = net_{1} ;
  else
    net = vl_simplenn_move(net, 'cpu') ;
  end
  if ~evaluateMode
	if (mod(epoch, 100) == 0)
		save(modelPath(epoch), 'net', 'info') ;
	end
  end

  figure(1) ; clf ;
  hasError = isa(opts.errorFunction, 'function_handle') ;
  subplot(1,1+hasError,1) ;
  if ~evaluateMode
    plot(1:epoch, info.train.objective, '.-', 'linewidth', 2) ;
    hold on ;
  end
  plot(1:epoch, info.val.objective, '.--') ;
  xlabel('training epoch') ; ylabel('energy') ;
  grid on ;
  h=legend(sets) ;
  set(h,'color','none');
  title('objective') ;
  if hasError
    subplot(1,2,2) ; leg = {} ;
    if ~evaluateMode
      plot(1:epoch, info.train.error', '.-', 'linewidth', 2) ;
      hold on ;
      leg = horzcat(leg, strcat('train ', opts.errorLabels)) ;
    end
    plot(1:epoch, info.val.error', '.--') ;
    leg = horzcat(leg, strcat('val ', opts.errorLabels)) ;
    set(legend(leg{:}),'color','none') ;
    grid on ;
    xlabel('training epoch') ; ylabel('error') ;
    title('error') ;
  end
  drawnow ;
  print(1, modelFigPath, '-dpdf') ;
end

% -------------------------------------------------------------------------
function err = error_multiclass(opts, labels, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
[~,predictions] = sort(predictions, 3, 'descend') ;
error = ~bsxfun(@eq, predictions, reshape(labels, 1, 1, 1, [])) ;
err(1,1) = sum(sum(sum(error(:,:,1,:)))) ;
err(2,1) = sum(sum(sum(min(error(:,:,1:5,:),[],3)))) ;

% -------------------------------------------------------------------------
function err = error_binaryclass(opts, labels, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
error = bsxfun(@times, predictions, labels) < 0 ;
err = sum(error(:)) ;

% -------------------------------------------------------------------------
function err = error_none(opts, labels, res)
% -------------------------------------------------------------------------
err = zeros(0,1) ;

% -------------------------------------------------------------------------
function  [net,stats,prof] = process_epoch(opts, getBatch, epoch, subset, learningRate, imdb, net)
% -------------------------------------------------------------------------

% validation mode if learning rate is zero
training = learningRate > 0 ;
if training, mode = 'training' ; else, mode = 'validation' ; end
if nargout > 2, mpiprofile on ; end

numGpus = numel(opts.gpus) ;
if numGpus >= 1
  one = gpuArray(single(1)) ;
else
  one = single(1) ;
end
res = [] ;
mmap = [] ;
stats = [] ;

top_1_Error_collection = zeros(1, 2); 
for t=1:opts.batchSize:numel(subset)
  fprintf('%s: epoch %02d: batch %3d/%3d: ', mode, epoch, ...
          fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
  batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
  batchTime = tic ;
  numDone = 0 ;
  error = [] ;
  for s=1:opts.numSubBatches
    % get this image batch and prefetch the next
    batchStart = t + (labindex-1) + (s-1) * numlabs ;
    batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
	[im, labels] = getBatch(imdb, batch) ;

    if opts.prefetch
      if s==opts.numSubBatches
        batchStart = t + (labindex-1) + opts.batchSize ;
        batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
      else
        batchStart = batchStart + numlabs ;
      end
      nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
      getBatch(imdb, nextBatch) ;
    end
	
	% Do horizontal flip for 50% images
	% I2 = imcrop(I, [xmin, ymin, width, height]) crop should work with resize
	% im(:, :, :, iN) = fliplr( im(:, :, :, iN) ); reflection
	% I2 = imrotate(I, angle, 'bilinear', 'crop')
	% I2 = imtranslate(I, [x, y], fill); NO!!!
	
	
	if training
		if opts.isDataAug%  && (epoch <= (opts.numEpochs-100) )
			fprintf('Do image augmentation (rotation+translation+scale+colorspace)... ');
			[H0, W0, C0, N0] = size(im);
			randIdx = randperm(N0); % translation
			randIdx2 = randperm(N0); % rotate
			randIdx3 = randperm(N0); % scale
			randIdx4 = randperm(N0); % RGB casting
			for iN = 1:N0
				if (randIdx(iN) <= (N0/2))
					transRangeX = (rand(1) - 0.5) * 10;
					transRangeY = (rand(1) - 0.5) * 10;
					im(:, :, :, iN) = imtranslate(im(:, :, :, iN),[transRangeX, transRangeY]);
				end
				
				if (randIdx2(iN) <= (N0/2))
					angle = (rand(1) - 0.5) * 10; % -5 ~ 5
					im(:, :, :, iN) = imrotate( im(:, :, :, iN), angle, 'bilinear', 'crop' );
				end
				
				if (randIdx3(iN) <= (N0/2))
					xmin = randperm(4);
					ymin = randperm(4);
					width = randperm(4) + 24;
					height = randperm(4) + 24;
					tmpIm = imcrop(im(:, :, :, iN), [xmin(1), ymin(1), width(1), height(1)]);
					im(:, :, :, iN) = imresize(tmpIm, [32, 32]);
				end
				
				if (randIdx4(iN) <= (N0/2))
					tagR = rand(1);
					tagG = rand(1);
					tagB = rand(1);
					% [H0, W0, C0, N0] = size(im);
					maskRGB = ( rand(H0, W0, C0, N0)*0.1 - 0.05 ) + 1;% 0-1 -> 0.95-1.05
					if (tagR < 0.5)
						im(:, :, 1, iN) = im(:, :, 1, iN) .* maskRGB(:, :, 1, iN);
					end
					if (tagG < 0.5)
						im(:, :, 2, iN) = im(:, :, 2, iN) .* maskRGB(:, :, 2, iN);
					end				
					if (tagB < 0.5)
						im(:, :, 3, iN) = im(:, :, 3, iN) .* maskRGB(:, :, 3, iN);
					end
				end
			end	
			fprintf('Done ... \n');
		end
	end
	
	if opts.isDropInput
		if training
			fprintf('Do drop input ');
			[H, W, C, N] = size(im);
			mask = single(rand(H, W) >= opts.dropInputRate) ;
			im = bsxfun(@times, im, mask);
		else
			% im = im * (1 - opts.dropInputRate);
		end
	end

    if numGpus >= 1
      im = gpuArray(im) ;
    end
	
	if training
		for l=1:numel(net.layers)
			if ~strcmp(net.layers{l}.type, 'conv_dropfilter'), continue ; end	
			% net.layers{l}.rand0_1 = rand(net.layers{l}.outSizeX, net.layers{l}.outSizeY, net.layers{l}.outSizeZ, net.layers{l}.nofSamples);
			% net.layers{l}.rand0_1n = randn(net.layers{l}.outSizeX, net.layers{l}.outSizeY, net.layers{l}.outSizeZ, net.layers{l}.nofSamples);
			% net.layers{l}.rand0_1_mask = (net.layers{l}.rand0_1 < opts.dropFilterRate); 
			
			net.layers{l}.rand0_1 = rand(net.layers{l}.outSizeX, net.layers{l}.outSizeY, net.layers{l}.outSizeZ);
			net.layers{l}.rand0_1n = randn(net.layers{l}.outSizeX, net.layers{l}.outSizeY, net.layers{l}.outSizeZ, net.layers{l}.nofSamples);
			net.layers{l}.rand0_1n = sum(net.layers{l}.rand0_1n, 4);
			
			if(opts.isDifferentRate)
				dropFilterRate_matrix = rand(size(net.layers{l}.rand0_1))*0.2 + 0.9; % 0.9-1.1
				
				% dropFilterRate_matrix = rand(size(net.layers{l}.rand0_1))*0.2 + 0.9;
				
				net.layers{l}.rand0_1_mask = (net.layers{l}.rand0_1 < dropFilterRate_matrix);
			else
				net.layers{l}.rand0_1_mask = (net.layers{l}.rand0_1 < 1 - opts.dropFilterRate);
			end
			
			% make a small modification: using a different dropfilter rate on the different regions of the mask
		end
	else
		% for l=1:numel(net.layers)
			% if ~strcmp(net.layers{l}.type, 'conv_dropfilter'), continue ; end	
				% if isfield(net.layers{l}, 'weights')
					% net.layers{l}.weights{1} = net.layers{l}.weights{1} * (1 - opts.dropFilterRate) ;
				% end
				% if isfield(net.layers{l}, 'filters')
					% net.layers{l}.filters = net.layers{l}.filters * (1 - opts.dropFilterRate) ;
				% end	
		% end 
	end

	% if ~training  %need to be confirmed: is weight or feature???
		% for l=1:numel(net.layers)
			% if ~strcmp(net.layers{l}.type, 'conv'), continue ; end	
			% if isfield(net.layers{l}, 'weights')
				% %net.layers{l}.weights{1} = net.layers{l}.weights{1} * (1 - net.layers{l}.droprate) ;
				% res(l+1).x = res(l+1).x * (1 - net.layers{l}.droprate);
			% end
			% if isfield(net.layers{l}, 'filters')
				% %net.layers{l}.filters = net.layers{l}.filters * (1 - net.layers{l}.droprate) ;
				% res(l+1).x = res(l+1).x * (1 - net.layers{l}.droprate);
			% end	
		% end
	% end
	
    % evaluate CNN
    net.layers{end}.class = labels ;
    if training, dzdy = one; else, dzdy = [] ; end
    res = vl_simplenn_dropfilter_plus(net, im, dzdy, res, ...
                      'accumulate', 0, ...
                      'disableDropout', ~training, ...
					  'disableBNorm', 0, ...
                      'conserveMemory', opts.conserveMemory, ...
                      'backPropDepth', opts.backPropDepth, ...
					  'isDropFilter', opts.isDropFilter, ...
					  'dropFilterRate', opts.dropFilterRate, ...
                      'sync', opts.sync) ;
	
    error = sum([error, [...
      sum(double(gather(res(end).x))) ;
      reshape(opts.errorFunction(opts, labels, res),[],1) ; ]],2) ;
    numDone = numDone + numel(batch) ;
  end

  % gather and accumulate gradients across labs
  if training
    if numGpus <= 1
      net = accumulate_gradients(opts, learningRate, batchSize, net, res) ;
    else
      if isempty(mmap)
        mmap = map_gradients(opts.memoryMapFile, net, res, numGpus) ;
      end
      write_gradients(mmap, net, res) ;
      labBarrier() ;
      [net,res] = accumulate_gradients(opts, learningRate, batchSize, net, res, mmap) ;
    end
	
	% if opts.isDropFilter
		% fprintf('Do drop kernel ');
		% for l=1:numel(net.layers)
			% if ~strcmp(net.layers{l}.type, 'conv'), continue ; end	
			% if isfield(net.layers{l}, 'weights')
				% net.layers{l}.weights{1} = net.layers{l}.weights{1} * (1 - net.layers{l}.droprate) ;
				% res(l+1).x = res(l+1).x * (1 - net.layers{l}.droprate);
			% end
			% if isfield(net.layers{l}, 'filters')
				% net.layers{l}.filters = net.layers{l}.filters * (1 - net.layers{l}.droprate) ;
				% res(l+1).x = res(l+1).x * (1 - net.layers{l}.droprate);
			% end	
		% end
	% end
	
  end

  % print learning statistics
  batchTime = toc(batchTime) ;
  stats = sum([stats,[batchTime ; error]],2); % works even when stats=[]
  speed = batchSize/batchTime ;

  fprintf(' %.2f s (%.1f data/s)', batchTime, speed) ;
  n = (t + batchSize - 1) / max(1,numlabs) ;
  fprintf(' obj:%.6g', stats(2)/n) ;
  for i=1:numel(opts.errorLabels)
    fprintf(' %s:%.3g', opts.errorLabels{i}, stats(i+2)/n) ;
	top_1_Error_collection(i) = top_1_Error_collection(i) + stats(i+2)/n;
  end
  fprintf(' [%d/%d]', numDone, batchSize);
  fprintf('\n') ;

  % debug info
  if opts.plotDiagnostics && numGpus <= 1
    figure(2) ; vl_simplenn_diagnose(net,res) ; drawnow ;
  end
end

top_1_Error_collection = top_1_Error_collection ./ numel(subset);
top_1_Error_collection = top_1_Error_collection .* opts.batchSize;
fprintf(' Overall top-1 Error: %.6g, overall top-5 Error: %.6g \n', top_1_Error_collection(1), top_1_Error_collection(2)) ;

if nargout > 2
  prof = mpiprofile('info');
  mpiprofile off ;
end

% -------------------------------------------------------------------------
function [net,res] = accumulate_gradients(opts, lr, batchSize, net, res, mmap)
% -------------------------------------------------------------------------
for l=1:numel(net.layers)
  for j=1:numel(res(l).dzdw)
    thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
    thisLR = lr * net.layers{l}.learningRate(j) ;

    % accumualte from multiple labs (GPUs) if needed
    if nargin >= 6
      tag = sprintf('l%d_%d',l,j) ;
      tmp = zeros(size(mmap.Data(labindex).(tag)), 'single') ;
      for g = setdiff(1:numel(mmap.Data), labindex)
        tmp = tmp + mmap.Data(g).(tag) ;
      end
      res(l).dzdw{j} = res(l).dzdw{j} + tmp ;
    end

    if isfield(net.layers{l}, 'weights')
      net.layers{l}.momentum{j} = ...
        opts.momentum * net.layers{l}.momentum{j} ...
        - thisLR * thisDecay * net.layers{l}.weights{j} ...
        - thisLR * (1 / batchSize) * res(l).dzdw{j} ;
      net.layers{l}.weights{j} = net.layers{l}.weights{j} + net.layers{l}.momentum{j} ;
    elseif isfield(net.layers{l}, 'filters')
      % Legacy code: to be removed
      if j == 1
        net.layers{l}.momentum{j} = ...
          opts.momentum * net.layers{l}.momentum{j} ...
          - thisLR * thisDecay * net.layers{l}.filters ...
          - thisLR * (1 / batchSize) * res(l).dzdw{j} ;
        net.layers{l}.filters = net.layers{l}.filters + net.layers{l}.momentum{j} ;
      else
        net.layers{l}.momentum{j} = ...
          opts.momentum * net.layers{l}.momentum{j} ...
          - thisLR * thisDecay * net.layers{l}.biases ...
          - thisLR * (1 / batchSize) * res(l).dzdw{j} ;
        net.layers{l}.biases = net.layers{l}.biases + net.layers{l}.momentum{j} ;
      end
    end
  end
  if ( strcmp(net.layers{l}.type, 'softrelu') ) 
		net.layers{l}.momentum{1} = ...
			opts.momentum_ReLU * net.layers{l}.momentum{1} ...
			- opts.learningRate_ReLU * (1 / batchSize) * res(l).dalpha ; 
		net.layers{l}.momentum{2} = ...
			opts.momentum_ReLU * net.layers{l}.momentum{1} ...
			- opts.learningRate_ReLU * (1 / batchSize) * res(l).dbeta ; 
			
		net.layers{l}.alpha = net.layers{l}.alpha + net.layers{l}.momentum{1};
		net.layers{l}.beta = net.layers{l}.beta + net.layers{l}.momentum{2};
  
		net.layers{l}.alpha = max(0.75, net.layers{l}.alpha);
		net.layers{l}.alpha = min(1.25, net.layers{l}.alpha);
		net.layers{l}.beta = max(0.01, net.layers{l}.beta);
		net.layers{l}.beta = min(0.20, net.layers{l}.beta);
  
		if (net.layers{l}.isAvg)
			[H_sr, W_sr, C_sr] = size(net.layers{l}.alpha);
			alpha_Avg = mean(net.layers{l}.alpha, 3);
			beta_Avg = mean(net.layers{l}.beta, 3);
			
			alpha_Avg = repmat(alpha_Avg, 1, 1, C_sr);
			beta_Avg = repmat(beta_Avg, 1, 1, C_sr);
			
			net.layers{l}.alpha = alpha_Avg;
			net.layers{l}.beta = beta_Avg;
		end
		
  end
end

% -------------------------------------------------------------------------
function mmap = map_gradients(fname, net, res, numGpus)
% -------------------------------------------------------------------------
format = {} ;
for i=1:numel(net.layers)
  for j=1:numel(res(i).dzdw)
    format(end+1,1:3) = {'single', size(res(i).dzdw{j}), sprintf('l%d_%d',i,j)} ;
  end
end
format(end+1,1:3) = {'double', [3 1], 'errors'} ;
if ~exist(fname) && (labindex == 1)
  f = fopen(fname,'wb') ;
  for g=1:numGpus
    for i=1:size(format,1)
      fwrite(f,zeros(format{i,2},format{i,1}),format{i,1}) ;
    end
  end
  fclose(f) ;
end
labBarrier() ;
mmap = memmapfile(fname, 'Format', format, 'Repeat', numGpus, 'Writable', true) ;

% -------------------------------------------------------------------------
function write_gradients(mmap, net, res)
% -------------------------------------------------------------------------
for i=1:numel(net.layers)
  for j=1:numel(res(i).dzdw)
    mmap.Data(labindex).(sprintf('l%d_%d',i,j)) = gather(res(i).dzdw{j}) ;
  end
end
