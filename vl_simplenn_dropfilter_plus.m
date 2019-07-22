function res = vl_simplenn(net, x, dzdy, res, varargin)
% VL_SIMPLENN  Evaluates a simple CNN
%   RES = VL_SIMPLENN(NET, X) evaluates the convnet NET on data X.
%   RES = VL_SIMPLENN(NET, X, DZDY) evaluates the convnent NET and its
%   derivative on data X and output derivative DZDY.
%
%   The network has a simple (linear) topology, i.e. the computational
%   blocks are arranged in a sequence of layers. Please note that
%   there is no need to use this wrapper, which is provided for
%   convenience. Instead, the individual CNN computational blocks can
%   be evaluated directly, making it possible to create significantly
%   more complex topologies, and in general allowing greater
%   flexibility.
%
%   The NET structure contains two fields:
%
%   - net.layers: the CNN layers.
%   - net.normalization: information on how to normalize input data.
%
%   The network expects the data X to be already normalized. This
%   usually involves rescaling the input image(s) and subtracting a
%   mean.
%
%   RES is a structure array with one element per network layer plus
%   one representing the input. So RES(1) refers to the zeroth-layer
%   (input), RES(2) refers to the first layer, etc. Each entry has
%   fields:
%
%   - res(i+1).x: the output of layer i. Hence res(1).x is the network
%     input.
%
%   - res(i+1).aux: auxiliary output data of layer i. For example,
%     dropout uses this field to store the dropout mask.
%
%   - res(i+1).dzdx: the derivative of the network output relative to
%     variable res(i+1).x, i.e. the output of layer i. In particular
%     res(1).dzdx is the derivative of the network output with respect
%     to the network input.
%
%   - res(i+1).dzdw: the derivative of the network output relative to
%     the parameters of layer i. It can be a cell array for multiple
%     parameters.
%
%   net.layers is a cell array of network layers. The following
%   layers, encapsulating corresponding functions in the toolbox, are
%   supported:
%
%   Convolutional layer::
%     The convolutional layer wraps VL_NNCONV(). It has fields:
%
%     - layer.type = 'conv'
%     - layer.weights = {filters, biases}
%     - layer.stride: the sampling stride (usually 1).
%     - layer.pad: the padding (usually 0).
%
%   Convolution transpose layer::
%     The convolution transpose layer wraps VL_NNCONVT(). It has fields:
%
%     - layer.type = 'convt'
%     - layer.weights = {filters, biases}
%     - layer.upsample: the upsampling factor.
%     - layer.crop: the amount of output cropping.
%
%   Max pooling layer::
%     The max pooling layer wraps VL_NNPOOL(). It has fields:
%
%     - layer.type = 'pool'
%     - layer.method: pooling method ('max' or 'avg').
%     - layer.pool: the pooling size.
%     - layer.stride: the sampling stride (usually 1).
%     - layer.pad: the padding (usually 0).
%
%   Normalization layer::
%     The normalization layer wraps VL_NNNORMALIZE(). It has fields
%
%     - layer.type = 'normalize'
%     - layer.param: the normalization parameters.
%
%   Spatial normalization layer:
%     This is similar to the layer above, but wraps VL_NNSPNORM():
%
%     - layer.type = 'spnorm'
%     - layer.param: the normalization parameters.
%
%   Batch normalization layer:
%     This layer wraps VL_NNBNORM(). It has fields:
%
%     - layer.type = 'bnorm'
%     - layer.weights = {multipliers, biases}.
%
%   ReLU and Sigmoid layers::
%     The ReLU layer wraps VL_NNRELU(). It has fields:
%
%     - layer.type = 'relu'
%
%     The sigmoid layer is the same, but for the sigmoid function, with
%     `relu` replaced by `sigmoid`.
%
%   Dropout layer::
%     The dropout layer wraps VL_NNDROPOUT(). It has fields:
%
%     - layer.type = 'dropout'
%     - layer.rate: the dropout rate.
%
%   Softmax layer::
%     The softmax layer wraps VL_NNSOFTMAX(). It has fields
%
%     - layer.type = 'softmax'
%
%   Log-loss layer::
%     The log-loss layer wraps VL_NNLOSS(). It has fields:
%
%     - layer.type = 'loss'
%     - layer.class: the ground-truth class.
%
%   Softmax-log-loss layer::
%     The softmax-log-loss layer wraps VL_NNSOFTMAXLOSS(). It has
%     fields:
%
%     - layer.type = 'softmaxloss'
%     - layer.class: the ground-truth class.
%
%   P-dist layer:
%     The pdist layer wraps VL_NNPDIST(). It has fields:
%
%     - layer.type = 'pdist'
%     - layer.p = P parameter of the P-distance
%     - layer.noRoot = whether to raise the distance to the P-th power
%     - layer.epsilon = regularization parameter for the derivatives
%
%   Custom layer::
%     This can be used to specify custom layers.
%
%     - layer.type = 'custom'
%     - layer.forward: a function handle computing the block.
%     - layer.backward: a function handle computing the block derivative.
%
%     The first function is called as res(i+1) = forward(layer, res(i), res(i+1))
%     where res() is the struct array specified before. The second function is
%     called as res(i) = backward(layer, res(i), res(i+1)). Note that the
%     `layer` structure can contain additional fields if needed.

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.res = [] ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.disableDropout = false ;
opts.disableBNorm = false ;
opts.freezeDropout = false ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.isDropFilter = 1;
opts.dropFilterRate = 0.5;
opts.backPropDepth = +inf ;
% opts.nofSamples = 200;
% opts.isresconv = 1;

meanValue = 1 - opts.dropFilterRate;
minValue  = 0.1;
maxValue  = 1;

opts = vl_argparse(opts, varargin);

n = numel(net.layers) ;

if (nargin <= 2) || isempty(dzdy)
  doder = false ;
else
  doder = true ;
end

if opts.cudnn
  cudnn = {'CuDNN'} ;
else
  cudnn = {'NoCuDNN'} ;
end

gpuMode = isa(x, 'gpuArray') ;

if nargin <= 3 || isempty(res)
  res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ...
    'dzdw', cell(1,n+1), ...
    'aux', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1))) ;
end
res(1).x = x ;
clear x;

for i=1:n
  l = net.layers{i} ;
  res(i).time = tic ;
  switch l.type
	case 'conv'
      if isfield(l, 'weights')
        res(i+1).x = vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                               'pad', l.pad, 'stride', l.stride, ...
                               cudnn{:}) ;
      else
        res(i+1).x = vl_nnconv(res(i).x, l.filters, l.biases, ...
                               'pad', l.pad, 'stride', l.stride, ...
                               cudnn{:}) ;
      end
  
  
    case 'conv_dropfilter'
      if isfield(l, 'weights')
	  
        res(i+1).x = vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                               'pad', l.pad, 'stride', l.stride, ...
                               cudnn{:}) ;
		if (opts.isDropFilter)&&(doder)	
			
			% 20190528 new idea of dropfilter plus
			% 1. get the average of weights (perhaps not needed)
			% meanW = mean( mean(l.weights{1}, 1), 2 )
			% 2. get the mean and std of res(i).x (every location or overall?)
			% 20190603: firstly try overall mean and std
			% input1D = reshape(res(i).x, 1, []);
			inputMean = mean(res(i).x(:));
			inputStd  = std(res(i).x(:));
			
			% 3. get the mean and std of F (dropfilter distribution)
			% weights1D = reshape(l.weights{1}, 1, []);
			weightsMean = mean(l.weights{1}(:));
			weightStd   = std(l.weights{1}(:));
			
			
			% 4. calculate the mean and std of a * W
			meanMulti = (inputMean * (weightStd^2) + weightsMean * (inputStd^2))/( weightStd^2 + inputStd^2 + 0.0001 );
			stdMulti  = sqrt( (inputStd^2 * weightStd^2)/( weightStd^2 + inputStd^2 + 0.0001 ) );
			
			% 5. Sample from the two distributions by using random number generator
			% 5.1. Generate the 0-1 rand value
			% nofSamples = size(res(i).x, 1) * size(res(i).x, 2);
			
			% can be moved out 
			% outSizeX   = size(res(i+1).x, 1);
			% outSizeY   = size(res(i+1).x, 2);
			% outSizeZ   = size(res(i+1).x, 3);
			
			% rand0_1 = rand(l.outSizeX, l.outSizeY, l.outSizeZ, l.nofSamples);
			% rand0_1_mask = (rand0_1 < opts.dropFilterRate); % 20190605 only one droprate
						
			% denominator
			% percentLower =  (meanMulti + stdMulti * randn(l.outSizeX, l.outSizeY, l.outSizeZ, l.nofSamples));
			percentLower =  (meanMulti + stdMulti * l.rand0_1n);
			% numerator
			% percentUpper = sum( (percentLower .* l.rand0_1_mask ), 4 );
			percentUpper = (percentLower .* l.rand0_1_mask );
			% percentLower = sum( percentLower, 4);
			
			% 6. Get the range of the percetage 
			% scaleRate = squeeze( percentUpper ./ percentLower );
			% scaleRate = gpuArray(scaleRate);
			scaleRate =  percentUpper ./ percentLower;
			res(i+1).x = bsxfun(@times, res(i+1).x, scaleRate);
			
			res(i).scaleRate_dzdx = scaleRate;
		else 
			res(i).scaleRate_dzdx = 1;
		end
		
      else
	  
        res(i+1).x = vl_nnconv(res(i).x, l.filters, l.biases, ...
                               'pad', l.pad, 'stride', l.stride, ...
                               cudnn{:}) ;
							   
      end
	case 'conv_sal'
      if isfield(l, 'weights')
        tmp = vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                               'pad', l.pad, 'stride', l.stride, ...
                               cudnn{:}) ;
      else
        tmp = vl_nnconv(res(i).x, l.filters, l.biases, ...
                               'pad', l.pad, 'stride', l.stride, ...
                               cudnn{:}) ;
      end
	  tmpMax = max(tmp, [], 1);   % res(i+1).x
	  maxElem = max(tmpMax, [], 2);
	  tmpMin = min(tmp, [], 1);
	  minElem = min(tmpMin, [], 2);
	  n1 = bsxfun(@minus, tmp, minElem);
	  n2 = bsxfun(@minus, maxElem, minElem);
	  res(i+1).x = bsxfun(@rdivide, n1, n2);
	  
	case 'hope_fast'
	  if isfield(l, 'weights')
        res(i+1).x = vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                               'pad', l.pad, 'stride', l.stride, ...
                               cudnn{:}) ;
      else
        res(i+1).x = vl_nnconv(res(i).x, l.filters, l.biases, ...
                               'pad', l.pad, 'stride', l.stride, ...
                               cudnn{:}) ;
      end
	case 'hope_fast_unsupervised'
	  if isfield(l, 'weights')
        tmpx = vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                               'pad', l.pad, 'stride', l.stride, ...
                               cudnn{:}) ;
      else
        tmpx = vl_nnconv(res(i).x, l.filters, l.biases, ...
                               'pad', l.pad, 'stride', l.stride, ...
                               cudnn{:}) ;
	  end
	  % Error? should not sum over all L2_Z. Should One by One
	  % L2_Z = sqrt( sum(sum(sum(tmpx.^2,1), 2), 3) );
	  L2_Z = sqrt( sum(tmpx.^2, 3) ); % Only sum over the depth
	  % size(L2_Z)
      L2_Z(L2_Z==0) = 10; % floor L2 norm to avoid the NAN
	  % Z norm -- Eq.(30) in [1]
      res(i+1).x = bsxfun(@rdivide, tmpx, L2_Z);
	  res(i+1).L2_Z = L2_Z;
	case 'hope_conv'
	%finshed the two step in the hope
	  hope_out_forward = vl_nnhope_conv(res(i).x, l.infilterSize, l.outfilterSize, l.filters, l.filters2, l.biases, l.beta);
	  res(i+1).x = hope_out_forward{1};
	  res(i).medout = hope_out_forward{2};
	case 'hope'
	%only the first step of the hope
	  res(i+1).x = vl_nnhope(res(i).x, l.infilterSize, l.outfilterSize, l.filters, l.biases, l.beta);
	case 'hopeip'
	  if isfield(l, 'weights')
		res(i+1).x = vl_nnhopeip(res(i).x, l.weights{1}, l.weights{2}, l.beta) ;
	  else
		res(i+1).x = vl_nnhopeip(res(i).x, l.filters, l.biases, l.beta);
	  end
    case 'convt'
      if isfield(l, 'weights')
        res(i+1).x = vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, ...
                               'crop', l.crop, 'upsample', l.upsample, ...
                               cudnn{:}) ;
      else
        res(i+1).x = vl_nnconv(res(i).x, l.filters, l.biases, ...
                               'crop', l.pad, 'upsample', l.upsample, ...
                               cudnn{:}) ;
      end
    case 'pool'
      res(i+1).x = vl_nnpool(res(i).x, l.pool, ...
                             'pad', l.pad, 'stride', l.stride, ...
                             'method', l.method, ...
                             cudnn{:}) ;
    case 'normalize'
      res(i+1).x = vl_nnnormalize(res(i).x, l.param) ;
    case 'softmax'
      res(i+1).x = vl_nnsoftmax(res(i).x) ;
    case 'loss'
      res(i+1).x = vl_nnloss(res(i).x, l.class) ;
    case 'softmaxloss'
      res(i+1).x = vl_nnsoftmaxloss(res(i).x, l.class) ;
	case 'sal_loss'
	  res(i+1).x = vl_nnsalerror(res(i).x, l.classes) ; 
    case 'relu'
      res(i+1).x = vl_nnrelu(res(i).x) ;
	  
	case 'res_acc'
	  % A. zeros_padding (this is finally used in the paper)
	  [H, W, C, N] = size(res(i).x);
	  [H_s, W_s, C_s, N_s] = size( res( i-(l.shortcut_depth) ).x );

	  if (C_s >= C)
		sc_input = res( i-(l.shortcut_depth) ).x;
	  else
		sc_input = zeros(H, W, C, N);
		sc_input = single(sc_input);
		sc_input = gpuArray(sc_input);
		% fprintf('\n');
		%fprintf(' H_s = %d, W_s = %d, C_s = %d, N_s = %d \n', H_s,W_s, C_s, N_s);
		%fprintf(' H = %d, W = %d, C = %d, N = %d \n', H, W, C, N);
		sc_input(:, :, 1:C_s, :) = res( i-(l.shortcut_depth) ).x;
	  end
	  % size(res(i).x)
	  % size(res( i-(l.shortcut_depth) ).x)
	  % size(sc_input)
	  % i
      % res(i+1).x = vl_nnrelu( (res(i).x + sc_input) ) ;
	  res(i+1).x = res(i).x + sc_input;
	  
	case 'res_acc_conv'
	  % A. zeros_padding (this is finally used in the paper)
	  % [H, W, C, N] = size(res(i).x);
	  % [H_s, W_s, C_s, N_s] = size( res( i-(l.shortcut_depth) ).x );
	  % save res.mat res;
	  % size(res(10).x)
	  
	  sc_input = vl_nnconv(res( i-(l.shortcut_depth) ).x, ...
							l.weights{1}, l.weights{2}, ...
                            'pad', l.pad, 'stride', l.stride, ...
                            cudnn{:}) ;
      % res(i+1).x = vl_nnrelu( (res(i).x + sc_input) ) ;
	  res(i+1).x = res(i).x + sc_input;
	
	case 'softrelu'
	  [res(i+1).x, tmp1, tmp2 ] = vl_nnsoftrelu(res(i).x, l.alpha, l.beta, l.isAvg) ;
    case 'sigmoid'
      res(i+1).x = vl_nnsigmoid(res(i).x) ;
    case 'noffset'
      res(i+1).x = vl_nnnoffset(res(i).x, l.param) ;
    case 'spnorm'
      res(i+1).x = vl_nnspnorm(res(i).x, l.param) ;
    case 'dropout'
      if opts.disableDropout
        res(i+1).x = res(i).x ;
      elseif opts.freezeDropout
        [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate, 'mask', res(i+1).aux) ;
      else
        [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate) ;
      end
    case 'bnorm'
	  if opts.disableBNorm
		res(i+1).x = res(i).x ;
	  else
		if isfield(l, 'weights')
			res(i+1).x = vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}) ;
		else
			res(i+1).x = vl_nnbnorm(res(i).x, l.filters, l.biases) ;
		end
	  end
    case 'pdist'
      res(i+1) = vl_nnpdist(res(i).x, l.p, 'noRoot', l.noRoot, 'epsilon', l.epsilon) ;
    case 'custom'
      res(i+1) = l.forward(l, res(i), res(i+1)) ;
    otherwise
      error('Unknown layer type %s', l.type) ;
  end
  % optionally forget intermediate results
  forget = opts.conserveMemory ;
  forget = forget & (~doder || strcmp(l.type, 'relu')) ;
  forget = forget & ~(strcmp(l.type, 'loss') || strcmp(l.type, 'softmaxloss')) ;
  forget = forget & (~isfield(l, 'rememberOutput') || ~l.rememberOutput) ;
  forget = forget & ~(strcmp(l.type, 'res_acc')) ;
  forget = forget & ~(strcmp(l.type, 'res_acc_conv')) ;
  forget = forget & ~(strcmp(l.type, 'bnorm')) ;
  forget = forget & (i < n-2);
  if forget
    res(i).x = [] ;
  end
  if gpuMode & opts.sync
    % This should make things slower, but on MATLAB 2014a it is necessary
    % for any decent performance.
    wait(gpuDevice) ;
  end
  res(i).time = toc(res(i).time) ;
end

if doder
  res(n+1).dzdx = dzdy ;
  for i=n:-1:max(1, n-opts.backPropDepth+1)
    l = net.layers{i} ;
    res(i).backwardTime = tic ;
    switch l.type
	
	  case 'conv'
        if ~opts.accumulate
          if isfield(l, 'weights')
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                          res(i+1).dzdx, ...
                          'pad', l.pad, 'stride', l.stride, ...
                          cudnn{:}) ;
          else
            % Legacy code: will go
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnconv(res(i).x, l.filters, l.biases, ...
                          res(i+1).dzdx, ...
                          'pad', l.pad, 'stride', l.stride, ...
                          cudnn{:}) ;
          end
        else
          dzdw = cell(1,2) ;
          if isfield(l, 'weights')
            [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                          res(i+1).dzdx, ...
                          'pad', l.pad, 'stride', l.stride, ...
                          cudnn{:}) ;
          else
            % Legacy code: will go
            [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                vl_nnconv(res(i).x, l.filters, l.biases, ...
                          res(i+1).dzdx, ...
                          'pad', l.pad, 'stride', l.stride, ...
                          cudnn{:}) ;
          end
          for j=1:2
            res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
          end
          clear dzdw ;
        end
	
	
      case 'conv_dropfilter'
        if ~opts.accumulate
          if isfield(l, 'weights')
			res(i+1).dzdx = bsxfun(@times, res(i+1).dzdx, res(i).scaleRate_dzdx);
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                          res(i+1).dzdx, ...
                          'pad', l.pad, 'stride', l.stride, ...
                          cudnn{:}) ;
          else
            % Legacy code: will go
			res(i+1).dzdx = bsxfun(@times, res(i+1).dzdx, res(i).scaleRate_dzdx);
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnconv(res(i).x, l.filters, l.biases, ...
                          res(i+1).dzdx, ...
                          'pad', l.pad, 'stride', l.stride, ...
                          cudnn{:}) ;
          end
		  % if (opts.isDropFilter)
			% res(i).dzdw{1} = bsxfun(@times, res(i).dzdw{1}, net.layers{i}.scaleRate_dzdw);
		  % end
        else
          dzdw = cell(1,2) ;
          if isfield(l, 'weights')
			res(i+1).dzdx = bsxfun(@times, res(i+1).dzdx, res(i).scaleRate_dzdx);
            [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                          res(i+1).dzdx, ...
                          'pad', l.pad, 'stride', l.stride, ...
                          cudnn{:}) ;
          else
            % Legacy code: will go
			res(i+1).dzdx = bsxfun(@times, res(i+1).dzdx, res(i).scaleRate_dzdx);
            [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                vl_nnconv(res(i).x, l.filters, l.biases, ...
                          res(i+1).dzdx, ...
                          'pad', l.pad, 'stride', l.stride, ...
                          cudnn{:}) ;
          end
		  % if (opts.isDropFilter)
			% dzdw{1} = bsxfun(@times, dzdw{1}, net.layers{i}.scaleRate_dzdw);
		  % end
          for j=1:2
            res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
          end
          clear dzdw ;
        end
		
	  case 'conv_sal'
        if ~opts.accumulate
          if isfield(l, 'weights')
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                          res(i+1).dzdx, ...
                          'pad', l.pad, 'stride', l.stride, ...
                          cudnn{:}) ;
          else
            % Legacy code: will go
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnconv(res(i).x, l.filters, l.biases, ...
                          res(i+1).dzdx, ...
                          'pad', l.pad, 'stride', l.stride, ...
                          cudnn{:}) ;
          end
        else
          dzdw = cell(1,2) ;
          if isfield(l, 'weights')
            [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                          res(i+1).dzdx, ...
                          'pad', l.pad, 'stride', l.stride, ...
                          cudnn{:}) ;
          else
            % Legacy code: will go
            [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                vl_nnconv(res(i).x, l.filters, l.biases, ...
                          res(i+1).dzdx, ...
                          'pad', l.pad, 'stride', l.stride, ...
                          cudnn{:}) ;
          end
          for j=1:2
            res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
          end
          clear dzdw ;
        end
		
	  case 'hope_fast'
        if isfield(l, 'weights')
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                          res(i+1).dzdx, ...
                          'pad', l.pad, 'stride', l.stride, ...
                          cudnn{:}) ;
          else
            % Legacy code: will go
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnconv(res(i).x, l.filters, l.biases, ...
                          res(i+1).dzdx, ...
                          'pad', l.pad, 'stride', l.stride, ...
                          cudnn{:}) ;
          end  
		%[H, W, C_in, C_out] = size(res(i).dzdw{1});
		% L2_W = sqrt(sum(l.weights{1}.^2, 1)); %[1, W, C_in, C_out]
		% % for i2 = 1:C_in
			% % for j2 = 1:C_out
				% % L2_W_Square(:, :, i2, j2) = transpose(L2_W(:, :, i2, j2))*L2_W(:, :, i2, j2);
			% % end
		% % end
		% %df_D = zeros(H, W, C_in, C_out, 'single');
		% %df_D = gpuArray(df_D);
		% for i2 = 1:C_in
			% for j2 = 1:C_out
				% weight = l.weights{1}(:, :, i2, j2);% can be optimized
				% % L2_W_S = L2_W_Square(:, :, i2, j2);
				% L2_W_sub = L2_W(:, :, i2, j2);
				% L2_W_S = transpose(L2_W_sub)*L2_W_sub;
				% C_matrix = transpose(weight) * weight;
				% G_matrix = C_matrix./L2_W_S;
				% B = sign(C_matrix)./L2_W_S;
				% % df_D(:, :, i2, j2) = weight*B - (weight./repmat(L2_W_sub.^2,[size(weight,1),1]))* diag(sum(G_matrix,1));
				% l.df_D(:, :, i2, j2) = weight*B - (bsxfun(@rdivide, weight, L2_W_sub))* diag(sum(G_matrix,1));
			% end
		% end
		% res(i).dzdw{1} = res(i).dzdw{1} + l.beta .* l.df_D;
	  case 'hope_fast_unsupervised'
		if isfield(l, 'weights')
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                          res(i+1).dzdx, ...
                          'pad', l.pad, 'stride', l.stride, ...
                          cudnn{:}) ;
          else
            % Legacy code: will go
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnconv(res(i).x, l.filters, l.biases, ...
                          res(i+1).dzdx, ...
                          'pad', l.pad, 'stride', l.stride, ...
                          cudnn{:}) ;
          end  
	  case 'hope_conv'
		%[res(i).dzdx, res(i).dzdw] = vl_nnhope(res(i).x, l.infilterSize, l.outfilterSize, l.weights, l.biases, l.beta, res(i+1).dzdx);
		hope_out = vl_nnhope_conv(res(i).x, l.infilterSize, l.outfilterSize, l.filters, l.filters2, l.biases, l.beta, res(i).medout, res(i+1).dzdx);
		res(i).dzdx = hope_out{1};
		res(i).dzdw = hope_out{2};
		res(i).dzdw2 = hope_out{3};
		clear hope_out;
		
	  case 'hope'
		%[res(i).dzdx, res(i).dzdw] = vl_nnhope(res(i).x, l.infilterSize, l.outfilterSize, l.weights, l.biases, l.beta, res(i+1).dzdx);
		hope_out = vl_nnhope(res(i).x, l.infilterSize, l.outfilterSize, l.filters, l.biases, l.beta, res(i+1).dzdx);
		res(i).dzdx = hope_out{1};
		res(i).dzdw = hope_out{2};
		clear hope_out;
		
	  case 'hopeip'
		
		if isfield(l, 'weights')
            hope_out = vl_nnhopeip(res(i).x, l.weights{1}, l.weights{2}, l.beta, res(i+1).dzdx);
			[H2, W2] = size(l.weights{2});
        else
            hope_out = vl_nnhopeip(res(i).x, l.filters, l.biases, l.beta, res(i+1).dzdx);
			[H2, W2] = size(l.biases);
        end
		res(i).dzdx = hope_out{1};
		res(i).dzdw{1} = hope_out{2};
		res(i).dzdw{2} = zeros(H2, W2, 'single');
		clear hope_out;

      case 'convt'
        if ~opts.accumulate
          if isfield(l, 'weights')
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, ...
                          res(i+1).dzdx, ...
                          'crop', l.crop, 'upsample', l.upsample, ...
                          cudnn{:}) ;
          else
            % Legacy code: will go
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnconvt(res(i).x, l.filters, l.biases, ...
                         res(i+1).dzdx, ...
                          'crop', l.crop, 'upsample', l.upsample, ...
                          cudnn{:}) ;
          end
        else
          dzdw = cell(1,2) ;
          if isfield(l, 'weights')
            [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, ...
                          res(i+1).dzdx, ...
                          'crop', l.crop, 'upsample', l.upsample, ...
                          cudnn{:}) ;
          else
            % Legacy code: will go
            [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                vl_nnconvt(res(i).x, l.filters, l.biases, ...
                          res(i+1).dzdx, ...
                          'crop', l.crop, 'upsample', l.upsample, ...
                          cudnn{:}) ;
          end
          for j=1:2
            res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
          end
          clear dzdw ;
        end
       
      case 'pool'
        res(i).dzdx = vl_nnpool(res(i).x, l.pool, res(i+1).dzdx, ...
                                'pad', l.pad, 'stride', l.stride, ...
                                'method', l.method, ...
                                cudnn{:}) ;
      case 'normalize'
        res(i).dzdx = vl_nnnormalize(res(i).x, l.param, res(i+1).dzdx) ;
      case 'softmax'
        res(i).dzdx = vl_nnsoftmax(res(i).x, res(i+1).dzdx) ;
      case 'loss'
        res(i).dzdx = vl_nnloss(res(i).x, l.class, res(i+1).dzdx) ;
      case 'softmaxloss'
        res(i).dzdx = vl_nnsoftmaxloss(res(i).x, l.class, res(i+1).dzdx) ;
	  case 'sal_loss'
        res(i).dzdx = vl_nnsalerror(res(i).x, l.class, res(i+1).dzdx) ;
      case 'relu'
        if ~isempty(res(i).x)
          res(i).dzdx = vl_nnrelu(res(i).x, res(i+1).dzdx) ;
        else
          % if res(i).x is empty, it has been optimized away, so we use this
          % hack (which works only for ReLU):
          res(i).dzdx = vl_nnrelu(res(i+1).x, res(i+1).dzdx) ;
        end
		
	  case 'res_acc'
		  res(i).dzdx = res(i+1).dzdx ;
		
	  case 'res_acc_conv'
		  %size(res( i-(l.shortcut_depth) ).x)
		  % size(l.weights{1})
		  [~, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnconv(res( i-(l.shortcut_depth) ).x, ...
						  l.weights{1}, l.weights{2}, ...
                          res(i+1).dzdx, ...
                          'pad', l.pad, 'stride', l.stride, ...
                          cudnn{:}) ;
		  res(i).dzdx = res(i+1).dzdx ;
		
	  case 'softrelu'
		if ~isempty(res(i).x)
		  [res(i).dzdx, res(i).dalpha, res(i).dbeta ] = vl_nnsoftrelu(res(i).x, l.alpha, l.beta, l.isAvg, res(i+1).dzdx) ;
		else
		  [res(i).dzdx, res(i).dalpha, res(i).dbeta ] = vl_nnsoftrelu(res(i+1).x, l.alpha, l.beta, l.isAvg, res(i+1).dzdx) ;
		end
      case 'sigmoid'
        res(i).dzdx = vl_nnsigmoid(res(i).x, res(i+1).dzdx) ;
      case 'noffset'
        res(i).dzdx = vl_nnnoffset(res(i).x, l.param, res(i+1).dzdx) ;
      case 'spnorm'
        res(i).dzdx = vl_nnspnorm(res(i).x, l.param, res(i+1).dzdx) ;
      case 'dropout'
        if opts.disableDropout
          res(i).dzdx = res(i+1).dzdx ;
        else
          res(i).dzdx = vl_nndropout(res(i).x, res(i+1).dzdx, ...
                                     'mask', res(i+1).aux) ;
        end
      case 'bnorm'
		if opts.disableBNorm
		  res(i).dzdx = res(i+1).dzdx ;
		else
		  if ~opts.accumulate
			if isfield(l, 'weights')
				[res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
					vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, ...
							res(i+1).dzdx) ;
			else
				[res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
					vl_nnbnorm(res(i).x, l.filters, l.biases, ...
							res(i+1).dzdx) ;
			end
		  else
			dzdw = cell(1,2) ;
			if isfield(l, 'weights')
				[res(i).dzdx, dzdw{1}, dzdw{2}] = ...
					vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, ...
							res(i+1).dzdx) ;
			else
				[res(i).dzdx, dzdw{1}, dzdw{2}] = ...
					vl_nnbnorm(res(i).x, l.filters, l.biases, ...
							res(i+1).dzdx) ;
			end
			for j=1:2
				res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
			end
			clear dzdw ;
		  end
		end
      case 'pdist'
        res(i).dzdx = vl_nnpdist(res(i).x, l.p, res(i+1).dzdx, ...
                                 'noRoot', l.noRoot, 'epsilon', l.epsilon) ;
      case 'custom'
        res(i) = l.backward(l, res(i), res(i+1)) ;
    end
    if opts.conserveMemory && (i ~= 1)
      res(i+1).dzdx = [] ;
    end
    if gpuMode & opts.sync
      wait(gpuDevice) ;
    end
    res(i).backwardTime = toc(res(i).backwardTime) ;
  end
end
