function net = cnn_cifar_init_2016(opts)

lr = [1 1] ;
%weightDecay
wdr = [1 0];
lr_bnorm = [1 1]
wdr_bnorm = [0, 0]
epsilon = 1e-3

% Define network CIFAR10-Hinton's dropout
fprintf('CNN dropfilter plus 20190605 new \n');
net.layers = {} ;

% net.layers{end+1} = struct('type', 'conv', ...
						   % 'weights', {{single(normrnd(0, 0.0001, 5, 5, 3, 24)), zeros(1, 24, 'single')}}, ...
						   % 'learningRate', lr, ...
						   % 'stride', 1, ...
						   % 'pad', 2, ...
						   % 'droprate', 0.15) ;
% net.layers{end+1} = struct('type', 'relu') ;
% net.layers{end+1} = struct('type', 'dropout', 'rate', 0.15) ;

% ConvBNReLU(3,64):add(nn.Dropout(0.3))
% ConvBNReLU(64,64)
% vgg:add(MaxPooling(2,2,2,2):ceil())

% ConvBNReLU(64,128):add(nn.Dropout(0.4))
% ConvBNReLU(128,128)
% vgg:add(MaxPooling(2,2,2,2):ceil())

% ConvBNReLU(128,256):add(nn.Dropout(0.4))
% ConvBNReLU(256,256):add(nn.Dropout(0.4))
% ConvBNReLU(256,256)
% vgg:add(MaxPooling(2,2,2,2):ceil())

% ConvBNReLU(256,512):add(nn.Dropout(0.4))
% ConvBNReLU(512,512):add(nn.Dropout(0.4))
% ConvBNReLU(512,512)
% vgg:add(MaxPooling(2,2,2,2):ceil())

% ConvBNReLU(512,512):add(nn.Dropout(0.4))
% ConvBNReLU(512,512):add(nn.Dropout(0.4))
% ConvBNReLU(512,512)
% vgg:add(MaxPooling(2,2,2,2):ceil())

% vgg:add(nn.View(512))
% vgg:add(nn.Dropout(0.5))
% vgg:add(nn.Linear(512,512))
% vgg:add(nn.BatchNormalization(512))
% vgg:add(nn.ReLU(true))
% vgg:add(nn.Dropout(0.5))
% vgg:add(nn.Linear(512,10))

% Block 0
% net.layers{end+1} = struct('type', 'conv', ...
						   % 'weights', {{single(normrnd(0, sqrt(2/(3*3*20)), 3, 3, 3, 20)), zeros(1, 20, 'single')}}, ...
						   % 'learningRate', lr, ...
						   % 'stride', 1, ...
						   % 'pad', 1, ...
						   % 'droprate', 0) ;%sqrt(2/(5*5*3))

% input data: 32*32*3*batchsize

% Block 1
net.layers{end+1} = struct('type', 'conv_dropfilter', ...
                           'weights', {{single(normrnd(0, sqrt(2/(3*3*64)), 3, 3, 3, 64)), zeros(1, 64, 'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 1, ...
						   'droprate', 0.3, ...
						   'nofSamples', 32*32, ...
						   'outSizeX', 32, ...
						   'outSizeY', 32, ...
						   'outSizeZ', 64) ;
						   
% add more entries at definition
% nofSamples = size(res(i).x, 1) * size(res(i).x, 2);
% outSizeX   = size(res(i+1).x, 1);
% outSizeY   = size(res(i+1).x, 2);
% outSizeZ   = size(res(i+1).x, 3);
% rand0_1 = rand(outSizeX, outSizeY, outSizeZ, nofSamples);
% rand0_1_mask = (rand0_1 < opts.dropFilterRate); 	

% net.layers{end+1} = struct('type', 'conv', ...
                           % 'weights', {{single(normrnd(0, sqrt(2/(3*3*64)), 3, 3, 3, 64)), zeros(1, 64, 'single')}}, ...
                           % 'learningRate', lr, ...
                           % 'stride', 1, ...
                           % 'pad', 1, ...
						   % 'droprate', 0) ;				   
						   
net.layers{end+1} = struct('type', 'bnorm', ...
                           'weights', {{ones(64, 1, 'single'), zeros(64, 1, 'single')}}, ...
                           'learningRate', lr_bnorm, ...
                           'weightDecay', wdr_bnorm, ...
						   'Epsilon', epsilon) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'dropout', 'rate', 0.3) ; 

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{single(normrnd(0, sqrt(2/(3*3*64)), 3, 3, 64, 64)), zeros(1, 64, 'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 1, ...
						   'droprate', 0) ;
net.layers{end+1} = struct('type', 'bnorm', ...
                           'weights', {{ones(64, 1, 'single'), zeros(64, 1, 'single')}}, ...
                           'learningRate', lr_bnorm, ...
                           'weightDecay', wdr_bnorm, ...
						   'Epsilon', epsilon) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
% net.layers{end+1} = struct('type', 'conv', ...
                           % 'weights', {{single(normrnd(0, sqrt(2/(2*2*64)), 2, 2, 64, 64)), zeros(1, 64, 'single')}}, ...
                           % 'learningRate', lr, ...
                           % 'stride', 2, ...
                           % 'pad', 0, ...
						   % 'droprate', 0) ;
% Block 2
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{single(normrnd(0, sqrt(2/(3*3*128)), 3, 3, 64, 128)), zeros(1,128,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 1, ...
						   'droprate', 0.40) ;
net.layers{end+1} = struct('type', 'bnorm', ...
                           'weights', {{ones(128, 1, 'single'), zeros(128, 1, 'single')}}, ...
                           'learningRate', lr_bnorm, ...
                           'weightDecay', wdr_bnorm, ...
						   'Epsilon', epsilon) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'dropout', 'rate', 0.40) ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{single(normrnd(0, sqrt(2/(3*3*128)), 3, 3, 128, 128)), zeros(1,128,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 1, ...
						   'droprate', 0) ;
net.layers{end+1} = struct('type', 'bnorm', ...
                           'weights', {{ones(128, 1, 'single'), zeros(128, 1, 'single')}}, ...
                           'learningRate', lr_bnorm, ...
                           'weightDecay', wdr_bnorm, ...
						   'Epsilon', epsilon) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ; % Emulate caffe


% Block 3
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{single(normrnd(0, sqrt(2/(3*3*256)), 3, 3, 128, 256)), zeros(1,256,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 1, ...
						   'droprate', 0.4) ;
net.layers{end+1} = struct('type', 'bnorm', ...
                           'weights', {{ones(256, 1, 'single'), zeros(256, 1, 'single')}}, ...
                           'learningRate', lr_bnorm, ...
                           'weightDecay', wdr_bnorm, ...
						   'Epsilon', epsilon) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'dropout', 'rate', 0.4) ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{single(normrnd(0, sqrt(2/(3*3*256)), 3, 3, 256, 256)), zeros(1,256,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 1, ...
						   'droprate', 0.4) ;
net.layers{end+1} = struct('type', 'bnorm', ...
                           'weights', {{ones(256, 1, 'single'), zeros(256, 1, 'single')}}, ...
                           'learningRate', lr_bnorm, ...
                           'weightDecay', wdr_bnorm, ...
						   'Epsilon', epsilon) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'dropout', 'rate', 0.4) ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{single(normrnd(0, sqrt(2/(3*3*256)), 3, 3, 256, 256)), zeros(1,256,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 1, ...
						   'droprate', 0) ;
net.layers{end+1} = struct('type', 'bnorm', ...
                           'weights', {{ones(256, 1, 'single'), zeros(256, 1, 'single')}}, ...
                           'learningRate', lr_bnorm, ...
                           'weightDecay', wdr_bnorm, ...
						   'Epsilon', epsilon) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ; % Emulate caffe


% Block 4
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{single(normrnd(0, sqrt(2/(3*3*512)), 3, 3, 256, 512)), zeros(1,512,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 1, ...
						   'droprate', 0.4) ;
net.layers{end+1} = struct('type', 'bnorm', ...
                           'weights', {{ones(512, 1, 'single'), zeros(512, 1, 'single')}}, ...
                           'learningRate', lr_bnorm, ...
                           'weightDecay', wdr_bnorm, ...
						   'Epsilon', epsilon) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'dropout', 'rate', 0.4) ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{single(normrnd(0, sqrt(2/(3*3*512)), 3, 3, 512, 512)), zeros(1,512,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 1, ...
						   'droprate', 0.4) ;
net.layers{end+1} = struct('type', 'bnorm', ...
                           'weights', {{ones(512, 1, 'single'), zeros(512, 1, 'single')}}, ...
                           'learningRate', lr_bnorm, ...
                           'weightDecay', wdr_bnorm, ...
						   'Epsilon', epsilon) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'dropout', 'rate', 0.4) ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{single(normrnd(0, sqrt(2/(3*3*512)), 3, 3, 512, 512)), zeros(1,512,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 1, ...
						   'droprate', 0) ;
net.layers{end+1} = struct('type', 'bnorm', ...
                           'weights', {{ones(512, 1, 'single'), zeros(512, 1, 'single')}}, ...
                           'learningRate', lr_bnorm, ...
                           'weightDecay', wdr_bnorm, ...
						   'Epsilon', epsilon) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;

% Block 5
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{single(normrnd(0, sqrt(2/(3*3*512)), 3, 3, 512, 512)), zeros(1,512,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 1, ...
						   'droprate', 0.4) ;
net.layers{end+1} = struct('type', 'bnorm', ...
                           'weights', {{ones(512, 1, 'single'), zeros(512, 1, 'single')}}, ...
                           'learningRate', lr_bnorm, ...
                           'weightDecay', wdr_bnorm, ...
						   'Epsilon', epsilon) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'dropout', 'rate', 0.4) ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{single(normrnd(0, sqrt(2/(3*3*512)), 3, 3, 512, 512)), zeros(1,512,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 1, ...
						   'droprate', 0.4) ;
net.layers{end+1} = struct('type', 'bnorm', ...
                           'weights', {{ones(512, 1, 'single'), zeros(512, 1, 'single')}}, ...
                           'learningRate', lr_bnorm, ...
                           'weightDecay', wdr_bnorm, ...
						   'Epsilon', epsilon) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'dropout', 'rate', 0.4) ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{single(normrnd(0, 0.01, 3, 3, 512, 512)), zeros(1,512,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 1, ...
						   'droprate', 0.5) ;
net.layers{end+1} = struct('type', 'bnorm', ...
                           'weights', {{ones(512, 1, 'single'), zeros(512, 1, 'single')}}, ...
                           'learningRate', lr_bnorm, ...
                           'weightDecay', wdr_bnorm, ...
						   'Epsilon', epsilon) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'dropout', 'rate', 0.5) ;
						   
% Block 6
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{single(normrnd(0, 0.01, 1, 1, 512, 512)), zeros(1,512,'single')}}, ...
                           'learningRate', lr, ...
						   'weightDecay', wdr, ...
                           'stride', 1, ...
                           'pad', 0, ...
						   'droprate', 0.5) ;
net.layers{end+1} = struct('type', 'bnorm', ...
                           'weights', {{ones(512, 1, 'single'), zeros(512, 1, 'single')}}, ...
                           'learningRate', lr_bnorm, ...
                           'weightDecay', wdr_bnorm, ...
						   'Epsilon', epsilon) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'dropout', 'rate', 0.5) ;

% Block 7
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{single(normrnd(0, 0.001, 1, 1, 512, 10)), zeros(1,10,'single')}}, ...
                           'learningRate', lr, ...
						   'weightDecay', wdr, ...
                           'stride', 1, ...
                           'pad', 0, ...
						   'droprate', 0) ;

% Loss layer
net.layers{end+1} = struct('type', 'softmaxloss') ;
