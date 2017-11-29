#!/usr/bin/env th
require 'nn'
require 'sys'
require 'xlua'
-- require 'dpnn'
require 'torch'
require 'optim'
require 'image'
require 'paths'
require 'image'
require 'paths'
require 'torch'
require 'torchx'

print('<==================== define deep nn model ====================>')

local M = {}
function M.spatialdeep(continue)

	net = nn.Sequential()    
	-- layer 1
	-- (input, output, kernal width, kernal height, stride width, stride height, padding one, padding two )
	net:add(nn.SpatialConvolution(9,64,3,3,1,1,1,1))
	-- (batch size)
	-- net:add(nn.SpatialBatchNormalization(64))
	net:add(nn.ReLU())
	-- max pooling 
	-- (pooling width, pooling height, stride width, stide height, padding one, padding two)
	net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

	-- layer 2 
	net:add(nn.SpatialConvolution(64,64,3,3,1,1,1,1))
	-- net:add(nn.SpatialBatchNormalization(64))
	net:add(nn.ReLU())
	-- max pooling 
	-- (pooling width, pooling height, stride width, stide height, padding one, padding two)
	net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

	-- layer 8
	-- net:add(nn.SpatialConvolution(64,128,3,3,1,1,1,1))
	-- net:add(nn.SpatialBatchNormalization(128))
	-- net:add(nn.ReLU())
	-- dropout with 0.5
	-- net:add(nn.Dropout(0.500000))
	    
	-- first fully connected layer
	-- (16*16*128)
	net:add(nn.View(12288))
	net:add(nn.Linear(12288,1024))
	-- dropout with 0.5
	net:add(nn.Dropout(0.500000))
	    
	-- final fully connected layer 
	net:add(nn.Linear(1024, 8))
	print(net:__tostring())
	return net
end

return M.spatialdeep()