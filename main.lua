#!/usr/bin/env th

require 'nn'
require 'sys'
require 'xlua'
-- require 'dpnn'
require 'torch'
require 'optim'
require 'paths'
require 'image'
require 'paths'
require 'torch'
require 'torchx'

local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)
print(opt)
torch.save(paths.concat(opt.save, 'opts.t7'), opt, 'ascii')
print('Saving everything to: ' .. opt.save)

torch.setdefaulttensortype('torch.FloatTensor')
if opt.cuda then
   print('\n<==================== CUDA support ====================>\n')
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.device)
end

-- torch.setnumthreads(opt.threads)
-- torch.manualSeed(opt.manualSeed)

print('<==================== execute all files ====================>')

local trainData, trainGT, testData, testGT = paths.dofile('data.lua')
-- local model = require 'models'
-- local criterion = paths.dofile('loss.lua')
-- local train = paths.dofile('train.lua')




-- print(trainData[1])
-- print(trainGT:size())
-- print(testData:size())
-- print(testGT:size())
-- local epoch = 1

-- print('<====================> train network ====================>')
-- for _=1, opt.nEpochs do
-- 	train.process(model, criterion, trainData, trainGT, testData, testGT, opt, epoch)
-- 	epoch = epoch + 1
-- end
