#!/usr/bin/env th
require 'nn'
require 'sys'
-- require 'dpnn'
require 'torch'
require 'optim'
require 'image'
require 'paths'
require 'xlua'
require 'image'
require 'paths'
require 'torch'
require 'torchx'

local train = {}
channel = 9
imgSize = {64,48}
outputSize = 8
-- save log file
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
function train.process(model, criterion, trainData, trainGT, testData, testGT, opt, epoch)

   local epochLoss = 0.0

   if opt.cuda then
      model:cuda()
      criterion:cuda()
   end

   if model then
      parameters,gradParameters = model:getParameters()
   end

   local batchSize = opt.batchSize
   local dataSize = trainData:size()[1]
   -- configure optimization
   local optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 0.1
         }
   optimMethod = optim.sgd

   print('==========> taining procedure')
   print('==========> doing epoch on training data')
   print("==========> epoch # " .. epoch)

   model:training()

   local tm = torch.Timer()

   for i = 1, dataSize, batchSize do
      -- disp progress
      xlua.progress(i, dataSize)

      -- create mini-batch
      local inputData = torch.Tensor(batchSize, channel, imgSize[1], imgSize[2])
      local groudTruth = torch.Tensor(batchSize, outputSize)

      local k = 1
      for j = i, math.min(i+batchSize-1, dataSize) do
         inputData[k] = torch.Tensor(channel, imgSize[1],imgSize[2]):copy(trainData[j])
         groudTruth[k] = torch.Tensor(outputSize):copy(trainGT[j])
         k = k + 1
      end
      
      if opt.cuda then
         inputData = inputData:cuda()
         groudTruth = groudTruth:cuda()
      end
      
      local train_loss = 0.0
      local output_mode = 0.0
      local gt_mode = 0.0
      local output_gt = 0.0

      collectgarbage()
      local feval = function(x)
         if x ~= parameters then
            parameters:copy(x)
         end

         --reset gradients
         gradParameters:zero()

         local outputs = model:forward(inputData)
         local loss = criterion:forward(outputs,groudTruth)

         -- define loss

         local train_loss = 0.0
         for index = 1, batchSize do
            output_mode = torch.sqrt(torch.cumsum(torch.pow(outputs[index],2))[outputSize])
            gt_mode = torch.sqrt(torch.cumsum(torch.pow(groudTruth[index],2))[outputSize])
            output_gt = torch.cumsum(torch.cmul(outputs[index],groudTruth[index]))[outputSize]
            train_loss = train_loss + output_gt/(output_mode,gt_mode)
         end

         train_loss = train_loss/batchSize
         --

         local gradients = criterion:backward(outputs,groudTruth)
         model:backward(inputData, gradients)

         gradParameters:div(inputData:size()[1])

         epochLoss = epochLoss + train_loss

         return loss,gradParameters
      end
      -- print(inputData[1])
      -- print(groudTruth[1])

      optimMethod(feval, parameters, optimState)

   end
   local iteration = dataSize/batchSize
   epochLoss = epochLoss / iteration
   trainLogger:add{['loss'] = epochLoss,}
   print('')
   print('==========> training loss: ', epochLoss)
   print('\n')
   time = tm:time().real
   timeSample = time / dataSize
   print('Time is: ', time)
   print('Time per sample is: ', timeSample)
   torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model:float():clearState())
   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)

   -- test dataset

   local testDataSize = testData:size()[1]
   model:evaluate()
   local test_loss = 0.0
   local output_mode = 0.0
   local gt_mode = 0.0
   local output_gt = 0.0

   for iT = 1, testDataSize do
      -- disp progress
      xlua.progress(iT, testDataSize)
     
      if opt.cuda then
         testData = testData:cuda()
         testGT = testGT:cuda()
      end

      local testOutput = model:forward(testData[iT])

      -- define testing loss
      output_mode = torch.sqrt(torch.cumsum(torch.pow(testOutput,2))[outputSize])
      gt_mode = torch.sqrt(torch.cumsum(torch.pow(testGT[iT],2))[outputSize])
      output_gt = torch.cumsum(torch.cmul(testOutput,testGT[iT]))[outputSize]
      test_loss = test_loss + output_gt/(output_mode,gt_mode)

      -- local meanError = torch.Tensor(outputSize):fill(0)
      -- for nO = 1, outputSize do
         -- meanError[nO] = torch.pow((testOutput[nO] - testGT[iT][nO]), 2)
      -- end
      -- print(meanError)
      -- test_loss = test_loss + torch.sqrt(torch.cumsum(meanError)[outputSize])/outputSize
      -- print(meanError)
      -- print(batchError)
   end
   test_loss = test_loss/testDataSize
   testLogger:add{['error'] = test_loss,}
   print('\n')
   print('==========> testing error: ', test_loss)
   print('\n')
end

return train
















