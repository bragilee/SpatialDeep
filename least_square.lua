require 'math'
require 'torch'
require 'math'


-- local inputData = torch.Tensor(9,16,12)
-- for i=1,inputData:size()[1] do
-- 	for j=1,inputData:size()[2] do
-- 		for k=1,inputData:size()[3] do
-- 			inputData[i][j][k] = math.random(255)
-- 		end
-- 	end
-- end

local inputData = torch.Tensor(192,9)
for i=1,inputData:size()[1] do
	for j=1,inputData:size()[2] do
		inputData[i][j] = math.random(255)
	end
end

print(inputData[1])
u, s, v = torch.svd(inputData)

