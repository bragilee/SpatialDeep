require 'math'
require 'torch'
require 'math'
require 'xlua'


local D={}
function D.dataset(d_number)
	local dataNumber = d_number
	local inputSize = {9,64,48}
	local trainDataSet = torch.Tensor(dataNumber,inputSize[1],inputSize[2],inputSize[3])
	local trainSetGT = torch.Tensor(dataNumber, inputSize[1]-1)

	for index = 1, dataNumber do 
		xlua.progress(index, dataNumber)
		local trainData = torch.Tensor(inputSize[1],inputSize[2],inputSize[3])
		for i=1,inputSize[1] do
			for j=1,inputSize[2] do
				for k=1,inputSize[3] do
					trainData[i][j][k] = math.random(255)
				end
			end
		end
		print(trainData[1][1])
		local matrix = torch.Tensor(inputSize[2]*inputSize[3],inputSize[1])
		for m=1,trainData:size()[1] do
			local flag = 1
			local vector = torch.Tensor(inputSize[2]*inputSize[3],1)
			for n=1,trainData:size()[2] do
				for l=1,trainData:size()[3] do
					vector[flag] = trainData[m][n][l]
					flag = flag + 1
				end
			end
			matrix:select(2,m):copy(vector)
		end
		print(matrix[{{1,14},1}])

		u,s,v = torch.svd(matrix)
		s = torch.diag(s)
		s[9][9] = 0
		x = v:select(2,9)
		x_cal = v:select(2,9)
		-- print(x)
		x:div(x[9])
		-- print(x)

		for tg_i = 1,inputSize[1]-1 do
			trainSetGT[index][tg_i] = x[tg_i]
		end
		-- print(trainSetGT[index])
		v = v:t()
		local us = torch.Tensor(inputSize[2]*inputSize[3],inputSize[1])
		for ii=1,u:size()[1] do
			for jj=1,s:size()[1] do
				z = torch.cmul(u:select(1,ii),s:select(2,jj))
				us[ii][jj] = torch.cumsum(z)[9]
			end
		end

		local usv = torch.Tensor(inputSize[2]*inputSize[3],9)

		for iii=1,us:size()[1] do
			for jjj=1,v:size()[1] do
				zz = torch.cmul(us:select(1,iii),v:select(2,jjj))
				usv[iii][jjj] = torch.cumsum(zz)[9]
			end
		end
		print(usv[{{1,14},1}])

		--test
		-- print(usv)

		for i_cal=1, usv:size()[1] do 
			z_cal = torch.cmul(usv:select(1,i_cal),x_cal)
			-- print(torch.cumsum(z_cal)[9])
		end

		for mm=1,usv:size()[2] do
			r_vector = usv:select(2,mm)
			local fflag = 1
			for nn=1,trainData:size()[2] do
				for ll=1,trainData:size()[3] do
					trainData[mm][nn][ll] = r_vector[fflag]
					fflag = fflag + 1
				end
			end
		end
		print(trainData[1][1])
		trainDataSet[index]:copy(trainData)
	end
	return trainDataSet,trainSetGT
end

local trainingData,trainingGT,testingData,testingGT
if not paths.filep(paths.concat(opt.cache, 'trainData.t7')) then
	trainingData,trainingGT = D.dataset(1)
	-- torch.save(paths.concat(opt.cache, 'trainData.t7'), trainingData)
	-- torch.save(paths.concat(opt.cache, 'trainGT.t7'), trainingGT)
else
	trainingData = torch.load(paths.concat(opt.cache, 'trainData.t7'))
	trainingGT = torch.load(paths.concat(opt.cache, 'trainGT.t7'))
end
if not paths.filep(paths.concat(opt.cache, 'testData.t7')) then
	testingData,testingGT = D.dataset(1)
	-- torch.save(paths.concat(opt.cache, 'testData.t7'), testingData)
	-- torch.save(paths.concat(opt.cache, 'testGT.t7'), testingGT)
else
	testingData = torch.load(paths.concat(opt.cache, 'testData.t7'))
	testingGT = torch.load(paths.concat(opt.cache, 'testGT.t7'))
end
return trainingData,trainingGT, testingData, testingGT


