require 'math'
require 'torch'

testOutput = torch.Tensor({0.0557,0.0553,0.1829,0.0171,0.0187,0.1824,0.0219,0.0715,0.1844})
testGT = torch.Tensor({0.035,0.4923,0.0335,-0.0191,0.1232,0.8792,-0.1948,-0.7487,1})
-- output_mode = torch.sqrt(torch.cumsum(torch.pow(testOutput,2))[9])
-- print(output_mode)
-- gt_mode = torch.sqrt(torch.cumsum(torch.pow(testGT,2))[9])
-- print(torch.cumsum(torch.pow(testGT,2)))
-- print(gt_mode)
-- output_gt = torch.cumsum(torch.cmul(testOutput,testGT))[9]
-- print(torch.cmul(testOutput,testGT))
-- print(output_gt)
-- print(output_gt/(output_mode*gt_mode))


print(testOutput:size()[1])