

output1 = convolutionCor(in1,filters1,bias1);
[output2, pospool1] = maxpooling(output1,3,3);
output3 = ReLu(output2);
output4 = convolutionCor(output3,filters2,bias2);
[output5, pospool2] = maxpooling(output4,2,2);
output6 = ReLu(output5);
output7 = convolutionCor(output6,filters3,bias3);
[output8, pospool3] = maxpooling(output7,2,2);
output9 = ReLu(output8);
output10 = reshape(output9,1,256*64);

output11 = output10 * weights1 + mlpbias1;
output12 = ReLu(output11);
output13 = output12 * weights2 + mlpbias2;

output14 = exp(output13)/sum(exp(output13));

[val pos] = max(output14)