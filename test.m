inputs = rand(100,3,231,231);
expected = randi(4,[1 100]);
% network details 
% 7x7 conv 225x225  -- 64 filters 
% 3x3 pooling 75x75 
% relu
% 4x4 conv 72x72 -- 128 filters 
% 3x3 pooling 24x24
% relu
% 4x4 conv 21x21 -- 256 filters
% 3x3 pooling 7x7
% relu
% reshape 49 * 256 neurons
% mlp 512 
% mlp 4 class ... output softmax bla bla bla 

filters1 = (-1+2*rand(64,3,7,7))/100;
gradfilters1 = zeros(64,3,7,7);

bias1 =  (-1+2*rand(1,64))/100;
gradbias1 = zeros(1,64);

filters2 = (-1+2*rand(128,64,4,4))/100;
gradfilters2 = zeros(128,64,4,4);

bias2 =  (-1+2*rand(1,128))/100;
gradbias2 = zeros(1,128);

filters3 = (-1+2*rand(256,128,4,4))/100;
gradfilters3 = zeros(256,128,4,4);

bias3 =  (-1+2*rand(1,256))/100;
gradbias3 = zeros(1,256);

weights1 = (-1+2*rand(49*256,512))/100;
gradweights1 = zeros(49*256,512);

mlpbias1 = (-1+2*rand(1,512))/100;
gradmlpbias1 = zeros(1,512);

weights2 = (-1+2*rand(512,4))/100;
gradweights2 = zeros(512,4);

mlpbias2 = (-1+2*rand(1,4))/100;
gradmlpbias2 = zeros(1,4);

lr = 0.001;

tic
%for i=1:100 
i=1 ;
output1 = convolutionCor(reshape(inputs(i,:,:,:),3,231,231),filters1,bias1);
toc
[output2, pospool1] = maxpooling(output1,3,3);
toc
output3 = ReLu(output2);
toc
output4 = gpuConvolutionCor(output3,filters2,bias2);
toc
[output5, pospool2] = maxpooling(output4,3,3);
toc
output6 = ReLu(output5);
toc
output7 = convolutionCor(output6,filters3,bias3);
toc
[output8, pospool3] = maxpooling(output7,3,3);
toc
output9 = ReLu(output8);
toc
output10 = reshape(output9,1,256*49);

output11 = output10 * weights1 + mlpbias1;
output12 = ReLu(output11);
output13 = output12 * weights2 + mlpbias2;

output14 = exp(output13)/sum(exp(output13));

t = zeros(1,4);
class = expected(1,i);
t(class) = 1;

errof14 = output14-t;

gradmlpbias2 = errof14 ;
gradweights2 = output12'*errof14;

errof13 = sum(output12'*errof14,2)'; %% add all coming errors 1x512
errof12 = errof13;
errof12(find(output11<0))=0; %% always err12 = err13 relu :S 

gradmlpbias1 = errof12;
gradweights1 = output10'*errof12;

errof11 = sum(output10'*errof12,2)'; %% add all coming errors 12544

errof10 = reshape(errof11,256,7,7);

errof9 = errof10;
errof9(find(output8<0))=0;
disp('bpmax')
errof8 = bpmaxpooling(errof9,pospool3,3,3);

toc
[errof7, gradfilters3, gradbias3] = bpConvolutionCor(output6,errof8,filters3);
toc
errof6 = errof7;
errof6(find(output5<0))=0;

errof5 = bpmaxpooling(errof6,pospool2,3,3);
toc

[errof4, gradfilters2, gradbias2] = gpubpConvolutionCor(output3,errof5,filters2);
toc
errof3 = errof4;
errof3(find(output2<0))=0;

errof2 = bpmaxpooling(errof3,pospool1,3,3);
toc

[errof1, gradfilters1, gradbias1] = bpConvolutionCor(reshape(inputs(i,:,:,:),3,231,231),errof2,filters1);
toc
%{
err14 = zeros(1,4)
for l=1:4
    if l==class
        err14(1,l) = output14(1,l)*(1-output14(1,l))
    else 
        err14(1,l) = -output14(1,l)*output14(1,class)
    end 
end



xoutput1 = convolutionNN(inputs(i,:,:),filters1,bias1);
xoutput2 = maxpooling(xoutput1,3,3);
xoutput3 = ReLu(xoutput2);
xoutput4 = convolutionNN(xoutput3,filters2,bias2);
xoutput5 = maxpooling(xoutput4,3,3);
xoutput6 = ReLu(xoutput5);
xoutput7 = convolutionNN(xoutput6,filters3,bias3);
xoutput8 = maxpooling(xoutput7,3,3);
xoutput9 = ReLu(xoutput8);
xoutput10 = reshape(xoutput9,1,256*49)
%}

toc

%end 


