inputs = imageArray/255;
outputcnt = 3
denomfactor = 100
%expected = zeros(1,60);
%for k=1:60 
%    if(k<31) 
%        expected(1,k)=1;
%    elseif(k<61)
   %     expected(1,k)=2;
  %  elseif(k<91)
  %      expected(1,k)=3;
  %  elseif(k<121)
  %      expected(1,k)=4;
  %  elseif(k<151)
  %      expected(1,k)=5;
  %  elseif(k<181)
  %      expected(1,k)=6;
  %  elseif(k<211)
  %      expected(1,k)=7;
  %  else
  %      expected(1,k)=8; 
        
%    end
%end
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

filters1 = (-1+2*rand(32,3,6,6))/denomfactor;
gradfilters1 = zeros(32,3,6,6);

bias1 =  (-1+2*rand(1,32))/denomfactor;
gradbias1 = zeros(1,32);

filters2 = (-1+2*rand(64,32,4,4))/denomfactor;
gradfilters2 = zeros(64,32,4,4);

bias2 =  (-1+2*rand(1,64))/denomfactor;
gradbias2 = zeros(1,64);

filters3 = (-1+2*rand(128,64,4,4))/denomfactor;
gradfilters3 = zeros(128,64,4,4);

bias3 =  (-1+2*rand(1,128))/denomfactor;
gradbias3 = zeros(1,128);

weights1 = (-1+2*rand(64*128,512))/denomfactor;
gradweights1 = zeros(64*128,512);

mlpbias1 = (-1+2*rand(1,512))/denomfactor;
gradmlpbias1 = zeros(1,512);

weights2 = (-1+2*rand(512,outputcnt))/denomfactor;
gradweights2 = zeros(512,outputcnt);

mlpbias2 = (-1+2*rand(1,outputcnt))/denomfactor;
gradmlpbias2 = zeros(1,outputcnt);

lr = 0.1;

tic

accerr = 0
iter_cnt = 100
cor = 0
sizeofin = size(inputs)
inputsize = sizeofin(1)
for k=1:iter_cnt
    accgradfilters1 = zeros(32,3,6,6);
    accgradbias1 = zeros(1,32);
    accgradfilters2 = zeros(64,32,4,4);
    accgradbias2 = zeros(1,64);
    accgradfilters3 = zeros(128,64,4,4);
    accgradbias3 = zeros(1,128);
    accgradweights1 = zeros(64*128,512);
    accgradmlpbias1 = zeros(1,512);

    accgradweights2 = zeros(512,outputcnt);
    accgradmlpbias2 = zeros(1,outputcnt);
accerr = 0;
cor = 0;
toc
for i=1:inputsize 
%i=1 ;
output1 = convolutionCor(reshape(inputs(i,:,:,:),3,128,128),filters1,bias1);
[output2, pospool1] = maxpooling(output1,3,3);
output3 = ReLu(output2);
output4 = convolutionCor(output3,filters2,bias2);
[output5, pospool2] = maxpooling(output4,2,2);
output6 = ReLu(output5);
output7 = convolutionCor(output6,filters3,bias3);
[output8, pospool3] = maxpooling(output7,2,2);
output9 = ReLu(output8);
output10 = reshape(output9,1,128*64);

output11 = output10 * weights1 + mlpbias1;
output12 = ReLu(output11);
output13 = output12 * weights2 + mlpbias2;

output14 = exp(output13)/sum(exp(output13));

[val pos] = max(output14);

t = zeros(1,outputcnt);
class = expected(1,i);
t(class) = 1;
if pos == class 
    cor = cor +1;
end 
crossenterr= (-t*log(output14)');
accerr= crossenterr + accerr;

errof14 = output14-t;

gradmlpbias2(:) = errof14 ;
gradweights2(:) = output12'*errof14;

errof13 = (weights2*errof14')'; %% add all coming errors 1x512
errof12 = errof13;
errof12(find(output11<0))=0; %% always err12 = err13 relu :S 

gradmlpbias1(:) = errof12;
gradweights1(:) = output10'*errof12;

errof11 = (weights1*errof12')'; %% add all coming errors 12544

errof10 = reshape(errof11,128,8,8);

errof9 = errof10;
errof9(find(output8<0))=0;

errof8 = bpmaxpooling(errof9,pospool3,2,2);

[errof7, gradfilters3(:), gradbias3(:)] = bpConvolutionCor(output6,errof8,filters3);
errof6 = errof7;
errof6(find(output5<0))=0;

errof5 = bpmaxpooling(errof6,pospool2,2,2);

[errof4, gradfilters2(:), gradbias2(:)] = bpConvolutionCor(output3,errof5,filters2);
errof3 = errof4;
errof3(find(output2<0))=0;

errof2 = bpmaxpooling(errof3,pospool1,3,3);

[errof1, gradfilters1(:), gradbias1(:)] = bpConvolutionCor(reshape(inputs(i,:,:,:),3,128,128),errof2,filters1);

accgradfilters1 = accgradfilters1 + gradfilters1;
accgradbias1 = accgradbias1+gradbias1;

accgradfilters2 = accgradfilters2 + gradfilters2;
accgradbias2 = accgradbias2 + gradbias2;

accgradfilters3 = accgradfilters3+gradfilters3;
accgradbias3 = accgradbias3 +gradbias3;

accgradweights1 = accgradweights1 + gradweights1;
accgradmlpbias1 = accgradmlpbias1 + gradmlpbias1;

accgradweights2 = accgradweights2 + gradweights2;
accgradmlpbias2 = accgradmlpbias2 + gradmlpbias2;

end

filters1 = filters1 - lr* accgradfilters1/inputsize;
bias1 =  bias1 - lr*accgradbias1/inputsize;

filters2 = filters2 - lr * accgradfilters2/inputsize;
bias2 =  bias2 - lr*accgradbias2/inputsize;

filters3 = filters3 - lr * accgradfilters3/inputsize;
bias3 = bias3 - lr * accgradbias3/inputsize;

weights1 = weights1 - lr* accgradweights1/inputsize;
mlpbias1 = mlpbias1 - lr * accgradmlpbias1/inputsize;

weights2 = weights2 - lr * accgradweights2/inputsize;
mlpbias2 = mlpbias2 - lr * accgradmlpbias2/inputsize;
aveerr= accerr/sizeofin(1) 
cor

end 

