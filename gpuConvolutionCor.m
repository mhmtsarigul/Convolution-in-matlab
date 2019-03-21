function output = gpuConvolutionCor(input,filters,biasvals) % nxmxk input nxmxkxp filters
sizeofinput = size(input);
sizeoffilters = size(filters);

n = sizeofinput(1);
x = sizeofinput(2);
y = sizeofinput(3);

fn = sizeoffilters(1);
fd = sizeoffilters(2);
fx = sizeoffilters(3);
fy = sizeoffilters(4);


output = zeros(fn,x-fx+1,y-fy+1);
output = gpuArray(output);
ginput=gpuArray(input);
gfilters = gpuArray(filters);
gbiasvals = gpuArray(biasvals);

combinp = zeros(n,fx*fy,(x-fx+1)*(y-fy+1));

combinp = gpuArray(combinp);
for i=1:n 
coninp = im2col(reshape(ginput(i,:,:),x,y),[fx fy]);
     combinp(i,:,:) = coninp;
end 

for j=1:fd 
ress = reshape(combinp(j,:,:),fx*fy,(x-fx+1)*(y-fy+1))';
    for i=1:fn 
    
    cf = im2col(reshape(filters(i,j,:,:),fx,fy),[fx fy]);
  outfilter = ress*cf;
   temp = reshape(outfilter,x-fx+1,y-fy+1);
    output(i,:,:) = output(i,:,:)+reshape(temp,1,x-fx+1,y-fy+1);
    end
end

for i=1:fn 
    output(i,:,:) = output(i,:,:) + biasvals(1,i);
end 

output=gather(output);

end 