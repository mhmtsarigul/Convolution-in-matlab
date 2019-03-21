function [inputerror,gradweights,gradbias] = gpubpConvolutionCor(input,errors,filters) % nxmxk input nxmxkxp filters
sizeofinput = size(input);
sizeoffilters = size(filters);
sizeoferrors = size(errors);
n = sizeofinput(1);
x = sizeofinput(2);
y = sizeofinput(3);

fn = sizeoffilters(1);
fd = sizeoffilters(2);
fx = sizeoffilters(3);
fy = sizeoffilters(4);

en= sizeoferrors(1);
ex= sizeoferrors(2);
ey = sizeoferrors(3);

gradweights = zeros(fn,fd,fx,fy);
gradbias = zeros(1,fn);

inputerror =  zeros(n,x,y);

combinp = zeros(n,ex*ey,(x-ex+1)*(y-ey+1));

for i=1:n 
coninp = im2col(reshape(input(i,:,:),x,y),[ex ey]);
     combinp(i,:,:) = coninp;
end

errors = gpuArray(errors);
combinp = gpuArray(combinp);
gradweights = gpuArray(gradweights);
for i=1:en
        cf = im2col(reshape(errors(i,:,:),ex,ey),[ex ey]);
for j=1:n 
  outfilter = reshape(combinp(j,:,:),ex*ey,(x-ex+1)*(y-ey+1))'*cf;
   temp = reshape(outfilter,1,1,(x-ex+1),(y-ey+1));
    gradweights(i,j,:,:) = gradweights(i,j,:,:)+temp;
end
end
errors = gather(errors);
gradweights = gather(gradweights);

gradbias(:,:) = sum(sum(sum(gradweights,4),3),2)';

for i=1:en
    curfil = reshape(filters(i,:,:,:),fd,fx,fy);
    for j=1:ex
        for k=1:ey
            if errors(i,j,k) ~=0
            errsum = errors(i,j,k)*curfil;
            inputerror(:,j:(j+fx-1),k:(k+fy-1)) = inputerror(:,j:(j+fx-1),k:(k+fy-1)) + errsum; 
            end
        end
    end
end

end 