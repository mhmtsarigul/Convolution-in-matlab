function [inputerror,gradweights,gradbias] = bpConvolutionCorEasy(input,errors,filters) % nxmxk input nxmxkxp filters
sizeofinput = size(input)
sizeoffilters = size(filters)
sizeoferrors = size(errors)
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

%combinp = zeros(n,ex*ey,(x-ex+1)*(y-ey+1));


for i=1:en 
        cf = im2col(reshape(errors(i,:,:),ex,ey),[ex ey]);

for j=1:n 
    conbinp = im2col(reshape(input(j,:,:),x,y),[ex ey]);
    size(conbinp)
   outfilter = reshape(conbinp(:,:),ex*ey,(x-ex+1)*(y-ey+1))'*cf;
   temp = reshape(outfilter,(x-ex+1),(y-ey+1));
    gradweights(i,j,:,:) = gradweights(i,j,:,:)+reshape(temp,1,1,x-ex+1,y-ey+1);
end
end

gradbias(:,:) = sum(sum(sum(gradweights,4),3),2)';

for i=1:en
    for j=1:ex
        for k=1:ey
            errsum = errors(i,j,k)*filters(i,:,:,:);
            size(errsum);
            errsum = reshape(errsum,fd,fx,fy);
        size(inputerror);
        inputerror(:,j:(j+fx-1),k:(k+fy-1)) = inputerror(:,j:(j+fx-1),k:(k+fy-1)) + errsum; 
       end
    end
end

end 