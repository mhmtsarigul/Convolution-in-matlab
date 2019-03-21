function output = convolutionFast(input,filters,biasvals) % nxmxk input nxmxk filters
sizeofinput = size(input);
sizeoffilters = size(filters);

n = sizeofinput(1);
x = sizeofinput(2);
y = sizeofinput(3);

fn = sizeoffilters(1);
fx = sizeoffilters(2);
fy = sizeoffilters(3);


output = zeros(fn,x-fx+1,y-fy+1);


for i=1:n 
coninp = im2col(reshape(input(i,:,:,:),x,y),[fx fy]);
    if i==1 
    combinp = coninp;
    else
    combinp = [combinp coninp];
    end 
end 
for i=1:fn 
    cf = im2col(reshape(filters(i,:,:),fx,fy),[fx fy]);
    size(cf);
    size(combinp);
    outfilter = combinp'*cf;
    size(outfilter);
    temp = reshape(outfilter,x-fx+1,y-fy+1,n);
    output(i,:,:) = sum(temp,3);
    output(i,:,:) = output(i,:,:) + biasvals(i);
end 

end 