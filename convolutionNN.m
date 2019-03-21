function output = convolutionNN(input,filters,biasvals) % nxmxk input nxmxk filters
%input = gpuArray(input)
%filters = gpuArray(filters)
%biasvals = gpuArray(biasvals)
sizeofinput = size(input);
sizeoffilters = size(filters);

n = sizeofinput(1);
x = sizeofinput(2);
y = sizeofinput(3);

fn = sizeoffilters(1);
fx = sizeoffilters(2);
fy = sizeoffilters(3);


output = zeros(fn,x-fx+1,y-fy+1);

for f=1:fn
    filters(f,:,:);
    curFilter = reshape(filters(f,:,:),fx,fy);
    for i=1:x-fx+1
        for j=1:y-fy+1
            for k=1:n
                t1 = reshape(input(k,i:i+fx-1,j:j+fy-1),fx,fy).*curFilter;
                output(f,i,j) = output(f,i,j) + sum(sum(t1));
            end
                output(f,i,j) = output(f,i,j) + biasvals(f);
            
        end 
    end
   
end 

end 