function output = convolutionM(input,filters) % nxm input nxmxk filters
sizeofinput = size(input);
sizeoffilters = size(filters);

x = sizeofinput(1);
y = sizeofinput(2);

n = sizeoffilters(1);
fx = sizeoffilters(2);
fy = sizeoffilters(3);


output = zeros(n,x-fx+1,y-fy+1);

for f=1:n
    filters(f,:,:);
    curFilter = reshape(filters(f,:,:),fx,fy);
    for i=1:x-fx+1
        for j=1:y-fy+1
            output(f,i,j) = sum(sum(input(i:i+fx-1,j:j+fy-1).*curFilter));
        end
    end 

end 

end 