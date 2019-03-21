function [output,positions] = maxpooling(input,n,m)

sizeofinput = size(input);
d = sizeofinput(1);
h = sizeofinput(2);
w = sizeofinput(3);

output = zeros(d,h/n,w/m);
positions = zeros(d,h/n,w/m);
for i=1:d 
    for j=1:h/n
        for k=1:w/m
            
            temp = input(i,(j-1)*n+1:j*n,(k-1)*m+1:k*m);
            [output(i,j,k),pos] = max(temp(:));
            positions(i,j,k) = pos;
           
        end 
    end 
end 

end 