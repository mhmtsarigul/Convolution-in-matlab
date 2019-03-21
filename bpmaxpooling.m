function output = bpmaxpooling(input,positions,n,m)

sizeofinput = size(input);
d = sizeofinput(1);
h = sizeofinput(2);
w = sizeofinput(3);


output = zeros(d,h*n,w*m);

for i=1:d 
    for j=1:h
        for k=1:w
            temp = zeros(n,m);
            temp(positions(i,j,k)) = input(i,j,k);
            output(i,(j-1)*n+1:j*n,(k-1)*m+1:k*m) = temp;
        end 
    end
end


end 