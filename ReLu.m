function output= ReLu(input) % nxmxk
input(input<0)=0;
output = input;
end 