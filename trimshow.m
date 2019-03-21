function trimshow(image)
sizeofimage = size(image);
d = sizeofimage(1); % if rgb w is 3
h = sizeofimage(2);
w = sizeofimage(3);

new = zeros(h,w,d);

for i=1:d 
    new(:,:,i) = image(i,:,:);
end 

imshow(new)
end 