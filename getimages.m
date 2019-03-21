% Specify the folder where the files live.
imageArray= zeros(236,3,128,128);
expected = zeros(1,236)

a=1

myFolder = '/home/msg/Matlab/DeepLearning/images2/';

for i=1:3
tempdir = [myFolder num2str(i,'%d') '/resized/']

% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isdir(tempdir)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', tempdir);
  uiwait(warndlg(errorMessage));
  return;
end
% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(tempdir, '*.jpg'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
for k = 1 : length(theFiles)
  baseFileName = theFiles(k).name;
  fullFileName = fullfile(tempdir, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  % Now do whatever you want with this file name,
  % such as reading it in as an image array with imread()
  temp = imread(fullFileName);
  imshow(temp)
  for j=1:3
      imageArray(a,j,:,:) = reshape(temp(:,:,j),1,128,128);
  end 
  
  expected(1,a)=i
  a= a+1
%  imshow(imageArray);  % Display image.
%  drawnow; % Force display to update immediately.
end

end 