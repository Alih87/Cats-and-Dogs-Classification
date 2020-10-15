path = "C:\Users\Alih1\Desktop\Image Recognition\Training_set\cats";
rgbcats = dir(fullfile(path, "*.jpg"));
graycats = fullfile(pwd, "graycats_t");
if ~exist(graycats, 'dir')
    mkdir(graycats);
end
for i = 1:numel(rgbcats)
    Rgb = imread(fullfile(path, rgbcats(i).name));
    gray = rgb2gray(Rgb);
    name = sprintf("%d.jpg", i);
    fullname = fullfile(graycats, name);
    imwrite(gray, fullname)
end