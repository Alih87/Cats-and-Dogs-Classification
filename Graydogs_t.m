path = "C:\Users\Alih1\Desktop\Image Recognition\Training_set\dogs";
rgbdogs = dir(fullfile(path, "*.jpg"));
graydogs = fullfile(pwd, "graydogs_t");
if ~exist(graydogs, 'dir')
    mkdir(graydogs);
end
for i = 1:numel(rgbdogs)
    Rgb = imread(fullfile(path, rgbdogs(i).name));
    gray = rgb2gray(Rgb);
    name = sprintf("%d.jpg", i);
    fullname = fullfile(graydogs, name);
    imwrite(gray, fullname)
end