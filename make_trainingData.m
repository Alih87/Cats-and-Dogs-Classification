path_c = "C:\Users\Alih1\Desktop\Image Recognition\graycats_t";
path_d = "C:\Users\Alih1\Desktop\Image Recognition\graydogs_t";
images_c = dir(fullfile(path_c, "*.jpg"));
images_d = dir(fullfile(path_d, "*.jpg"));
row_c = zeros(numel(images_c), 10000);
row_d = zeros(numel(images_d), 10000);

for i = 1:numel(images_c)
img_c = imread(fullfile(path_c, images_c(i).name));
img_d = imread(fullfile(path_d, images_d(i).name));
img_c = imresize(img_c, [100 100]);
img_d = imresize(img_d, [100 100]);
row_c(i,:) = reshape(img_c, [1, numel(img_c)]);
row_d(i,:) = reshape(img_d, [1, numel(img_d)]);
end

y_c = 2.*ones(size(row_c,1),1);
y_d = 3.*ones(size(row_d,1),1);
cats = [row_c, y_c];
dogs = [row_d, y_d];
dat = [cats; dogs];
data = dat(randperm(size(dat,1)),:);
save trainingData.mat data
