function [cur_tmp, cur_img, cur_dim, cur_in_mat] = imrescale(tmp, img, scale, in_mat, dim, tz_square)
% Rescale images and modify the related parameters
%
% Usage:
%   [cur_tmp, cur_img, cur_dim_tmp, cur_dim_img, cur_in_mat] = imrescale(tmp, img, scale, in_mat, dim, tz_square)
%
% Inputs:
%   tmp         = template image (in YCbCr format)
%   img         = camera frame (in YCbCr format)
%   scale       = current scale
%   in_mat      = original intrinsic matrix
%   dim         = dimension variables
%   tz_square   = min_z * max_z
%
% Outputs:
%   cur_tmp     = current template image in this scale (in YCbCr format)
%   cur_img     = camera frame in this scale (in YCbCr format)
%   cur_dim_tmp = current dimension of template image in this scale (contain w and h)
%   cur_dim_img = current dimension of camera frame in this scale (contain w and h)
%   cur_in_mat  = current intrinsic matrix in this scale

cur_tmp = imresize(tmp, scale);
cur_img = imresize(img, scale);
cur_dim = dim;
[cur_dim.tmp.h, cur_dim.tmp.w, ~] = size(cur_tmp);
[cur_dim.img.h, cur_dim.img.w, ~] = size(cur_img);
cur_in_mat = eye(4);
offset = 0.5;
cur_in_mat(1, 1) = in_mat(1, 1) * scale;
cur_in_mat(2, 2) = in_mat(2, 2) * scale;
cur_in_mat(1, 3) = (in_mat(1, 3) - offset) * scale + offset;
cur_in_mat(2, 3) = (in_mat(2, 3) - offset) * scale + offset;

area = (2*cur_in_mat(1, 1)*cur_dim.tmp_real_w) * (2*cur_in_mat(2, 2)*cur_dim.tmp_real_h) / tz_square;
length = sqrt(area);
TV = calTotalVariation(cur_tmp(:,:,1), area);
while (TV > 8.42*length) % 8.42 is obtained emperically
    cur_tmp = imgaussfilt(cur_tmp, 1, 'Padding', 'symmetric');
	cur_img = imgaussfilt(cur_img, 1, 'Padding', 'symmetric');
    TV = calTotalVariation(cur_tmp(:,:,1), area);
end
