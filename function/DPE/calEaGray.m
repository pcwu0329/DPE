function Ea = calEaGray(img, tmp_pxls, x, y, in_mat, ex_mat)
% Calculate Ea score between I and warping T
%
% Usage:
%   Ea = calEa(img, tmp_pxls, x, y, in_mat, ex_mat)
%
% Inputs:
%   img      = ih*iw camera frame
%   tmp_pxls = n*1 valid template region
%   x        = n*1 valid x coordinates
%   y        = n*1 valid y coordinates
%   in_mat   = 4*4 intrinsic matrix
%   ex_mat   = 4*4 extrinsic matrix
%
% Output:
%   Ea = Ea score

H = getHomoMatFromInExNm(in_mat, ex_mat, eye(3));
img_pxls = reshape(getMappedPixels(img(:,:,1), x, y, H, 'linear'), [], 1);
valid_indices = ~isnan(img_pxls(:));
Ea = sum(abs(img_pxls(valid_indices) - tmp_pxls(valid_indices))) / sum(valid_indices);
