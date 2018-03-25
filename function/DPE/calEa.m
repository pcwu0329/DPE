function Ea = calEa(img, tmp_pxls, x, y, in_mat, ex_mat)
% Calculate Ea score between I and warping T
%
% Usage:
%   Ea = calEa(img, tmp_pxls, x, y, in_mat, ex_mat)
%
% Inputs:
%   img      = ih*iw*3 camera frame
%   tmp_pxls = n*3 valid template region
%   x        = n*1 valid x coordinates
%   y        = n*1 valid y coordinates
%   in_mat   = 4*4 intrinsic matrix
%   ex_mat   = 4*4 extrinsic matrix
%
% Output:
%   Ea = Ea score

H = getHomoMatFromInExNm(in_mat, ex_mat, eye(3));
img_pxls = getMappedPixels(img, x, y, H, 'linear');
valid_indices = ~isnan(img_pxls(:,1));
errors = sum(abs(img_pxls(valid_indices,:) - tmp_pxls(valid_indices,:))) / sum(valid_indices);
Ea = errors(1)*0.5 + errors(2)*0.25 + errors(3)*0.25;
