function [new_tmp, new_img, dim] = preCalForRefine(tmp, img, min_dim)
% Pre-calculate the necessary variables for pose efinement process
%
% Usage:
%   [new_tmp, new_img, dim] = preCalForRefine(tmp, img, in_mat, min_dim, verbose)
%
% Inputs:
%   tmp     = original template image
%   img     = original camera frame
%   min_dim = half of the template side length (the shorter one) in the real world
%
% Outputs:
%   tmp = template image after transforming to YCbCr, and normalization
%   img = camera frame after transforming to YCbCr, and normalization
%   dim = dimension variables

% dimensions of images
[th, tw, ~] = size(tmp);
[ih, iw, ~] = size(img);
dim.tmp.w = tw;
dim.tmp.h = th;
dim.img.w = iw;
dim.img.h = ih;

% search range in pose domain
dim.tmp_real_w = tw/min(tw, th)*min_dim;
dim.tmp_real_h = th/min(tw, th)*min_dim;

% rgb to ycbcr
new_tmp = rgb2ycbcrNormalization(tmp);
new_img = rgb2ycbcrNormalization(img);
