function [new_tmp, new_img, bounds, steps, dim] = preCal(tmp, img, min_dim, min_tz, max_tz, epsilon)
% Pre-calculate the necessary variables
%
% Usage:
%   [new_tmp, new_img, bounds, steps, dim] = preCal(tmp, img, min_dim, min_tz, max_tz, epsilon)
%
% Inputs:
%   tmp     = original template image
%   img     = original camera frame
%   min_dim = half of the template side length (the shorter one) in the real world
%   min_tz  = minimum translation z value
%   max_tz  = maximum translation z value
%   epsilon = delone set parameter
%
% Outputs:
%   new_tmp = template image after bluring, transforming to YCbCr, and normalization
%   new_img = camera frame after bluring, transforming to YCbCr, and normalization
%   bounds  = pose boundaries 
%   steps   = steps for creating set (refer to createSet_mex for more details)
%   dim     = dimension variables

% dimensions of images
[th, tw, ~] = size(tmp);
[ih, iw, ~] = size(img);
dim.tmp.w = tw;
dim.tmp.h = th;
dim.img.w = iw;
dim.img.h = ih;

% search range in pose domain
min_rz = -pi;
max_rz = pi;
min_rx = 0;
max_rx = pi*(80/180);
dim.tmp_real_w = tw/min(tw, th)*min_dim;
dim.tmp_real_h = th/min(tw, th)*min_dim;

% calculate steps and bounds
% we want to fit the constraint O(epsilon * (fx*dim/mean_tz))
bounds.rz = [min_rz, max_rz];
bounds.rx = [min_rx, max_rx];
bounds.tz = [min_tz, max_tz];
mean_tz = sqrt(min_tz*max_tz);
% the original steps of rz0 and rz1 are delta*sqrt(2)/mdian_tz * 'tz'
% but we set them to be independent of real tz for the sake of accuracy
% so they should be delta*sqrt(2)/mdian_tz * 'mdian_tz'
steps.rz0 = epsilon*sqrt(2);
steps.rx = epsilon/sqrt(2)/mean_tz;
steps.rz1 = epsilon*sqrt(2);
steps.tx = epsilon/sqrt(2)/mean_tz*2*min_dim;
steps.ty = epsilon/sqrt(2)/mean_tz*2*min_dim;
steps.tz = epsilon/sqrt(2)/mean_tz;

% rgb to ycbcr
new_tmp = rgb2ycbcrNormalization(tmp);
new_img = rgb2ycbcrNormalization(img);
