function [ex_mat, ex_mats] = dpe(tmp, img, in_mat, min_dim, min_tz, max_tz, epsilon, delta, prm_lvls, photo_inva, verbose)
% The proposed 6 DoF pose estimation method with compilation
%
% Usage:
%   [ex_mat, ex_mats] = dpe(tmp, img, in_mat, min_dim, min_tz, max_tz, epsilon, delta, prm_lvls, photo_inva, verbose)
%
% Input:
%   tmp          = template image (double)
%   img          = camera image (double)
%   in_mat       = 4*4 camera intrinsic matrix
%   min_dim      = length of the shorter side of the target
%   min_tz       = minimum distance between camera and target
%   max_tz       = maximum distance between camera and target
%   epsilon      = initial delone set parameter (defalut: 0.25)
%   delta        = initial random sample parameter (default: 0.15)
%   prm_lvls     = pyramid levels
%   photo_inva   = need to be photometric invariant
%   verbose      = show the state of the method
%
% Output:
%   ex_mat  = estimated extrinsic matrix
%   ex_mats = estimated extrinsic matrix candidates

% check if mex files exist
existence = exist(['createSetMex.', mexext], 'file') && ...
            exist(['poseToTransMatMex.', mexext], 'file') && ...
            exist(['evaluateEaColorMex.', mexext], 'file') && ...
            exist(['evaluateEaInvarMex.', mexext], 'file');
if ~existence, CompileDpeMex; end

% ensure the data type of images is double
if ~isa(img,'double') || ~isa(tmp,'double')
	error('img and tmp should both be of class ''double'' (in the range [0,1])');
end

t1 = tic;
% Pre-calculation
[tmp_ycbcr, img_ycbcr, bounds, steps, dim] = preCal(tmp, img, min_dim, min_tz, max_tz, epsilon);
% Coarse-to-fine pose estimation
[last_ex_mat, ~, ~] = coarseToFinePoseEstimation(tmp_ycbcr, img_ycbcr, in_mat, bounds, steps, dim, epsilon, delta, prm_lvls, photo_inva, verbose);
if (verbose)
    fprintf('[*** Approximation Pose Estimation ***] Runtime: %f seconds\n', toc(t1));
end

% Pose refinement
t2 = tic;
[ex_mat, ex_mats] = refinePose(tmp_ycbcr, img_ycbcr, in_mat, last_ex_mat, dim, 3, photo_inva, verbose);
if (verbose)
    fprintf('[*** Pose Refinement ***] Runtime: %f seconds\n', toc(t2));
end

