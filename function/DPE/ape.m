function ex_mat = ape(tmp, img, in_mat, min_dim, min_tz, max_tz, epsilon, delta, prm_lvls, photo_inva, need_compile, verbose)
% The proposed approximated pose estimation method with compilation
%
% Usage:
%   ex_mat = ape(tmp, img, in_mat, min_dim, min_tz, max_tz, epsilon, delta, prm_lvls, photo_inva, need_compile, verbose)
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
%   need_compile = need to compile the mex file
%   verbose      = show the state of the method
%
% Output:
%   ex_mat       = 4*4 estimated extrinsic matrix

if ~exist('need_compile','var')
    need_compile = 0;
end
if need_compile == 1
    CompileMex;
end

% set default values for optional variables
if ~exist('min_tz','var')
	min_tz = 3;
end
if ~exist('max_tz','var')
	max_tz = 8;
end
if ~exist('epsilon','var')
	epsilon = 0.25;
end
if ~exist('delta','var')
	delta = 0.15;
end
if ~exist('photo_inva','var')
	photo_inva = 0;
end
if ~exist('verbose','var')
	verbose = 0;
end

% ensure the data type of images is double
if ~isa(img,'double') || ~isa(tmp,'double')
	error('img and tmp should both be of class ''double'' (in the range [0,1])');
end

% Pre-calculation
t1 = tic;
[tmp_ycbcr, img_ycbcr, bounds, steps, dim] = preCal(tmp, img, min_dim, min_tz, max_tz, epsilon);
if (verbose)
    fprintf('pre-time: %f\n', toc(t1));
end
  
% Coarse-to-fine pose estimation
t2 = tic;
[ex_mat, ~, ~] = coarseToFinePoseEstimation(tmp_ycbcr, img_ycbcr, in_mat, bounds, steps, dim, epsilon, delta, prm_lvls, photo_inva, verbose);
if (verbose)
    fprintf('post-time: %f\n', toc(t2));
end


