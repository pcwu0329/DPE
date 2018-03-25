function ex_mat = apeCuda(tmp, img, fx, fy, cx, cy, min_dim, min_tz, max_tz, ...
                          epsilon, prm_lvls, photo_inva, verbose)                    
% Approximate Pose Estimation
%
% Usage:
%   ex_mat = apeCuda(tmp, img, fx, fy, cx, cy, min_dim, min_tz, max_tz, ...
%                    epsilon, prm_lvls, photo_inva, verbose);                 
%
% Inputs:
%   tmp        = template image (uint8, m*n*3)
%   img        = camera frame (uint8, m*n*3)
%   fx         = focal length along the X-axis
%   fy         = focal length along the Y-axis
%   cx         = x coordinate of the camera's principle point 
%   cy         = y coordinate of the camera's principle point 
%   min_dim    = half the length of the shorter side (real dimension)
%   min_tz     = lower bound of translation z
%   max_tz     = upper bound of translation z
%   epsilon    = delone set parameter
%   prm_lvls   = pyramid levels
%   photo_inva = need to be photometric invariant
%   verbose    = show the state of the method
%
% Outputs:
%   ex_mat = 4*4 estimated extrinsic matrix

try
    % check if opencv_world320.dll exists
    existence = exist(['apeCudaMex.', mexext], 'file');
    if ~existence, CompileApeCudaMex; end

    ex_mat = apeCudaMex(tmp, img, fx, fy, cx, cy, min_dim, min_tz, max_tz, ...
                        epsilon, int32(prm_lvls), photo_inva, verbose);
catch ME
    warning(ME.message);
    ex_mat = zeros(4,4);
end
