function ex_mat = prCuda(tmp, img, ex_mat_ape, fx, fy, cx, cy, ...
                         min_dim, tmp_real_w, tmp_real_h, prm_lvls, photo_inva, verbose)
% Pose Refinement
%
% Usage:
%   ex_mat = prCuda(tmp, img, ex_mat_ape, fx, fy, cx, cy, ...
%                   min_dim, tmp_real_w, tmp_real_h, prm_lvls, photo_inva, verbose);
%
% Inputs:
%   tmp        = template image (uint8)
%   img        = camera frame (uint8)
%   ex_mat_ape = approximately estimated extrinsic matrices (4*4)
%   fx         = focal length along the X-axis
%   fy         = focal length along the Y-axis
%   cx         = x coordinate of the camera's principle point 
%   cy         = y coordinate of the camera's principle point 
%   min_dim    = half the length of the shorter side (real dimension)
%   tmp_real_w = template width in real unit
%   tmp_real_h = template height in real unit
%   prm_lvls   = pyramid levels
%   photo_inva = need to be photometric invariant
%   verbose    = show the state of the method
%
% Outputs:
%   ex_mat = refined extrinsic matrix (4*4)

try
    % check if mex files exist
    existence = exist(['prCudaMex.', mexext], 'file');
    if ~existence, CompilePrCudaMex; end

    amb_ex_mats = getAmbiguousExMats(ex_mat_ape, tmp_real_w, tmp_real_h);
    num = size(amb_ex_mats, 3);
    ex_mats = zeros(4, 4*num);
    for i = 1:num
        ex_mats(1:4, i*4-3:i*4) = amb_ex_mats(:,:,i);
    end
    ex_mat = prCudaMex(tmp, img, ex_mats, fx, fy, cx, cy, min_dim, int32(prm_lvls), photo_inva, verbose);
catch ME
    warning(ME.message);
    ex_mat = zeros(4,4);
end
