function [poses, num_poses] = createEpsilonCoverSet(in_mat, bounds, steps, dim)
% Create the epsilon cover set
%
% Usage:
%   [poses, num_poses] = createEpsilonCoverSet(in_mat, bounds, steps, dim)
%
% Inputs:
%   in_mat = intrinsic matrix
%   bounds = pose boundaries 
%   steps  = steps for creating set
%   dim    = dimension variables
%
% Outputs:
%   poses     = all candidate poses
%   num_poses = number of candidate poses

% call mex function
poses = createSetMex(bounds.rz(1), bounds.rz(2), ...
                     bounds.rx(1), bounds.rx(2), ...
                     bounds.tz(1), bounds.tz(2), ...
                     steps.rz0, steps.rx, steps.rz1, ...
                     steps.tx, steps.ty, steps.tz, ...
                     in_mat(1,1), in_mat(2,2), ...
                     in_mat(1,3)-1, in_mat(2,3)-1, ... % -1 for c++
                     dim.tmp_real_w, dim.tmp_real_h);
num_poses = size(poses, 2);
