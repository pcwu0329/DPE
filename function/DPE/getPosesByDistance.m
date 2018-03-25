function [good_poses, percentage, too_high_percentage, thresh] = getPosesByDistance(poses, best_Ea, epsilon, distances)
% Get good poses from all candidates
%
% Usage:
%   [good_poses, percentage, too_high_percentage, thresh] = getPosesByDistance(poses, best_Ea, epsilon, distances, verbose)
%
% Inputs:
%   poses     = candidate poses (6*n)
%   best_Ea   = the smallest Ea value with the best pose candidate
%   epsilon   = delone set parameter
%   distances = Ea values within poses
%
% Outputs:
%   good_poses          = poses with smaller Ea
%   percentage          = ratio between good poses and all poses
%   too_high_percentage = check if the ratio is too high, which means distinguishable
%   thresh              = determined distance threshold

thresh = best_Ea + getThreshPerEpsilon(epsilon);
good_poses = poses(:, distances <= thresh);
num_poses = size(good_poses, 2);
percentage = num_poses/size(poses, 2);
too_high_percentage = (percentage > 0.1);
% reduce the size of pose set to prevent from out of memory
while (num_poses > 27000)
    thresh = thresh * 0.99;
    if thresh < best_Ea
        thresh = mean([thresh / 0.99, best_Ea]);
    end
    good_poses = poses(:, distances <= thresh);
    num_poses = size(good_poses, 2);
end
if (num_poses == 0)
    [~, index] = min(distances);
    good_poses = poses(:, index);
end



