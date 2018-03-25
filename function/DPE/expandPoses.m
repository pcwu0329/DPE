function expanded_poses = expandPoses(poses, bounds, steps, multiple, tmp_real_w, tmp_real_h)
% Expand the pose set 
%
% Usage:
%   expanded_poses = expandPoses(poses, bounds, steps, multiple, tmp_real_w, tmp_real_h)
%
% Inputs:
%   poses      = pose set
%   bounds     = pose boundaries 
%   steps      = steps for creating set (refer to createSet_mex for more details)
%   multiple    = expanded multiple of each pose
%   tmp_real_w = half template width in the real world lenght unit
%   tmp_real_h = half template height in the real world lenght unit
%
% Outputs:
%   expanded_poses = expanded pose set

num_poses = size(poses, 2);

% random vectors in {-1,0,1}^6
% instead of using all the 3^6 = 729 poses, here we just select a portion of them
randvec = floor(3*rand(6, multiple*num_poses)-1);
expanded = repmat(poses,[1, multiple]);
ranges = [steps.rz0; steps.rx; steps.rz1; steps.tx; steps.ty; steps.tz];
addvec = repmat(ranges, [1, multiple*num_poses]);

weight = expanded(6,:) - norm([tmp_real_w, tmp_real_h]) .* sin(expanded(2,:));

% calculate steps according to nearby pose

pos_rx = 1./addvec(2,:).*(asin(2 - 1./ (1./(2-sin(expanded(2,:))) + addvec(2,:))) - expanded(2,:));
neg_rx = 1./addvec(2,:).*(expanded(2,:) - asin(2 - 1./ (1./(2-sin(expanded(2,:))) - addvec(2,:))));
addvec(2,:) = addvec(2,:).*((randvec(2,:) >= 0) .* pos_rx + (randvec(2,:) < 0).* neg_rx);
addvec(4,:) = addvec(4,:).*weight;
addvec(5,:) = addvec(5,:).*weight;
addvec(6,:) = (randvec(6,:) >= 0) .* addvec(6,:).*(expanded(6,:).^2) ./ (1 - addvec(6,:).*expanded(6,:)) ...
            + (randvec(6,:) <  0) .* addvec(6,:).*(expanded(6,:).^2) ./ (1 + addvec(6,:).*expanded(6,:));

% expand poses
expanded_poses = expanded + randvec.*addvec;

% delete unwanted poses
[~, cols] = find(imag(expanded_poses));
expanded_poses(:, cols) = [];
[~, cols] = find(expanded_poses(2,:) > bounds.rx(2) | expanded_poses(2,:) < bounds.rx(1));
expanded_poses(:, cols) = [];
[~, cols] = find(expanded_poses(6,:) > bounds.tz(2) | expanded_poses(6,:) < bounds.tz(1));
expanded_poses(:, cols) = [];





