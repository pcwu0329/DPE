function [ex_mat, epsilon, steps] = coarseToFinePoseEstimation(tmp, img, in_mat, bounds, steps, dim, epsilon, delta, prm_lvls, photo_inva, verbose)
% Perform the coarse-to-fine pose estimation
%
% Usage:
%   [ex_mat, epsilon, steps] = coarseToFinePoseEstimation(tmp, img, in_mat, bounds, steps, dim, epsilon, delta, prm_lvls, photo_inva, verbose)
%
% Inputs:
%   tmp        = template image (in YCbCr format)
%   img        = camera frame (in YCbCr format)
%   in_mat     = intrinsic matrix
%   bounds     = pose boundaries 
%   steps      = steps for creating set (refer to createSet for more details)
%   dim        = dimension variables
%   epsilon    = delone set parameter
%   delta      = random sample parameter
%   prm_lvls   = pyramid levels
%   photo_inva = need to be photometric invariant
%   verbose    = show the state of the method
%
% Outputs:
%   ex_mat  = 4*4 estimated extrinsic matrix
%   epsilon = last delone set parameter
%   steps   = last steps for creating set

% start coarse-to-fine estimation
num_points = round(10/delta^2);
epsilon_factor = 1 / 1.511;
level = 0;
level_p = 0; % pyramid level (the last level at previous scale)
best_dists = zeros(1 ,8);
total_time = 0;
if (photo_inva)
    c1 = 0.075; c2 = 0.15;
else
    c1 = 0.05; c2 = 0.1;
end

% for multi-scale
scale = 1/(4^(prm_lvls-1));
while scale <= 1
    if (verbose)
        fprintf('pyramid: %f\n', scale);
    end
    [cur_tmp, cur_img, cur_dim, cur_in_mat] = imrescale(tmp, img, scale, in_mat, dim, bounds.tz(1) * bounds.tz(2));
    while (1)
        if level == 0
            % create the epsilon-cover pose set
            poses = createEpsilonCoverSet(cur_in_mat, bounds, steps, cur_dim);
        end
        
        level = level + 1;
        if (verbose)
            fprintf('  -- level %d -- epsilon %.3f', level, epsilon);
        end

        % calculate transformation matrix from poses
        pose_to_trans_mat_start_time = tic;
        area_thres = 0.01 * cur_dim.img.w * cur_dim.img.h;
        [trans_mats, insiders] = poseToTransMatMex(poses, int32(cur_dim.img.h), int32(cur_dim.img.w),...
                                                   cur_in_mat(1,1), cur_in_mat(2,2), cur_in_mat(1,3)-1, cur_in_mat(2,3)-1, ... % -1 for c++
                                                   cur_dim.tmp_real_w, cur_dim.tmp_real_h, ...
                                                   area_thres);
        in_boundary_indices = find(insiders);
        trans_mats = trans_mats(:, in_boundary_indices);
        poses = poses(:, in_boundary_indices);
        num_poses = size(poses, 2);
        pose_to_trans_mat_time = toc(pose_to_trans_mat_start_time);
        if (verbose)
            fprintf(', Number of Poses %d', num_poses);
        end

        % evaluate Ea of all poses
        evaluate_Ea_start_time = tic;      
        xs_mex = randi([0, cur_dim.tmp.w - 1], [1,ceil(num_points*scale)]);
        ys_mex = randi([0, cur_dim.tmp.h - 1], [1,ceil(num_points*scale)]);
        
        if (photo_inva)
            distances = evaluateEaInvarMex(cur_tmp(:,:,1), cur_tmp(:,:,2), cur_tmp(:,:,3), ...
                                           cur_img(:,:,1), cur_img(:,:,2), cur_img(:,:,3), ...
                                           trans_mats, int32(xs_mex), int32(ys_mex), cur_dim.tmp_real_w, cur_dim.tmp_real_h);
        else
            distances = evaluateEaColorMex(cur_tmp(:,:,1), cur_tmp(:,:,2), cur_tmp(:,:,3), ...
                                           cur_img(:,:,1), cur_img(:,:,2), cur_img(:,:,3), ...
                                           trans_mats, int32(xs_mex), int32(ys_mex), cur_dim.tmp_real_w, cur_dim.tmp_real_h);
        end
        evaluate_Ea_time = toc(evaluate_Ea_start_time);
        total_time = total_time + pose_to_trans_mat_time + evaluate_Ea_time;
        
        % get best pose with minimun Ea
        [best_Ea, index] = min(distances);
        [ex_mat, ~] = getTransAndExMatFromZXZ(poses(:, index), in_mat);    
        best_dists(level) = best_Ea;
        if (verbose)
            fprintf(', Evaluation Time: %f, Best Ea %.4f', total_time, best_Ea);   
        end
        
        % early terminate
        level_s = max(level - 3, level_p); % start level
        level_e = level - 1;               % end level
        if (best_Ea < 0.005) || ((scale == 1) && (best_Ea < 0.015)) ||...
           ((level_p > 0) && (level_p ~= level) && (scale == 1) && (best_Ea > mean(best_dists(level_s:level_e))*0.97))
            if (verbose)
                fprintf('\n');
            end
            break;
        end
        
        % select poses within threshold to be in the next round
        [good_poses, percentage, too_high_percentage, ~] = getPosesByDistance(poses, best_Ea, epsilon, distances);
        if (verbose)
            fprintf(', Survived percentage %.1f\n', percentage * 100);
        end
        
        % expand the pose set for next round
        % if the initial pose set is not decent enough, recreate another new epsilon-covering set with smaller epsilon
        if ((level == 1) && ...
           ((too_high_percentage && (best_Ea > c1) && (num_poses < 7500000)) || ((best_Ea > c2) && (num_poses < 5000000))) )
            level = 0;
            factor = 0.9;
            epsilon   = factor * epsilon;
            steps.rz0 = factor * steps.rz0;
            steps.rx  = factor * steps.rx;
            steps.rz1 = factor * steps.rz1;
            steps.tx  = factor * steps.tx;
            steps.ty  = factor * steps.ty;
            steps.tz  = factor * steps.tz;
        else
            epsilon   = epsilon_factor * epsilon;
            steps.rz0 = epsilon_factor * steps.rz0;
            steps.rx  = epsilon_factor * steps.rx;
            steps.rz1 = epsilon_factor * steps.rz1;
            steps.tx  = epsilon_factor * steps.tx;
            steps.ty  = epsilon_factor * steps.ty;
            steps.tz  = epsilon_factor * steps.tz;
            expanded_poses = expandPoses(good_poses, bounds, steps, 80, cur_dim.tmp_real_w, cur_dim.tmp_real_h);
            poses = [good_poses, expanded_poses];
            pixelMaxMovement = epsilon * max(cur_in_mat(1:1),cur_in_mat(2,2)) * max(cur_dim.tmp_real_w,cur_dim.tmp_real_h) * 2 / mean(poses(6,:));
            if (pixelMaxMovement < 1)
                level_p = level + 1;
                break;
            end
        end
    end
    scale = scale * 4;
end






