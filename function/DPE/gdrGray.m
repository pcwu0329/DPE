function [ex_mat_new, Ea] = gdr(img, tmp_valid, x_valid, y_valid, in_mat, ex_mat, pts_num, epsilon_r, epsilon_t, max_iter, scale, verbose, pose_gt)
% Gradient Descent Refinement - Forward Additive for Rigid Transformation
%
% Usage:
%   [ex_mat_new, Ea] = gdr(img, tmp_valid, x_valid, y_valid, in_mat, ex_mat, pts_num, epsilon_r, epsilon_t, max_iter, verbose, pose_gt)
%
% Inputs:
%   img        = iw*ih*3 camera frame
%   tmp_valid  = n*3 valid template region
%   x_valid    = n*1 valid x coordinates
%   y_valid    = n*1 valid y coordinates
%   in_mat     = 4*4 intrinsic matrix
%   ex_mat     = 4*4 extrinsic matrix
%   pts_num    = number of sampling points
%   epsilon_r  = rotation difference in degree
%   epsilon_t  = translation difference in ratio
%   max_iter   = max iteration times
%   scale      = img_scale
%   verbose    = print message or not
%   pose_gt    = ground truth pose
%
% Outputs:
%   ex_mat_new = updated ex_mat
%   Ea         = Ea score

[Iu_y, Iv_y] = imgradientxy(img(:,:,1));
Iu_y = Iu_y * 0.125;
Iv_y = Iv_y * 0.125;
I_size = size(img);

p = getPFromExMat(ex_mat);
diff_r = 1e10;
diff_t = 1e10;

% sub-samping (we do not use all the points)
total_num = size(tmp_valid, 1);
if pts_num > 0 && pts_num < total_num
    T_w = tmp_valid(1:pts_num,1);
    x = x_valid(1:pts_num);
    y = y_valid(1:pts_num);
else
    T_w = tmp_valid(:, 1);
    x = x_valid;
    y = y_valid;
end

num = numel(x);
iter = 0;
Ea = calEaGray(img, tmp_valid, x_valid, y_valid, in_mat, ex_mat);

if verbose
    [err_r, err_t] = calPoseDiff(pose_gt, p);
    fprintf(1,'Initial Condition: epsilon_r = %.6f, epsilon_t = %.6f, Ea = %.6f, err_r = %.6f, err_t = %.6f\n', ...
        epsilon_r, epsilon_t, Ea, err_r, err_t);
end

% if the current camera frame is a smaller one, then we can allow more iterations
max_iter = max_iter / scale;
while (diff_r > epsilon_r || diff_t > epsilon_t) && iter < max_iter
    
    % --- Step 1: Compute Jfa ---
    uv = [x, y, ones(num,1)] * getHomoMatFromInP(in_mat, p).';
    u = uv(:, 1)./uv(:, 3);
    v = uv(:, 2)./uv(:, 3);
    valid_I_indices = (u>=1 & u<= I_size(2) & v>=1 & v<=I_size(1));
    nu = u(valid_I_indices);
    nv = v(valid_I_indices);
    nx = x(valid_I_indices);
    ny = y(valid_I_indices);
    tic
    Iu_y_w = interp2(Iu_y, nu, nv, 'linear');
    Iv_y_w = interp2(Iv_y, nu, nv, 'linear');   
    J = getJfa(Iu_y_w, Iv_y_w, nx, ny, p, in_mat);
    
    % --- Step 2: Compute H ---
    H = J.'*J;

    % --- Step 3: Compute delta p
    I_w = interp2(img(:,:,1), nu, nv, 'linear');
    E = T_w(valid_I_indices) - I_w;
    JtE = J.'*E(:);
    delta_p = H\JtE;
    
    % --- Step 4: Backtracking line search
    alpha = 0.5;
    c = 1e-4;
    new_Ea = calEaGray(img, tmp_valid, x_valid, y_valid, in_mat, getExMatFromP(p+delta_p));
    [diff_r, diff_t] = calPoseDiff(p, p + delta_p);
    while (new_Ea > Ea + (-2.0 * c / num * dot(delta_p, JtE))) && (diff_r > epsilon_r || diff_t > epsilon_t)
        delta_p = delta_p * alpha;
        new_Ea = calEaGray(img, tmp_valid, x_valid, y_valid, in_mat, getExMatFromP(p+delta_p));
        [diff_r, diff_t] = calPoseDiff(p, p + delta_p);
    end
    Ea = new_Ea;
    
    % --- Step 5: Update the parameters ---
    p = p + delta_p;
    iter = iter + 1;
    
    % === print message ===
    if verbose
        [err_r, err_t] = calPoseDiff(pose_gt, p);
        fprintf(1,'Iteration: %3d, Ea = %.6f, diff_r = %.6f, diff_t = %.6f, err_r = %.6f, err_t = %.6f\n', ...
            iter, Ea, diff_r, diff_t, err_r, err_t);
    end
end
ex_mat_new = getExMatFromP(p);
