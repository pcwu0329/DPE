function ex_mat = refineOnePose(tmp, img, in_mat, ex_mat_ini, dim, prm_lvls, photo_inva, verbose, pose_gt)
% The proposed pose refinement scheme for only one pose
%
% Usage:
%   ex_mat = refineOnePose(tmp, img, in_mat, ex_mat_ini, dim, photo_inva, verbose, pose_gt)
%
% Input:
%   tmp        = template image
%   img        = camera frame
%   in_mat     = 4*4 camera intrinsic matrix 
%   ex_mat_ini = 4*4 initial extrinsic matrix 
%   dim        = dimension variables
%   prm_lvls   = pyramid levels
%   photo_inva = need to be photometric invariant
%   verbose    = show the state of the method
%   pose_gt    = ground truth pose
%
% Output:
%   ex_mat  = estimated extrinsic matrix

% normalized matrix of template coordinate
nm_mat = eye(3);
x_center = 0.5*(dim.tmp.w + 1);
y_center = 0.5*(dim.tmp.h + 1);

nm_mat(1, 1) = 2 * dim.tmp_real_w / dim.tmp.w;
nm_mat(1, 3) = -x_center * nm_mat(1, 1);
nm_mat(2, 2) = -2 * dim.tmp_real_h / dim.tmp.h;
nm_mat(2, 3) = -y_center * nm_mat(2, 2);

% Gradient descent refinement
max_iter = 10;
ex_mat = ex_mat_ini;

% normalize intensity
if photo_inva
    [X, Y] = meshgrid(1:dim.tmp.w, 1:dim.tmp.h);
    tmp_y = reshape(tmp(:,:,1), [], 1);
    mean_tmp = mean(tmp_y);
    sig_tmp = std(tmp_y);
    H = getHomoMatFromInExNm(in_mat, ex_mat_ini, nm_mat);
    img_y = getMappedPixels(img(:,:,1), X(:), Y(:), H, 'linear');
    mean_img = mean(img_y);
    sig_img = std(img_y);
    alpha = sig_tmp/sig_img;
    beta = -mean_img*alpha + mean_tmp;
    img(:,:,1) = alpha * img(:,:,1) + beta;
end

scale = 1 / (2^(prm_lvls-1));
offset = [0,0, 0.5; 0, 0, 0.5];
small_in_mat = in_mat;
pts_num = 65536;
epsilon_r = 1 / scale / scale / 256;
epsilon_t = 1 / scale / scale / 256;
Ea = 1;
while scale <= 1
    small_img = imresize(img, scale);
    small_in_mat(1:2, 1:3) = (in_mat(1:2, 1:3)-offset)*scale + offset;
    if Ea ~= 2
        H = getHomoMatFromInExNm(small_in_mat, ex_mat, nm_mat);
        [tmp_valid, x_valid, y_valid, validness] = calValidCoors(small_img, tmp, H, nm_mat);
        if validness
            [ex_mat, Ea] = gdr(small_img, tmp_valid, x_valid, y_valid, small_in_mat, ex_mat, pts_num, epsilon_r, epsilon_t, max_iter, scale, verbose, pose_gt);
        else
            Ea = 2;
        end
    end
    scale = scale * 2;
    epsilon_r = epsilon_r / 4;
    epsilon_t = epsilon_t / 4;
    pts_num = pts_num * 4;
end

if verbose
    [err_r, err_t] = calPoseDiff(pose_gt, reshape(ex_mat(1:3, 1:4), 1, 12));
    fprintf(1, 'final condition: err_r = %.6f, err_t = %.6f\n', err_r, err_t);
end


