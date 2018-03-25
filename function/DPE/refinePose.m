function [ex_mat, ex_mats] = refinePose(tmp, img, in_mat, ex_mat_ini, dim, prm_lvls, photo_inva, verbose)
% The proposed pose refinement scheme with compilation
%
% Usage:
%   [ex_mat, ex_mats] = refinePose(tmp, img, in_mat, ex_mat_ini, dim, photo_inva, verbose)
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
%
% Output:
%   ex_mat  = estimated extrinsic matrix
%   ex_mats = estimated extrinsic matrix candidates

% normalized matrix of template coordinate
nm_mat = eye(3);
x_center = 0.5*(dim.tmp.w + 1);
y_center = 0.5*(dim.tmp.h + 1);

nm_mat(1, 1) = 2 * dim.tmp_real_w / dim.tmp.w;
nm_mat(1, 3) = -x_center * nm_mat(1, 1);
nm_mat(2, 2) = -2 * dim.tmp_real_h / dim.tmp.h;
nm_mat(2, 3) = -y_center * nm_mat(2, 2);

% get two extrinsic matrices
ex_mats = getAmbiguousExMats(ex_mat_ini, dim.tmp_real_w, dim.tmp_real_h);

% Gradient descent refinement
num = size(ex_mats, 3);
Eas = ones(1, num);
max_iter = 10;

% normalize intensity
if photo_inva
    mean_imgs = zeros(num, 1);
    sig_imgs = zeros(num, 1);
    [X, Y] = meshgrid(1:dim.tmp.w, 1:dim.tmp.h);
    tmp_y = reshape(tmp(:,:,1), [], 1);
    mean_tmp = mean(tmp_y);
    sig_tmp = std(tmp_y);
    for i = 1:num
        H = getHomoMatFromInExNm(in_mat, ex_mats(:,:,i), nm_mat);
        img_y = getMappedPixels(img(:,:,1), X(:), Y(:), H, 'linear');
        mean_imgs(i) = mean(img_y);
        sig_imgs(i) = std(img_y);
    end
    mean_img = mean(mean_imgs);
    sig_img = mean(sig_imgs);
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
while scale <= 1 / 2
    small_img = imresize(img, scale);
    small_in_mat(1:2, 1:3) = (in_mat(1:2, 1:3)-offset)*scale + offset;
    for i = 1:num
        if Eas(i) ~= 2
            H = getHomoMatFromInExNm(small_in_mat, ex_mats(:,:,i), nm_mat);
            [tmp_valid, x_valid, y_valid, validness] = calValidCoors(small_img, tmp, H, nm_mat);
            if validness
                [ex_mats(:,:,i), Eas(i)] = gdr(small_img, tmp_valid, x_valid, y_valid, small_in_mat, ex_mats(:,:,i), pts_num, epsilon_r, epsilon_t, max_iter, scale, verbose);
            else
                Eas(i) = 2;
            end
        end
    end
    scale = scale * 2;
    epsilon_r = epsilon_r / 4;
    epsilon_t = epsilon_t / 4;
    pts_num = pts_num * 4;
end

% get the one with smaller Ea
if sum(Eas) == num * 2
    ex_mat = ex_mat_ini;
    return
elseif (num == 1 || Eas(1) <= Eas(2))
    ex_mat = ex_mats(:, :, 1);
else
    if verbose
        fprintf('switch to the other pose!\n');
    end
    ex_mat = ex_mats(:, :, 2);
end

% final refinement
H = getHomoMatFromInExNm(in_mat, ex_mat, nm_mat);
[tmp_valid, x_valid, y_valid, validness] = calValidCoors(img, tmp, H, nm_mat);
if validness
    [ex_mat, ~] = gdr(img, tmp_valid, x_valid, y_valid, in_mat, ex_mat, pts_num, epsilon_r, epsilon_t, max_iter, 1, verbose);
else
    ex_mat = ex_mat_ini;
end
