function [tmp_valid, x_valid, y_valid, validness] = calValidCoors(img, tmp, homo, nm_mat)
% Calculate the valid template region appeared in the camera image
%
% Usage:
%   [tmp_valid, x_valid, y_valid] = calValidCoors(img, tmp, homo, nm_mat)
%
% Inputs:
%   img       = iw*ih*3 camera frame
%   tmp       = tw*th*3 template image
%   homo      = 3*3 homography transformation matrix
%   nm_mat    = normalization matrix
%   validness = check if the template area is large enough
%
% Output:
%   tmp_valid = n*3 (ycbcr) valid template pixels 
%   x_valid   = n*1 valid x coordinates
%   y_valid   = n*1 valid y coordinates

[ih, iw, ~] = size(img);
[th, tw, ~] = size(tmp);

boundary = [1,1,tw,tw;1,th,th,1]; 
corners = homo*[boundary;1,1,1,1];
corners(1,:) = corners(1,:)./corners(3,:);
corners(2,:) = corners(2,:)./corners(3,:);
U_max = min(ceil(max(corners(1,:))), iw);
V_max = min(ceil(max(corners(2,:))), ih);
U_min = max(floor(min(corners(1,:))), 1);
V_min = max(floor(min(corners(2,:))), 1);
[U, V] = meshgrid(U_min:U_max, V_min:V_max);
XY = homo\[reshape(U,1,[]); reshape(V,1,[]); ones(1,numel(U))];
X = XY(1,:)./XY(3,:);
Y = XY(2,:)./XY(3,:);
% resize the template image so that it would have the similar size with the projected one
area = polyarea(corners(1,:), corners(2, :));
if area < 16 || isnan (area)
    tmp_valid = [];
    x_valid = [];
    y_valid = [];
    validness = false;
    return
end
area_ori = polyarea(boundary(1,:), boundary(2, :));
scale = sqrt(area / area_ori);
if scale > 4
    tmp_valid = [];
    x_valid = [];
    y_valid = [];
    validness = false;
    return
end
%tmp = imresize(imresize(tmp, scale), [th, tw], 'bilinear');
tmp = imgaussfilt(tmp, scale*0.5, 'Padding', 'symmetric');
tmp_patch_y = interp2(tmp(:,:,1), X, Y, 'linear');
tmp_patch_cb = interp2(tmp(:,:,2), X, Y, 'linear');
tmp_patch_cr = interp2(tmp(:,:,3), X, Y, 'linear');
index = ~isnan(tmp_patch_y);
if sum(index) < 16
    tmp_valid = [];
    x_valid = [];
    y_valid = [];
    validness = false;
    return
end
tmp_valid = zeros(sum(index), 3);
tmp_valid(:, 1) = tmp_patch_y(index);
tmp_valid(:, 2) = tmp_patch_cb(index);
tmp_valid(:, 3) = tmp_patch_cr(index);
x_valid = X(index).';
y_valid = Y(index).';
x_valid = nm_mat(1, 1) * x_valid + nm_mat(1, 3);
y_valid = nm_mat(2, 2) * y_valid + nm_mat(2, 3);

% sort the values by gradient magnitude
gra_mag = interp2(imgradient(tmp(:, :, 1)), X, Y);
gra_mag_valid = gra_mag(index);
[~, inds] = sort(gra_mag_valid, 'descend');
tmp_valid = tmp_valid(inds, :);
x_valid = x_valid(inds);
y_valid = y_valid(inds);
validness = true;
