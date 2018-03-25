function img = blendImage(bg, tmp, H)
% Blend image from background image and warped template image
%
% Usage:
%   img = blendImage(bg, tmp, H)
%
% Inputs:
%   bg  = background image
%   tmp = template image
%   H = 3x3 homography transformation matrix
%
% Output:
%   img = blend image

[I_h, I_w, ~] = size(bg);
[T_h, T_w, ~] = size(tmp);
boundary = [1,1,T_w,T_w;1,T_h,1,T_h]; 
corner = H*[boundary;1,1,1,1];
corner(1,:) = corner(1,:)./corner(3,:);
corner(2,:) = corner(2,:)./corner(3,:);
X_max = ceil(max(corner(1,:)));
Y_max = ceil(max(corner(2,:)));
X_min = floor(min(corner(1,:)));
Y_min = floor(min(corner(2,:)));

if (X_max > I_w || X_min < 1 || Y_max > I_h || Y_min < 1)
    error('X_max = %f, X_min = %f, Y_max = %f, Y_min = %f\n', X_max, X_min, Y_max, Y_min);
end

% ---  super sampling  --- %
[X, Y] = meshgrid(X_min:X_max, Y_min:Y_max);
[h, w] = size(X);
bg_patch = bg(Y_min:Y_max, X_min:X_max, :);
channel_num = size(bg_patch, 3);
sum = zeros(h, w, 3);
rand_x = rand(16, 1) - 0.5;
rand_y = rand(16, 1) - 0.5;
for i = 1:16
    Xx = X + rand_x(i);
    Yy = Y + rand_y(i);
    UV = H\[reshape(Xx, 1,[]); reshape(Yy, 1,[]); ones(1, numel(X))];
    U = reshape(UV(1,:)./UV(3,:), h, w);
    V = reshape(UV(2,:)./UV(3,:), h, w);
    sum_temp = zeros(h, w, 3);
    for j = 1:channel_num
        sum_temp(:,:,j) = interp2(tmp(:,:,j), U, V, 'bicubic');
    end
    index = isnan(sum_temp);
    sum_temp(index) = bg_patch(index);
    sum = sum + sum_temp;
end
img_estimated = sum / 16;
img_estimated(img_estimated > 1) = 1;
img_estimated(img_estimated < 0) = 0;
img = bg;
img(Y_min:Y_max, X_min:X_max, :) = img_estimated;


%[X, Y] = meshgrid(X_min:X_max, Y_min:Y_max);
%[h, w] = size(X);
%UV = H\[reshape(X,1,[]); reshape(Y,1,[]); ones(1,numel(X))];
%U = reshape(UV(1,:)./UV(3,:), h, w);
%V = reshape(UV(2,:)./UV(3,:), h, w);
%I_patch = I(Y_min:Y_max, X_min:X_max, :);
%[~,~,channel_num] = size(I);
%T_patch = zeros(h, w, channel_num);
%for i = 1:channel_num
%    T_patch(:,:,i) = interp2(T(:,:,i), U, V, 'bicubic');
%end
%index = isnan(T_patch);
%T_patch(index) = I_patch(index);
%BI = I;
%BI(Y_min:Y_max, X_min:X_max,:) = 0.5*T_patch+0.5*I_patch;
