function is_valid = checkValidity(tf_mat, Iw, Ih, tmp_real_w, tmp_real_h)
% Calculate the validity of transformation matrix
%
% Usage:
%   is_valid = checkValidity(tf_mat, Iw, Ih, tmp_real_w, tmp_real_h)
%
% Inputs:
%   tf_mat      = transformation matrix
%   Iw          = image width
%   Ih          = image height
%   tmp_real_w  = real template (half) width
%   tmp_real_h  = real template (half) height
%
% Output:
%   is_valid = if the transformation matrix is valid

coors = tf_mat * [-tmp_real_w, tmp_real_w, tmp_real_w, -tmp_real_w;
                  -tmp_real_h, -tmp_real_h, tmp_real_h, tmp_real_h;
                            0,           0,          0,          0;
                            1,           1,          1,          1];
coors(1,:) = coors(1,:)./coors(3,:);
coors(2,:) = coors(2,:)./coors(3,:);

% reject transformations which make marker too small
two_area =  (coors(1,1) - coors(1,2)) * (coors(2,1) + coors(2,2)) ...
          + (coors(1,2) - coors(1,3)) * (coors(2,2) + coors(2,3)) ...
          + (coors(1,3) - coors(1,4)) * (coors(2,3) + coors(2,4)) ...
          + (coors(1,4) - coors(1,1)) * (coors(2,4) + coors(2,1));
area = abs(two_area/2);
minx = min(coors(1,:));
maxx = max(coors(1,:));
miny = min(coors(2,:));
maxy = max(coors(2,:));

margin = 1;
area_thres = 0.01*Iw*Ih;
if (area > area_thres && (minx >= margin+1) && (maxx <= Iw-margin) && (miny >= margin+1) && (maxy <= Ih-margin))
    is_valid = true;
else
    is_valid = false;
end
