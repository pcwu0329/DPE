function ex_mats = getAmbiguousExMats(ex_mat, tmp_real_w, tmp_real_h)
% Calculate ambiguous extrinsic matrices from an initial one
%
% Usage:
%   ex_mats = getAmbiguousExMats(ex_mat, dim)
%
% Inputs:
%   ex_mat     = extrinsic matrix (3*4 or 4*4)
%   tmp_real_w = template width in real unit
%   tmp_real_h = template height in real unit
%
% Outputs:
%   poses = ambiguous extrinsic matrices (4*4*n)

% addpath(genpath('../OPnP'));
tgt = [-tmp_real_w,tmp_real_w,tmp_real_w,-tmp_real_w;
       -tmp_real_h,-tmp_real_h,tmp_real_h,tmp_real_h;
       0,0,0,0;
       1,1,1,1];
src = ex_mat*tgt;
src(1,:) = src(1,:)./src(3,:);
src(2,:) = src(2,:)./src(3,:);
[RR, tt, ~, flag] = OPnP(tgt(1:3,:), src(1:2,:));
if (flag)
    ex_mats = ex_mat;
else
    num = min(size(RR, 3), 2);
    ex_mats = zeros(4, 4, num);
    for i = 1:num
        ex_mats(:, :, i) = eye(4);
        ex_mats(1:3, 1:4, i) = [RR(:, :, i), tt(:, i)];
    end
end



