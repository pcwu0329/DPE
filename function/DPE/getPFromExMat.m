function p = getPFromExMat(ex_mat)
% Get axis angles and translation vector from extrinsic matrix
%
% Usage:
%   p = getPFromExMat(ex_mat)
%
% Inputs:
%   ex_mat = 4x4 (or 3x4) extrinsic matrix
%
% Output:
%   p = [rx, ry, rz, tx, ty, tz]

r = getPFromRotMat(ex_mat(1:3, 1:3));
p = [r; ex_mat(1,4); ex_mat(2,4); ex_mat(3,4)];


