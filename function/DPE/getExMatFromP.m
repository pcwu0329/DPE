function ex_mat = getExMatFromP(p)
% Get extrinsic matrix from axis angles and translation vector
%
% Usage:
%   ex_mat = getExMatFromP(p)
%
% Inputs:
%   p = [rx, ry, rz, tx, ty, tz]
%
% Output:
%   ex_mat = 4x4 extrinsic matrix

ex_mat = zeros(4);
ex_mat(4,4) = 1;
ex_mat(1:3, 1:3) = getRotMatFromP(p(1:3));
ex_mat(1:3, 4) = [p(4); p(5); p(6)];
