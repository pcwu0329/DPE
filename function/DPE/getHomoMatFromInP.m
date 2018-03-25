function H = getHomoMatFromInP(in_mat, p)
% Get homography matrix from intrinsic matrix and 6 d.o.f. parameters
%
% Usage:
%   H = getHomoMatFromInP(in_mat, p)
%
% Inputs:
%   in_mat = intrinsic matrix
%   p      = [rx, ry, rz, tx, ty, tz]
%
% Output:
%   H = 3x3 homography transformation matrix

M_4_4 = in_mat * getExMatFromP(p);
H = [M_4_4(1:3,1:2), M_4_4(1:3,4)];
H = H/H(3,3);
