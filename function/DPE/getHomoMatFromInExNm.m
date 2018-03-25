function H = getHomoMatFromInExNm(in_mat, ex_mat, nm_mat)
% Get homography matrix with intrinsic matrix and rigid-transformation pose
%
% Usage:
%   H = getHomoMatFromInExNm(in_mat, ex_mat, nm_mat)
%
% Inputs:
%   in_mat = intrinsic matrix
%   ex_mat = extrinsic matrix
%   nm_mat = normalization matrix
%
% Output:
%   H = 3x3 homography transformation matrix

mat = in_mat * ex_mat;
H = [mat(1:3,1:2), mat(1:3,4)]*nm_mat;
H = H/H(3,3);
