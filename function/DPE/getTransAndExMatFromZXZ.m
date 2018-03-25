function [ex_mat, trans_mat] = getTransAndExMatFromZXZ(pose, in_mat)
% Calculate the extrinsic and tranformation matrix from pose [rz0; rx; rz1; tx; ty; tz]
%
% Usage:
%   [ex_mat, trans_mat] = getTransAndExMatFromZXZ(pose, in_mat)
%
% Inputs:
%   pose   = [rz0; rx; rz1; tx; ty; tz]
%   in_mat = intrinsic matrix
%
% Outputs:
%   ex_mat    = extrinsic matrix
%   trans_mat = in_mat * ex_mat

rz0 = pose(1);
rx  = pose(2);
rz1 = pose(3);
tx  = pose(4);
ty  = pose(5);
tz  = pose(6);

% Calculate rotation matrix R and translation vector t
R = rotz(rad2deg(rz0))*rotx(rad2deg(pi + rx))*rotz(rad2deg(rz1));
t = [tx; ty; tz];

% Calculate extrinsic matrix
ex_mat = eye(4);
ex_mat(1:3, 1:4) = [R, t];

% Calculate transformation matrix 
trans_mat = in_mat * ex_mat;



