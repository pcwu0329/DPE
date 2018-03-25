function ex_mat = calExMat(rz0, rx, rz1, tx, ty, tz)
% Calculate the extrinsic matrix from pose parameters
%
% Usage:
%   ex_mat = calExMat(rz0, rx, rz1, tx, ty, tz)
%
% Inputs:
%   rz0 = rotation z0
%   rx  = rotation x
%   rz1 = rotation z1
%   tx  = translation x
%   ty  = translation y
%   tz  = translation z
%
% Output:
%   ex_mat = extrinsic matrix

ex_mat = eye(4);

% from degree to radian
rz0 = degtorad(rz0);
rx  = degtorad(rx) + pi; 
rz1 = degtorad(rz1);

% calculate rotation matrix
ex_mat(1,1) =  cos(rz0)*cos(rz1) - sin(rz0)*cos(rx)*sin(rz1);
ex_mat(1,2) = -cos(rz0)*sin(rz1) - sin(rz0)*cos(rx)*cos(rz1);
ex_mat(1,3) =  sin(rz0)*sin(rx);
ex_mat(2,1) =  sin(rz0)*cos(rz1) + cos(rz0)*cos(rx)*sin(rz1);
ex_mat(2,2) = -sin(rz0)*sin(rz1) + cos(rz0)*cos(rx)*cos(rz1);
ex_mat(2,3) = -cos(rz0)*sin(rx);
ex_mat(3,1) =  sin(rx)*sin(rz1);
ex_mat(3,2) =  sin(rx)*cos(rz1);
ex_mat(3,3) =  cos(rx);

% fill in the translation parameters
ex_mat(1,4) = tx;
ex_mat(2,4) = ty;
ex_mat(3,4) = tz;
