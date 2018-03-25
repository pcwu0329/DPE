function pose = getZXZFromExMat(ex_mat)
% Calculate the pose [rz0; rx; rz1; tx; ty; tz] from extrinsic matrix
%
% Usage:
%   pose = getZXZFromExMat(ex_mat)
%
% Inputs:
%   ex_mat = extrinsic matrix (3*4 or 4*4)
%
% Outputs:
%   pose = [rz0; rx; rz1; tx; ty; tz]

R = ex_mat(1:3, 1:3);
t = ex_mat(1:3, 4);
rx = pi - acos(R(3,3));
if (rx == 0)
    rz0 = 0;
    rz1 = atan2(-R(1,2), R(1,1));
else
    rz0 = atan2(-R(1,3), R(2,3));
    rz1 = atan2(-R(3,1), -R(3,2));
end
pose = [rz0; rx; rz1; t(1); t(2); t(3)];



