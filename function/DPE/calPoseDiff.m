function [diff_r, diff_t] = calPoseDiff(pose1, pose2)
% Get rotation difference (degree) and translation difference (percentage)
%
% Usage:
%   [diff_r, diff_t] = calPoseDiff(pose1, pose2)
%
% Inputs:
%   pose1 = 6*1 pose vector (or 1*12)
%   pose2 = 6*1 pose vector (or 1*12)
%
% Outputs:
%   diff_r = rotation difference in degree
%   diff_t = translation difference in percentage
%
% Explanation:
%   a = |r|
%   W = cross product matrix of r
%   R = I + (sin(a)/a)*W + W^2*((1-cos(a))/a^2

if numel(pose1) == 6
    R1 = getRotMatFromP(pose1(1:3));
    t1 = pose1(4:6);
else
    R1 = reshape(pose1(1:9), 3, 3);
    t1 = pose1(10:12).';
end
if numel(pose2) == 6
    R2 = getRotMatFromP(pose2(1:3));
    t2 = pose2(4:6);
else
    R2 = reshape(pose2(1:9), 3, 3);
    t2 = pose2(10:12).';
end
v = (trace(R2.'*R1)-1)/2;
if v > 1
    v = 2-v;
elseif v < -1
    v = -2-v;
end
diff_r = acosd(v);
diff_t = norm(t2-t1)/norm(t1)*100;      
