 function r = getPFromRotMat(R)
% Get the axis angles from a rotation Matrix
%
% Usage:
%   function r = getPFromRotMat(R)
%
% Inputs: 
%   R = 3x3 rotation matrix
%
% Outputs: 
%   r = 3-vector of the axis angles
%
% Explanation:
%   a = acos((trace(R)-1)/2)
%   r = (a/2sin(a)) * [R(3, 2)-R(2, 3), R(1, 3)-R(3, 1), R(2, 1)-R(1, 2)]

EPS = 1e-25;
a = acos((trace(R)-1)/2);
if a < EPS
    r = 0.5 * [R(3, 2)-R(2, 3); R(1, 3)-R(3, 1); R(2, 1)-R(1, 2)];
elseif a > (pi-EPS)
    S = 0.5*(R-eye(3));
    b = sqrt(S(1, 1)+1);
    c = sqrt(S(2, 2)+1);
    d = sqrt(S(3, 3)+1);
    if b > EPS
        c = S(2, 1) / b;
        d = S(3, 1) / b;
    elseif c > EPS
        b = S(1, 2) / c;
        d = S(3, 2) / c;
    else
        b = S(1, 3) / d;
        c = S(2, 3) / d;
    end
    r = [b; c; d];
else
    r = (a/2/sin(a)) * [R(3, 2)-R(2, 3); R(1, 3)-R(3, 1); R(2, 1)-R(1, 2)];
end











