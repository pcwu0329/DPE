function R = getRotMatFromP(r)
% Get rotation matrix from a 3-vector of axis angles
%
% Usage:
%   R = getRotMatFromP(r)
%
% Inputs:
%   w = 3-vector of axis angles
%
% Outputs:
%   R = 3x3 rotation matrix
%
% Explanation:
%   a = |r|
%   W = cross product matrix of r
%   R = I + (sin(a)/a)*W + W^2*((1-cos(a))/a^2

EPS = 1e-25;
I = eye(3);
a = norm(r);
a2 = a * a;
W = getCrossProductMatrix(r);
W2 = W * W;
if a < EPS
    R = I + W + 0.5 * W2;
else
    R = I + W * sin(a) / a + W2 * (1-cos(a)) / a2;
end
