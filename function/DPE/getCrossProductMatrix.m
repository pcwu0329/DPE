function v_x = getCrossProductMatrix(v)
% Get cross product matrix from a 3-vector
%
% Usage:
%   v_x = getCrossProductMatrix(v)
%
% Inputs:
%   v   = 3-d vector
%
% Outputs:
%   v_x = cross product matrix

v_x = [0, -v(3), v(2); v(3), 0, -v(1); -v(2), v(1), 0];
