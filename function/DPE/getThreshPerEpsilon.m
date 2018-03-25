function thresh = getThreshPerEpsilon(epsilon)
% Compute the distance threshold according to epsilon
%
% Usage:
%   thresh = GetThreshPerEpsilon(epsilon)
%
% Inputs:
%   epsilon = delone set parameter
%
% Outputs:
%   thresh = distance (Ea) threshold

% The coefficients are computed by linear regression based on synthetic normal dataset
thresh = 0.19 * epsilon + 0.01372;
