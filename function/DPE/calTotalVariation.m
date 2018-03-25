function TV = calTotalVariation(tmp, area)
% Calculate the total variation of template in camera frame
%
% Usage:
%   TV = calTotalVariation(tmp, area)
%
% Inputs:
%   tmp    = template image (in gray scale format)
%   area   = area of the projected template area in camera frame
%
% Outputs:
%   TV     = total variation of template in camera frame

local_max = ordfilt2(tmp, 9, true(3));
local_min = ordfilt2(tmp, 1, true(3));
variations = max(local_max - tmp, tmp - local_min);
TV = mean2(variations) * area;
