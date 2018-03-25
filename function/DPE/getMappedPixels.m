function pixels = getMappedPixels(I, x, y, H, method)
% Get the mapped pixels according to input data
%
% Usage:
%   Ea = calEa(I, T, H)
%
% Inputs:
%   I      = iw*ih*l camera frame
%   x      = n*1 x coordinates
%   y      = n*1 y coordinates
%   H      = 3*3 homograpy transformation matrix
%   method = sampling method
%
% Output:
%   pixels = mapped pixels

[ih, iw, layer] = size(I);
UV = [x, y, ones(numel(x), 1)] * H.';
u = UV(:,1)./UV(:,3);
v = UV(:,2)./UV(:,3);
u(u>iw) = iw;
u(u<1) = 1;
v(v>ih) = ih;
v(v<1) = 1;
pixels = zeros(numel(x), layer);
for i = 1:layer
    pixels(:,i) = interp2(I(:,:,i), u, v, method);
end
