function new_img = ycbcr2rgbDenormalization(img)
% Change the input image to RGB format and denormalize it
%
% Usage:
%   new_img = ycbcr2rgbDenormalization(img)
%
% Inputs:
%   img = image in YCbCR format
%
% Outputs:
%   new_img = image in original RGB format

new_img(:,:,1) = img(:,:,1) + 1.403*(img(:,:,3)-0.5);
new_img(:,:,2) = img(:,:,1) - 0.714*(img(:,:,3)-0.5)-0.344*(img(:,:,2)-0.5);
new_img(:,:,3) = img(:,:,1) + 1.773*(img(:,:,2)-0.5);
