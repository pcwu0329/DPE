function new_img = rgb2ycbcrNormalization(img)
% Change the input image to YCbCr format and normalize it
%
% Usage:
%   new_img = rgb2ycbcrNormalization(img)
%
% Inputs:
%   img = image in RGB format
%
% Outputs:
%   new_img = image in normalized YCbCR format

new_img(:,:,1) = 0.299*img(:,:,1) + 0.587*img(:,:,2) + 0.114*img(:,:,3);
new_img(:,:,2) = (img(:,:,3)-new_img(:,:,1))*0.564+0.5;
new_img(:,:,3) = (img(:,:,1)-new_img(:,:,1))*0.713+0.5;
