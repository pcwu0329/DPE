function blur_sigma = calSigmaValue(tmp, fx, fy, dim, tz_square)
% Calculate the blur variable sigma for getting images with decent total variation
%
% Usage:
%   blur_sigma = calSigmaValue(tmp, fx, fy, dim, tz_square)
%
% Inputs:
%   tmp       = grayscale template image
%   fx        = focal length in x direction
%   fy        = focal length in y direction
%   dim       = dimension variables
%   tz_square = min_z * max_z
%
% Outputs:
%   blur_sigma = blur sigma value

blur_sigma = zeros(1, 3);
blur_tmp = tmp;
scale = 0.0625;
for i = 1:3
    total_variation_of_template_in_camera_frame = 4210;
    area = (2*fx*dim.tmp_real_w) * (2*fy*dim.tmp_real_h) * (scale^2) / tz_square;
    length = sqrt(area);
    while (total_variation_of_template_in_camera_frame > 8.42*length) % 8.42 is obtained emperically
        blur_sigma(i) = blur_sigma(i) + 1;
        blur_tmp = imgaussfilt(tmp, blur_sigma(i), 'Padding', 'symmetric');
        local_max = ordfilt2(blur_tmp, 9, true(3));
        local_min = ordfilt2(blur_tmp, 1, true(3));
        variations = max(local_max - blur_tmp, blur_tmp - local_min);
        total_variation_of_template_in_camera_frame = mean2(variations) * area;
    end
    scale = scale * 4;
end
