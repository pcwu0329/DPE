function coor2D = coorTrans3Dto2D(M, coor3D)
% Transform the coordinate from coor3D (x, y, z) to coor2D (u, v) via M
%
% Usage:
%   coor2D = coorTrans3Dto2D( M, coor3D )
%
% Inputs:
%   M = 4x4 transformation matrix
%   coor3D = original 3d coordinates (size = n*3)
%
% Outputs:
%   coor2D = new 2d coordinate (size = n*2)
%
% Details:
%   [hu] =[M11 M12 M13 M14]   [x]
%   [hv] =[M21 M22 M23 M24] * [y] 
%   [h ] =[M31 M32 M33 M34]   [z]
%   [1 ] =[M41 M42 M43 M44]   [1]

B = [coor3D, ones(size(coor3D, 1), 1)];
A = B * M.';
coor2D = A(:,1:2) ./ repmat(A(:,3), 1, 2);

