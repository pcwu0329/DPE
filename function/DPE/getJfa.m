function J = getJfa(Ix_w, Iy_w, x, y, p, in_mat)
% Get Jfa (Jabobian w/p at p=pc and convolved with gradient)
%
% Usage:
%   J = getJfa(Ix_w, Iy_w, p, S, in_mat)
%
% Inputs:
%   Ix_w   = warped gradient x
%   Iy_w   = warped gradient y
%   x      = normalized x coordinates
%   y      = normalized y coordinates
%   p      = [rx, ry, rz, tx, ty, tz];
%   in_mat = intrinsic matrix
%
% Output:
%   J = Jabobian convolved with gradient

% parameters preparation
fx = in_mat(1,1);
fy = in_mat(2,2);
rx = p(1); 
ry = p(2); 
rz = p(3); 
tx = p(4); 
ty = p(5); 
tz = p(6); 

% J_Ip = [J_IC * J_CR * J_Rr, J_IC]
% J_IC = [fx/zc     0 -(xc*fx)/zc^2;
%             0 fy/zc -(yc*fy)/zc^2];
%              
% J_CR = [x y 0 0 0 0;
%         0 0 x y 0 0;
%         0 0 0 0 x y];
%         
% J_IR = J_IC*J_CR
%      = (fx*x)/zc (fx*y)/zc         0         0 -(fx*x*xc)/zc^2 -(fx*xc*y)/zc^2;
%                0         0 (fy*x)/zc (fy*y)/zc -(fy*x*yc)/zc^2 -(fy*y*yc)/zc^2]

r = [rx; ry; rz];        
J_Rr = getJRr(r);
J_Rr([3, 6, 9], :) = [];
R = getRotMatFromP(r);
temp_coor = [x,y]*(R(:,1:2).');
xc = temp_coor(:,1) + tx;
yc = temp_coor(:,2) + ty;
izc = 1./(temp_coor(:,3) + tz);
izc2 = izc .* izc;
fxIx_zc = fx*Ix_w.*izc;
fyIy_zc = fy*Iy_w.*izc;
fxxcIx_zc2_ = -fx*xc.*Ix_w.*izc2;
fyycIy_zc2_ = -fy*yc.*Iy_w.*izc2;

Jr = [fxIx_zc.*x, fxIx_zc.*y, ...
      fyIy_zc.*x, fyIy_zc.*y, ...
      fxxcIx_zc2_.*x + fyycIy_zc2_.*x, fxxcIx_zc2_.*y + fyycIy_zc2_.*y] * J_Rr;
Jt = [fxIx_zc, fyIy_zc, fxxcIx_zc2_ + fyycIy_zc2_];
J = [Jr, Jt];


%% Appendix: Jacobian Calculation
%    [ku]   [fx  0 cx]          [R11 R12 tx+ox] [x]
%I = [kv] = [ 0 fy cy] * C, C = [R21 R22 ty+oy] [y], x = [-0.5, 0.5], y = [-0.5, 0.5]
%    [ k]   [ 0  0  1]          [R31 R32 tz+oz] [1]
% 
%dI/dp = dI/dC * dC/dRt * dRt/dp
%      = [dI/dC * dC/dR * dR/dr, dI/dC]
%							   
%For dI/dC							   
%syms fx fy cx cy xc yc zc;
%u = fx*xc/zc + cx;
%v = fy*yc/zc + cy;
%J_IC = jacobian([u, v], [xc yc zc]);
%     = [ fx/zc,     0, -(xc*fx)/zc^2]
%       [     0, fy/zc, -(yc*fy)/zc^2]
%	  
%For dC/dR							   
%dR = d(R11 R12 R13 R21 R22 R23 R31 R32 R33)
%J_CR = dC/dR = [x y z 0 0 0 0 0 0]
%               [0 0 0 x y z 0 0 0]
%               [0 0 0 0 0 0 x y z]
%		
%For dR/dr		
%% For a >= 1e-25
%syms rx ry rz;
%r = [rx; ry; rz];
%I = eye(3);
%a = norm(r);
%a2 = a * a;
%W = [0, -r(3), r(2); r(3), 0, -r(1); -r(2), r(1), 0];
%W2 = W * W;
%R = I + W * sin(a) / a + W2 * (1-cos(a)) / a2;
%J_Rr = jacobian(reshape(R.', 1, []), [rx ry rz]);
%% >>>>> Simplification >>>>>
%l2 = rx^2 + ry^2 + rz^2;
%l = l2^(0.5);
%l3 = l^3;
%l4 = l2^2;
%rx2ry2 = rx^2 + ry^2;
%rx2rz2 = rx^2 + rz^2;
%ry2rz2 = ry^2 + rz^2;
%cl = cos(l);
%sl = sin(l);
%cl1 = cl - 1;
%rxsl = rx*sl;
%rysl = ry*sl;
%rzsl = rz*sl;
%rxcl = rx*cl;
%rycl = ry*cl;
%rzcl = rz*cl;
%sl_l = sl/l;
%rx_l2 = rx/l2;
%ry_l2 = ry/l2;
%rz_l2 = rz/l2;
%rx_l3 = rx/l3;
%ry_l3 = ry/l3;
%rz_l3 = rz/l3;
%cl1_l4 = cl1/l4;
%rxcl1_l4 = rx*cl1_l4;
%rycl1_l4 = ry*cl1_l4;
%rzcl1_l4 = rz*cl1_l4;
%J_Rr_new = [ - (sl*ry2rz2*rx_l3) - (2*ry2rz2*rxcl1_l4),(2*cl1*ry_l2) - (sl*ry2rz2*ry_l3) - (2*ry2rz2*rycl1_l4),(2*cl1*rz_l2) - (sl*ry2rz2*rz_l3) - (2*ry2rz2*rzcl1_l4);
%            (rzsl*rx_l3) - (rzcl*rx_l2) - (cl1*ry_l2) + (rx*rysl*rx_l3) + (2*rx*ry*rxcl1_l4), (rzsl*ry_l3) - (rzcl*ry_l2) - (cl1*rx_l2) + (rx*rysl*ry_l3) + (2*rx*ry*rycl1_l4),      (rzsl*rz_l3) - (rzcl*rz_l2) - sl_l + (rx*rysl*rz_l3) + (2*rx*ry*rzcl1_l4);
%            (rycl*rx_l2) - (cl1*rz_l2) - (rysl*rx_l3) + (rx*rzsl*rx_l3) + (2*rx*rz*rxcl1_l4),      sl_l + (rycl*ry_l2) - (rysl*ry_l3) + (rx*rzsl*ry_l3) + (2*rx*rz*rycl1_l4), (rycl*rz_l2) - (cl1*rx_l2) - (rysl*rz_l3) + (rx*rzsl*rz_l3) + (2*rx*rz*rzcl1_l4);
%            (rzcl*rx_l2) - (cl1*ry_l2) - (rzsl*rx_l3) + (rx*rysl*rx_l3) + (2*rx*ry*rxcl1_l4), (rzcl*ry_l2) - (cl1*rx_l2) - (rzsl*ry_l3) + (rx*rysl*ry_l3) + (2*rx*ry*rycl1_l4),      sl_l + (rzcl*rz_l2) - (rzsl*rz_l3) + (rx*rysl*rz_l3) + (2*rx*ry*rzcl1_l4);
%            (2*cl1*rx_l2) - (sl*rx2rz2*rx_l3) - (2*rx2rz2*rxcl1_l4),- (sl*rx2rz2*ry_l3) - (2*rx2rz2*rycl1_l4),(2*cl1*rz_l2) - (sl*rx2rz2*rz_l3) - (2*rx2rz2*rzcl1_l4);
%            (rxsl*rx_l3) - (rxcl*rx_l2) - sl_l + (ry*rzsl*rx_l3) + (2*ry*rz*rxcl1_l4), (rxsl*ry_l3) - (rxcl*ry_l2) - (cl1*rz_l2) + (ry*rzsl*ry_l3) + (2*ry*rz*rycl1_l4), (rxsl*rz_l3) - (rxcl*rz_l2) - (cl1*ry_l2) + (ry*rzsl*rz_l3) + (2*ry*rz*rzcl1_l4);
%            (rysl*rx_l3) - (rycl*rx_l2) - (cl1*rz_l2) + (rx*rzsl*rx_l3) + (2*rx*rz*rxcl1_l4),      (rysl*ry_l3) - (rycl*ry_l2) - sl_l + (rx*rzsl*ry_l3) + (2*rx*rz*rycl1_l4), (rysl*rz_l3) - (rycl*rz_l2) - (cl1*rx_l2) + (rx*rzsl*rz_l3) + (2*rx*rz*rzcl1_l4);
%            sl_l + (rxcl*rx_l2) - (rxsl*rx_l3) + (ry*rzsl*rx_l3) + (2*ry*rz*rxcl1_l4), (rxcl*ry_l2) - (cl1*rz_l2) - (rxsl*ry_l3) + (ry*rzsl*ry_l3) + (2*ry*rz*rycl1_l4), (rxcl*rz_l2) - (cl1*ry_l2) - (rxsl*rz_l3) + (ry*rzsl*rz_l3) + (2*ry*rz*rzcl1_l4);
%            (2*cl1*rx_l2) - (sl*rx2ry2*rx_l3) - (2*rx2ry2*rxcl1_l4),(2*cl1*ry_l2) - (sl*rx2ry2*ry_l3) - (2*rx2ry2*rycl1_l4),- (sl*rx2ry2*rz_l3) - (2*rx2ry2*rzcl1_l4)]
%% <<<<< Simplification <<<<<
%
%% For a < 1e-25
%syms rx ry rz;
%r = [rx; ry; rz];
%I = eye(3);
%a = norm(r);
%a2 = a * a;
%W = [0, -r(3), r(2); r(3), 0, -r(1); -r(2), r(1), 0];
%W2 = W * W;
%R = I + W + 0.5 * W2;
%J_Rr = jacobian(reshape(R.', 1, []), [rx ry rz]);
%     = [    0,  -ry,  -rz;
%         ry/2, rx/2,   -1;
%         rz/2,    1, rx/2;
%         ry/2, rx/2,    1;
%          -rx,    0,  -rz;
%           -1, rz/2, ry/2;
%         rz/2,   -1, rx/2;
%            1, rz/2, ry/2;
%          -rx,  -ry,    0]

  