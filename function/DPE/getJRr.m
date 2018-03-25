function J_Rr = getJRr(r)
% Get J_Rr (Jabobian R/r)
%
% Usage:
%   J_Rr = getJRr(r)
%
% Inputs:
%   r = warped gradient x
%
% Output:
%   J_Rr = Jabobian (9*3 matrix) (dR/dr)

% Reference: [JMIV2015] A Compact Formula for the Derivative of a 3-D Rotation in Exponential Coordinates
EPS = 1e-25;
a = norm(r);
if a < EPS
    J_Rr = [    0,  -ry,  -rz;
             ry/2, rx/2,   -1;
             rz/2,    1, rx/2;
             ry/2, rx/2,    1;
              -rx,    0,  -rz;
               -1, rz/2, ry/2;
             rz/2,   -1, rx/2;
                1, rz/2, ry/2;
              -rx,  -ry,    0];
else
    W = getCrossProductMatrix(r);
    R = getRotMatFromP(r);
    Id_R = eye(3) - R;
    M = W * Id_R;
    R_a2 = R/a/a;
    dR_dr1 = (r(1)*W + getCrossProductMatrix(M(:, 1)))*R_a2;
    dR_dr2 = (r(2)*W + getCrossProductMatrix(M(:, 2)))*R_a2;
    dR_dr3 = (r(3)*W + getCrossProductMatrix(M(:, 3)))*R_a2;
    J_Rr = [reshape(dR_dr1.', [], 1), reshape(dR_dr2.', [], 1), reshape(dR_dr3.', [], 1)]; 
end

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
%
%% For a < 1e-5
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
%% <<<<< Simplification <<<<<
  