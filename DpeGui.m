function varargout = DpeGui(varargin)

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @DpeGui_OpeningFcn, ...
                   'gui_OutputFcn',  @DpeGui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before DpeGui is made visible.
function DpeGui_OpeningFcn(hObject, eventdata, handles, varargin)
addpath(genpath('function/'));
clc;
handles.output = hObject;
guidata(hObject, handles);
global Ih Iw fx fy cx cy in_mat Th Tw tmp_real_w tmp_real_h min_dim nm_mat
global min_rz0 max_rz0 min_rx max_rx min_rz1 max_rz1
global min_tx max_tx min_ty max_ty min_tz max_tz
global min_bs max_bs min_ns max_ns min_sf max_sf
global rz0 rx rz1 tx ty tz ex_mat tf_mat
global bs ns sf
global epsilon delta prm_lvls photo_inva verbose
global tmp_idx bg_idx tmp_imgs bg_imgs tmp bg img
global tmp_uint8
global ocv_install_dir egn_install_dir

% dependency directories
ocv_install_dir = '';
egn_install_dir = '';

% DPE parameters
epsilon = 0.25;
delta = 0.15;
prm_lvls = 2;
photo_inva = true;
verbose = true;

% in_mat
Ih = 600;
Iw = 800;
fx = norm([Iw, Ih]);
fy = fx;
cx = (Iw+1)/2;
cy = (Ih+1)/2;
in_mat = [fx,0,cx,0;0,fy,cy,0;0,0,1,0;0,0,0,1];

% nm_mat
Th = 480;
Tw = 640;
min_dim = 0.5;
tmp_real_w  = min_dim * 640 / 480;
tmp_real_h = min_dim;
nm_mat = eye(3);
x_center = 0.5*(640 + 1);
y_center = 0.5*(480 + 1);
nm_mat(1, 1) = 2 * tmp_real_w / Tw;
nm_mat(1, 3) = -x_center * nm_mat(1, 1);
nm_mat(2, 2) = -2 * tmp_real_h / Th;
nm_mat(2, 3) = -y_center * nm_mat(2, 2);

% boundary
min_rz0 = -180;
max_rz0 = 179.99;
min_rx = 0;
max_rx = 80;
min_rz1 = -180;
max_rz1 = 179.99;
min_tz = 3;
max_tz = 9;
max_tx = cx * ((min_tz+max_tz)/2) / fx - min_dim;
min_tx = -max_tx;
max_ty = cy * ((min_tz+max_tz)/2) / fy - min_dim;
min_ty = -max_ty;
min_bs = 0;
max_bs = 5;
min_ns = 0;
max_ns = 5;
min_sf = 0.5;
max_sf = 1;

% pose parameters
rz0 = 0;
rx = 40;
rz1 = 0;
tx = 0;
ty = 0;
tz = (min_tz + max_tz) / 2;

% degradation parameters 
bs = 0;
ns = 0;
sf = 1;

% images
if ~exist('background', 'dir')
    disp('==> downloading background images ...');
    if strcmp('win64', computer('arch'))
        system(['powershell -Command "Invoke-WebRequest ', ...
                'http://media.ee.ntu.edu.tw/research/DPE/datasets/background.zip ', ...
                '-OutFile background.zip"']);
    else
        system(['wget ', ...
                'http://media.ee.ntu.edu.tw/research/DPE/datasets/background.zip']);
    end
    unzip('background')
    delete background.zip;
end
if ~exist('template', 'dir')
    disp('==> downloading template images ...');
    if strcmp('win64', computer('arch'))
        system(['powershell -Command "Invoke-WebRequest ', ...
                'http://media.ee.ntu.edu.tw/research/DPE/datasets/template.zip ', ...
                '-OutFile template.zip"']);
    else
        system(['wget ', ...
                'http://media.ee.ntu.edu.tw/research/DPE/datasets/template.zip']);
    end
    unzip('template');
    delete template.zip;
end
tmp_idx = 1;
bg_idx = 1;
set(handles.pm_tmp, 'Value', tmp_idx);
set(handles.pm_bg, 'Value', bg_idx);
tmp_imgs = {
    'template/BumpSign.png'; ...
    'template/StopSign.png'; ...
    'template/Lucent.png'; ...
    'template/MacMiniBoard.png'; ...
    'template/Isetta.png'; ...
    'template/Philadelphia.png'; ...
    'template/Grass.png'; ...
    'template/Wall.png'};
bg_imgs = {
    'background/00.png'; ...
    'background/01.png'; ...
    'background/02.png'; ...
    'background/03.png'; ...
    'background/04.png'; ...
    'background/05.png'; ...
    'background/06.png'; ...
    'background/07.png'; ...
    'background/08.png'; ...
    'background/09.png'; ...
    'background/10.png'; ...
    'background/11.png'; ...
    'background/12.png'; ...
    'background/13.png'; ...
    'background/14.png'; ...
    'background/15.png'; ...
    'background/16.png'; ...
    'background/17.png'; ...
    'background/18.png'; ...
    'background/19.png'; ...
    'background/20.png'; ...
    'background/21.png'; ...
    'background/22.png'; ...
    'background/23.png'; ...
    'background/24.png'; ...
    'background/25.png'; ...
    'background/26.png'; ...
    'background/27.png'; ...
    'background/28.png'; ...
    'background/29.png'; ...
    'background/30.png'; ...
    'background/31.png'; ...
    'background/32.png'; ...
    'background/33.png'; ...
    'background/34.png'; ...
    'background/35.png'; ...
    'background/36.png'; ...
    'background/37.png'; ...
    'background/38.png'; ...
    'background/39.png'; ...
    'background/40.png'; ...
    'background/41.png'; ...
    'background/42.png'; ...
    'background/43.png'; ...
    'background/44.png'; ...
    'background/45.png'; ...
    'background/46.png'; ...
    'background/47.png'; ...
    'background/48.png'; ...
    'background/49.png'; ...
    'background/50.png'; ...
    'background/51.png'; ...
    'background/52.png'; ...
    'background/53.png'; ...
    'background/54.png'; ...
    'background/55.png'; ...
    'background/56.png'; ...
    'background/57.png'; ...
    'background/58.png'; ...
    'background/59.png'; ...
    'background/60.png'; ...
    'background/61.png'; ...
    'background/62.png'; ...
    'background/63.png'; ...
    'background/64.png'; ...
    'background/65.png'; ...
    'background/66.png'; ...
    'background/67.png'; ...
    'background/68.png'; ...
    'background/69.png'; ...
    'background/70.png'; ...
    'background/71.png'; ...
    'background/72.png'; ...
    'background/73.png'; ...
    'background/74.png'; ...
    'background/75.png'; ...
    'background/76.png'; ...
    'background/77.png'; ...
    'background/78.png'; ...
    'background/79.png'; ...
    'background/80.png'; ...
    'background/81.png'; ...
    'background/82.png'; ...
    'background/83.png'; ...
    'background/84.png'; ...
    'background/85.png'; ...
    'background/86.png'; ...
    'background/87.png'; ...
    'background/88.png'; ...
    'background/89.png'; ...
    'background/90.png'; ...
    'background/91.png'; ...
    'background/92.png'; ...
    'background/93.png'; ...
    'background/94.png'; ...
    'background/95.png'; ...
    'background/96.png'; ...
    'background/97.png'; ...
    'background/98.png'; ...
    'background/99.png'};
tmp_uint8 = imread(tmp_imgs{tmp_idx});
tmp = im2double(tmp_uint8);
bg = im2double(imread(bg_imgs{bg_idx}));
ex_mat = calExMat(rz0, rx, rz1, tx, ty, tz);
tf_mat = in_mat * ex_mat;
img = blendImage(bg, tmp, tf_mat(1:3, [1, 2, 4])*nm_mat);
display(handles);

% read icon files
[I,~,A]=imread('icon/random_pose_generation.png');
alpha = double(repmat(A,1,1,3))./255;
I = uint8(double(I).*alpha + 207*(1-alpha));
set(handles.pb_rpg,'CData',I);

[I,~,A]=imread('icon/pose_estimation.png');
alpha = double(repmat(A,1,1,3))./255;
I = uint8(double(I).*alpha + 207*(1-alpha));
set(handles.pb_pe,'CData',I);

% set sliders
set(handles.sl_rz0, 'Min', min_rz0);
set(handles.sl_rz0, 'Max', max_rz0);
set(handles.sl_rz0, 'Value', rz0);
set(handles.tx_rz0, 'String', sprintf('Rotation z_0: %.2f', rz0));
set(handles.sl_rx, 'Min', min_rx);
set(handles.sl_rx, 'Max', max_rx);
set(handles.sl_rx, 'Value', rx);
set(handles.tx_rx, 'String', sprintf('Rotation x: %.2f', rx));
set(handles.sl_rz1, 'Min', min_rz1);
set(handles.sl_rz1, 'Max', max_rz1);
set(handles.sl_rz1, 'Value', rz1);
set(handles.tx_rz1, 'String', sprintf('Rotation z_1: %.2f', rz1));
set(handles.sl_tx, 'Min', min_tx);
set(handles.sl_tx, 'Max', max_tx);
set(handles.sl_tx, 'Value', tx);
set(handles.tx_tx, 'String', sprintf('Translation x: %.2f', tx));
set(handles.sl_ty, 'Min', min_ty);
set(handles.sl_ty, 'Max', max_ty);
set(handles.sl_ty, 'Value', ty);
set(handles.tx_ty, 'String', sprintf('Translation y: %.2f', ty));
set(handles.sl_tz, 'Min', min_tz);
set(handles.sl_tz, 'Max', max_tz);
set(handles.sl_tz, 'Value', tz);
set(handles.tx_tz, 'String', sprintf('Translation z: %.2f', tz));
set(handles.sl_bs, 'Min', min_bs);
set(handles.sl_bs, 'Max', max_bs);
set(handles.sl_bs, 'Value', bs);
set(handles.tx_bs, 'String', sprintf('Gaussian Blur Sigma: %.2f', bs));
set(handles.sl_ns, 'Min', min_ns);
set(handles.sl_ns, 'Max', max_ns);
set(handles.sl_ns, 'Value', ns);
set(handles.tx_ns, 'String', sprintf('Gaussian Noise Sigma: %.2f', ns));
set(handles.sl_sf, 'Min', min_sf);
set(handles.sl_sf, 'Max', max_sf);
set(handles.sl_sf, 'Value', min_sf+max_sf-sf);
set(handles.tx_sf, 'String', sprintf('Intensity Scale Factor: %.2f', sf));

% --- Outputs from this function are returned to the command line.
function varargout = DpeGui_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

function pm_tmp_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
function pm_bg_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
function sl_rz0_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
function sl_rx_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
function sl_rz1_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
function sl_tx_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
function sl_ty_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
function sl_tz_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
function sl_bs_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
function sl_ns_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
function sl_sf_CreateFcn(hObject, eventdata, handles)
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

% --- Executes on button press in pb_rpg.
function pb_rpg_Callback(hObject, eventdata, handles)
global tmp_idx bg_idx tmp_imgs bg_imgs tmp bg img
global Iw Ih fx fy cx cy
global min_dim tmp_real_w tmp_real_h
global min_rz0 max_rz0 min_rx max_rx min_rz1 max_rz1
global min_tx max_tx min_ty max_ty min_tz max_tz
global min_bs max_bs min_ns max_ns min_sf max_sf
global rz0 rx rz1 tx ty tz
global in_mat ex_mat nm_mat tf_mat
global bs ns sf
global tmp_uint8
while true
    if ~get(handles.cb_tmp, 'Value')
        tmp_idx = randi(size(get(handles.pm_tmp, 'String'), 1));
        tmp_uint8 = imread(tmp_imgs{tmp_idx});
        tmp = im2double(tmp_uint8);
        set(handles.pm_tmp, 'Value', tmp_idx);
    end
    if ~get(handles.cb_bg, 'Value')
        bg_idx = randi(size(get(handles.pm_bg, 'String'), 1));
        bg = im2double(imread(bg_imgs{bg_idx}));
        set(handles.pm_bg, 'Value', bg_idx);
    end
    if ~get(handles.cb_rz0, 'Value')
        rz0 = min_rz0 + rand*(max_rz0-min_rz0);
        set(handles.sl_rz0, 'Value', rz0);
        set(handles.tx_rz0, 'String', sprintf('Rotation z_0: %.2f', rz0));
    end
    if ~get(handles.cb_rx, 'Value')
        rx = min_rx + rand*(max_rx-min_rx);
        set(handles.sl_rx, 'Value', rx);
        set(handles.tx_rx, 'String', sprintf('Rotation x: %.2f', rx));
    end
    if ~get(handles.cb_rz1, 'Value')
        rz1 = min_rz1 + rand*(max_rz1-min_rz1);
        set(handles.sl_rz1, 'Value', rz0);
        set(handles.tx_rz1, 'String', sprintf('Rotation z_1: %.2f', rz1));
    end
    if ~get(handles.cb_tz, 'Value')
        tz = min_tz + rand*(max_tz-min_tz);
        max_tx = cx * tz / fx - min_dim;
        min_tx = -max_tx;
        max_ty = cy * tz / fy - min_dim;
        min_ty = -max_ty;
        if tx > max_tx, tx = max_tx; elseif tx < min_tx, tx = min_tx; end
        if ty > max_ty, ty = max_ty; elseif ty < min_ty, ty = min_ty; end
        set(handles.sl_tx, 'Min', min_tx);
        set(handles.sl_tx, 'Max', max_tx);
        set(handles.sl_ty, 'Min', min_ty);
        set(handles.sl_ty, 'Max', max_ty);
        set(handles.sl_tx, 'Value', tx);
        set(handles.sl_ty, 'Value', ty);
        set(handles.sl_tz, 'Value', tz);
        set(handles.tx_tx, 'String', sprintf('Translation x: %.2f', tx));
        set(handles.tx_ty, 'String', sprintf('Translation y: %.2f', ty));
        set(handles.tx_tz, 'String', sprintf('Translation z: %.2f', tz));
    end
    if ~get(handles.cb_tx, 'Value')
        tx = min_tx + rand*(max_tx-min_tx);
        set(handles.sl_tx, 'Value', tx);
        set(handles.tx_tx, 'String', sprintf('Translation x: %.2f', tx));
    end
    if ~get(handles.cb_ty, 'Value')
        ty = min_ty + rand*(max_ty-min_ty);
        set(handles.sl_ty, 'Value', ty);
        set(handles.tx_ty, 'String', sprintf('Translation y: %.2f', ty));
    end
    if ~get(handles.cb_bs, 'Value')
        bs = min_bs + rand*(max_bs-min_bs);
        set(handles.sl_bs, 'Value', bs);
        set(handles.tx_bs, 'String', sprintf('Gaussian Blur Sigma: %.2f', bs));
    end
    if ~get(handles.cb_ns, 'Value')
        ns = min_ns + rand*(max_ns-min_ns);
        set(handles.sl_ns, 'Value', ns);
        set(handles.tx_ns, 'String', sprintf('Gaussian Noise Sigma: %.2f', ns));
    end
    if ~get(handles.cb_sf, 'Value')
        sf = min_sf + rand*(max_sf-min_sf);
        set(handles.sl_sf, 'Value', min_sf+max_sf-sf);
        set(handles.tx_sf, 'String', sprintf('Intensity Scale Factor: %.2f', sf));
    end
    % check if the template image is within the background image
    ex_mat = calExMat(rz0, rx, rz1, tx, ty, tz);
    tf_mat = in_mat * ex_mat;
    is_valid = checkValidity(tf_mat, Iw, Ih, tmp_real_w, tmp_real_h);
    if is_valid, break; end
end
img = blendImage(bg, tmp, tf_mat(1:3, [1, 2, 4])*nm_mat);

display(handles);

% --- Executes on button press in pb_pe.
function pb_pe_Callback(hObject, eventdata, handles)
global fx fy cx cy
global tmp_real_w tmp_real_h
global in_mat ex_mat
global min_dim min_tz max_tz
global epsilon delta prm_lvls photo_inva verbose
global res tmp
global tmp_uint8

setEnable(handles, 'off');
thres = [20, 10];
if get(handles.rb_cpu, 'Value')
    ex_mat_dpe = dpe(tmp, res, in_mat, min_dim, min_tz, max_tz, epsilon, delta, prm_lvls, photo_inva, verbose);
else
    res_uint8 = uint8(res*255);
    ex_mat_ape = apeCuda(tmp_uint8, res_uint8, fx, fy, cx, cy, ...
                         min_dim, min_tz, max_tz, epsilon, prm_lvls, photo_inva, verbose);
    if  ex_mat_ape(4,4)
        ex_mat_dpe = prCuda(tmp_uint8, res_uint8, ex_mat_ape, fx, fy, cx, cy, ...
                            min_dim, tmp_real_w, tmp_real_h, prm_lvls, photo_inva, verbose);
    else
        ex_mat_dpe = zeros(4,4);
    end
end
if ex_mat_dpe(4,4)
    [err_r, err_t] = calPoseDiff(ex_mat(1:3, :), ex_mat_dpe(1:3, :));
    is_dpe_good = err_r < thres(1) && err_t < thres(2);
    if is_dpe_good
        colors = {[0, 0, 255]; [0, 255, 255]; [0, 255, 0]};
    else
        colors = {[255, 0, 0]; [255, 0, 255]; [255, 255, 0]};
    end
    M = in_mat * ex_mat_dpe;
    drawBox(handles, M, tmp_real_w, tmp_real_h, colors);
    if verbose
        [err_r, err_t] = calPoseDiff(ex_mat(1:3, 1:4), ex_mat_dpe(1:3, 1:4));
        fprintf(1, '[*** Final condition ***] Rotation error: %.6f degrees, Translation error: %.6f %%\n', err_r, err_t);
    end
end    
setEnable(handles, 'on');

% --- Executes on selection change in pm_tmp.
function pm_tmp_Callback(hObject, eventdata, handles)
global tmp_idx tmp_imgs tmp bg 
global tf_mat nm_mat img
global tmp_uint8
temp_tmp_idx = get(handles.pm_tmp, 'Value');
if temp_tmp_idx == tmp_idx, return; end
tmp_idx = temp_tmp_idx;
tmp_uint8 = imread(tmp_imgs{tmp_idx});
tmp = im2double(tmp_uint8);
img = blendImage(bg, tmp, tf_mat(1:3, [1, 2, 4])*nm_mat);
display(handles);

% --- Executes on selection change in pm_bg.
function pm_bg_Callback(hObject, eventdata, handles)
global bg_idx bg_imgs bg tmp
global tf_mat nm_mat img
temp_bg_idx = get(handles.pm_bg, 'Value');
if temp_bg_idx == bg_idx, return; end
bg_idx = temp_bg_idx;
bg = im2double(imread(bg_imgs{bg_idx}));
img = blendImage(bg, tmp, tf_mat(1:3, [1, 2, 4])*nm_mat);
display(handles);

% --- Executes on slider movement in sl_rz0.
function sl_rz0_Callback(hObject, eventdata, handles)
global tmp bg img
global Iw Ih tmp_real_w tmp_real_h
global rz0 rx rz1 tx ty tz
global in_mat ex_mat nm_mat tf_mat
temp_rz0 = get(handles.sl_rz0, 'Value');
if temp_rz0 == rz0, return; end
temp_ex_mat = calExMat(temp_rz0, rx, rz1, tx, ty, tz);
temp_tf_mat = in_mat * temp_ex_mat;
is_valid = checkValidity(temp_tf_mat, Iw, Ih, tmp_real_w, tmp_real_h);
if ~is_valid
    set(handles.sl_rz0, 'Value', rz0);
    return;
end
rz0 = temp_rz0;
ex_mat = temp_ex_mat;
tf_mat = temp_tf_mat;
img = blendImage(bg, tmp, tf_mat(1:3, [1, 2, 4])*nm_mat);
display(handles);
set(handles.tx_rz0, 'String', sprintf('Rotation z_0: %.2f', rz0));

% --- Executes on slider movement in sl_rx.
function sl_rx_Callback(hObject, eventdata, handles)
global tmp bg img
global Iw Ih tmp_real_w tmp_real_h
global rz0 rx rz1 tx ty tz
global in_mat ex_mat nm_mat tf_mat
temp_rx = get(handles.sl_rx, 'Value');
if temp_rx == rx, return; end
temp_ex_mat = calExMat(rz0, temp_rx, rz1, tx, ty, tz);
temp_tf_mat = in_mat * temp_ex_mat;
is_valid = checkValidity(temp_tf_mat, Iw, Ih, tmp_real_w, tmp_real_h);
if ~is_valid
    set(handles.sl_rx, 'Value', rx);
    return;
end
rx = temp_rx;
ex_mat = temp_ex_mat;
tf_mat = temp_tf_mat;
img = blendImage(bg, tmp, tf_mat(1:3, [1, 2, 4])*nm_mat);
display(handles);
set(handles.tx_rx, 'String', sprintf('Rotation x: %.2f', rx));

% --- Executes on slider movement in sl_rz1.
function sl_rz1_Callback(hObject, eventdata, handles)
global tmp bg img
global Iw Ih tmp_real_w tmp_real_h
global rz0 rx rz1 tx ty tz
global in_mat ex_mat nm_mat tf_mat
temp_rz1 = get(handles.sl_rz1, 'Value');
if temp_rz1 == rz1, return; end
temp_ex_mat = calExMat(rz0, rx, temp_rz1, tx, ty, tz);
temp_tf_mat = in_mat * temp_ex_mat;
is_valid = checkValidity(temp_tf_mat, Iw, Ih, tmp_real_w, tmp_real_h);
if ~is_valid
    set(handles.sl_rz1, 'Value', rz1);
    return;
end
rz1 = temp_rz1;
ex_mat = temp_ex_mat;
tf_mat = temp_tf_mat;
img = blendImage(bg, tmp, tf_mat(1:3, [1, 2, 4])*nm_mat);
display(handles);
set(handles.tx_rz1, 'String', sprintf('Rotation z_1: %.2f', rz1));

% --- Executes on slider movement in sl_tx.
function sl_tx_Callback(hObject, eventdata, handles)
global tmp bg img
global Iw Ih tmp_real_w tmp_real_h
global rz0 rx rz1 tx ty tz
global in_mat ex_mat nm_mat tf_mat
temp_tx = get(handles.sl_tx, 'Value');
if temp_tx == tx, return; end
temp_ex_mat = calExMat(rz0, rx, rz1, temp_tx, ty, tz);
temp_tf_mat = in_mat * temp_ex_mat;
is_valid = checkValidity(temp_tf_mat, Iw, Ih, tmp_real_w, tmp_real_h);
if ~is_valid
    set(handles.sl_tx, 'Value', tx);
    return;
end
tx = temp_tx;
ex_mat = temp_ex_mat;
tf_mat = temp_tf_mat;
img = blendImage(bg, tmp, tf_mat(1:3, [1, 2, 4])*nm_mat);
display(handles);
set(handles.tx_tx, 'String', sprintf('Translation x: %.2f', tx));

% --- Executes on slider movement in sl_ty.
function sl_ty_Callback(hObject, eventdata, handles)
global tmp bg img
global Iw Ih tmp_real_w tmp_real_h
global rz0 rx rz1 tx ty tz
global in_mat ex_mat nm_mat tf_mat
temp_ty = get(handles.sl_ty, 'Value');
if temp_ty == ty, return; end
temp_ex_mat = calExMat(rz0, rx, rz1, tx, temp_ty, tz);
temp_tf_mat = in_mat * temp_ex_mat;
is_valid = checkValidity(temp_tf_mat, Iw, Ih, tmp_real_w, tmp_real_h);
if ~is_valid
    set(handles.sl_ty, 'Value', ty);
    return;
end
ty = temp_ty;
ex_mat = temp_ex_mat;
tf_mat = temp_tf_mat;
img = blendImage(bg, tmp, tf_mat(1:3, [1, 2, 4])*nm_mat);
display(handles);
set(handles.tx_ty, 'String', sprintf('Translation y: %.2f', ty));

% --- Executes on slider movement in sl_tz.
function sl_tz_Callback(hObject, eventdata, handles)
global tmp bg img
global Iw Ih tmp_real_w tmp_real_h
global rz0 rx rz1 tx ty tz
global in_mat ex_mat nm_mat tf_mat
global min_tx max_tx min_ty max_ty
global cx cy fx fy min_dim
temp_tz = get(handles.sl_tz, 'Value');
if temp_tz == tz, return; end
temp_ex_mat = calExMat(rz0, rx, rz1, tx, ty, temp_tz);
temp_tf_mat = in_mat * temp_ex_mat;
is_valid = checkValidity(temp_tf_mat, Iw, Ih, tmp_real_w, tmp_real_h);
if ~is_valid
    set(handles.sl_tz, 'Value', tz);
    return;
end
tz = temp_tz;
ex_mat = temp_ex_mat;
tf_mat = temp_tf_mat;
img = blendImage(bg, tmp, tf_mat(1:3, [1, 2, 4])*nm_mat);
max_tx = cx * tz / fx - min_dim;
min_tx = -max_tx;
max_ty = cy * tz / fy - min_dim;
min_ty = -max_ty;
display(handles);
set(handles.sl_tx, 'Min', min_tx);
set(handles.sl_tx, 'Max', max_tx);
set(handles.sl_ty, 'Min', min_ty);
set(handles.sl_ty, 'Max', max_ty);
set(handles.tx_tz, 'String', sprintf('Translation z: %.2f', tz));

% --- Executes on slider movement in sl_bs.
function sl_bs_Callback(hObject, eventdata, handles)
global bs
temp_bs = get(handles.sl_bs, 'Value');
if temp_bs == bs, return; end
bs = temp_bs;
display(handles);
set(handles.tx_bs, 'String', sprintf('Gaussian Blur Sigma: %.2f', bs));

% --- Executes on slider movement in sl_ns.
function sl_ns_Callback(hObject, eventdata, handles)
global ns
temp_ns = get(handles.sl_ns, 'Value');
if temp_ns == ns, return; end
ns = temp_ns;
display(handles);
set(handles.tx_ns, 'String', sprintf('Gaussian Noise Sigma: %.2f', ns));

% --- Executes on slider movement in sl_sf.
function sl_sf_Callback(hObject, eventdata, handles)
global sf min_sf max_sf
temp_sf = min_sf + max_sf - get(handles.sl_sf, 'Value');
if temp_sf == sf, return; end
sf = temp_sf;
display(handles);
set(handles.tx_sf, 'String', sprintf('Intensity Scale Factor: %.2f', sf));

function cb_tmp_Callback(hObject, eventdata, handles)
function cb_bg_Callback(hObject, eventdata, handles)
function cb_sf_Callback(hObject, eventdata, handles)
function cb_ns_Callback(hObject, eventdata, handles)
function cb_bs_Callback(hObject, eventdata, handles)
function cb_rz1_Callback(hObject, eventdata, handles)
function cb_rx_Callback(hObject, eventdata, handles)
function cb_rz0_Callback(hObject, eventdata, handles)
function cb_tz_Callback(hObject, eventdata, handles)
function cb_ty_Callback(hObject, eventdata, handles)
function cb_tx_Callback(hObject, eventdata, handles)

function setEnable(handles, enable)
set(handles.rb_cpu, 'Enable', enable);
set(handles.rb_gpu, 'Enable', enable);
set(handles.pb_rpg, 'Enable', enable);
set(handles.pb_pe, 'Enable', enable);
set(handles.pm_tmp, 'Enable', enable);
set(handles.cb_tmp, 'Enable', enable);
set(handles.pm_bg, 'Enable', enable);
set(handles.cb_bg, 'Enable', enable);
set(handles.sl_rz0, 'Enable', enable);
set(handles.cb_rz0, 'Enable', enable);
set(handles.sl_rx, 'Enable', enable);
set(handles.cb_rx, 'Enable', enable);
set(handles.sl_rz1, 'Enable', enable);
set(handles.cb_rz1, 'Enable', enable);
set(handles.sl_tx, 'Enable', enable);
set(handles.cb_tx, 'Enable', enable);
set(handles.sl_ty, 'Enable', enable);
set(handles.cb_ty, 'Enable', enable);
set(handles.sl_tz, 'Enable', enable);
set(handles.cb_tz, 'Enable', enable);
set(handles.sl_bs, 'Enable', enable);
set(handles.cb_bs, 'Enable', enable);
set(handles.sl_ns, 'Enable', enable);
set(handles.cb_ns, 'Enable', enable);
set(handles.sl_sf, 'Enable', enable);
set(handles.cb_sf, 'Enable', enable);
drawnow;

function display(handles)
global img res
global bs ns sf
res = img;
if bs ~= 0
    blur_size = round(1+bs*4);
    blur_kernel  = fspecial('gaussian', blur_size, bs);
    res = imfilter(res, blur_kernel,'symmetric');
end
if sf ~= 1
    res = res * sf;
end
if ns ~= 0
    res = imnoise(res,'gaussian', 0, ns/256);
end
axes(handles.ax_fig);
cla;
imagesc(res);
set(handles.ax_fig, 'XTick', []);
set(handles.ax_fig, 'YTick', []);
drawnow;

function drawBox(handles, M, u, v, colors)
global res
c = cell(4, 1);
c{1} = [uint8(colors{1}), 1].';
c{4} = [uint8(colors{3}), 1].';
c{2} = [uint8(colors{1} * (1/3) + colors{2} * (2/3)), 1].';
c{3} = [uint8(colors{3} * (1/3) + colors{2} * (2/3)), 1].';
x = [-u, u];
y = [-v, v];
z = [ 0, v];
B1 = coorTrans3Dto2D(M,[x(2),y(2),z(1)]);
B2 = coorTrans3Dto2D(M,[x(2),y(1),z(1)]);
B3 = coorTrans3Dto2D(M,[x(1),y(1),z(1)]);
B4 = coorTrans3Dto2D(M,[x(1),y(2),z(1)]);
T1 = coorTrans3Dto2D(M,[x(2),y(2),z(2)]);
T2 = coorTrans3Dto2D(M,[x(2),y(1),z(2)]);
T3 = coorTrans3Dto2D(M,[x(1),y(1),z(2)]);
T4 = coorTrans3Dto2D(M,[x(1),y(2),z(2)]);
axes(handles.ax_fig);
cla;
imagesc(res);
set(handles.ax_fig, 'XTick', []);
set(handles.ax_fig, 'YTick', []);
hold on;
line_width = 2;
p = plot([B1(1);B2(1)], [B1(2);B2(2)], 'LineWidth', line_width, 'LineStyle', '-');
drawnow;
set(p.Edge, 'ColorBinding','interpolated', 'ColorData', [c{3}, c{1}]);
p = plot([B2(1);B3(1)], [B2(2);B3(2)], 'LineWidth', line_width, 'LineStyle', '-');
drawnow;
set(p.Edge, 'ColorBinding','interpolated', 'ColorData', [c{1}, c{1}]);
p = plot([B3(1);B4(1)], [B3(2);B4(2)], 'LineWidth', line_width, 'LineStyle', '-');
drawnow;
set(p.Edge, 'ColorBinding','interpolated', 'ColorData', [c{1}, c{3}]);
p = plot([B4(1);B1(1)], [B4(2);B1(2)], 'LineWidth', line_width, 'LineStyle', '-');
drawnow;
set(p.Edge, 'ColorBinding','interpolated', 'ColorData', [c{3}, c{3}]);
p = plot([B1(1);T1(1)], [B1(2);T1(2)], 'LineWidth', line_width, 'LineStyle', '-');
drawnow;
set(p.Edge, 'ColorBinding','interpolated', 'ColorData', [c{3}, c{4}]);
p = plot([B2(1);T2(1)], [B2(2);T2(2)], 'LineWidth', line_width, 'LineStyle', '-');
drawnow;
set(p.Edge, 'ColorBinding','interpolated', 'ColorData', [c{1}, c{2}]);
p = plot([B3(1);T3(1)], [B3(2);T3(2)], 'LineWidth', line_width, 'LineStyle', '-');
drawnow;
set(p.Edge, 'ColorBinding','interpolated', 'ColorData', [c{1}, c{2}]);
p = plot([B4(1);T4(1)], [B4(2);T4(2)], 'LineWidth', line_width, 'LineStyle', '-');
drawnow;
set(p.Edge, 'ColorBinding','interpolated', 'ColorData', [c{3}, c{4}]);
p = plot([T1(1);T2(1)], [T1(2);T2(2)], 'LineWidth', line_width, 'LineStyle', '-');
drawnow;
set(p.Edge, 'ColorBinding','interpolated', 'ColorData', [c{4}, c{2}]);
p = plot([T2(1);T3(1)], [T2(2);T3(2)], 'LineWidth', line_width, 'LineStyle', '-');
drawnow;
set(p.Edge, 'ColorBinding','interpolated', 'ColorData', [c{2}, c{2}]);
p = plot([T3(1);T4(1)], [T3(2);T4(2)], 'LineWidth', line_width, 'LineStyle', '-');
drawnow;
set(p.Edge, 'ColorBinding','interpolated', 'ColorData', [c{2}, c{4}]);
p = plot([T4(1);T1(1)], [T4(2);T1(2)], 'LineWidth', line_width, 'LineStyle', '-');
drawnow;
set(p.Edge, 'ColorBinding','interpolated', 'ColorData', [c{4}, c{4}]);
hold off;
drawnow;
