function CompileDpeMex

% Change to the current script directory
cur_dir = pwd;
[folder, ~, ~] = fileparts(mfilename('fullpath'));
cd(folder);
disp('COMPILING MEX FILES...')

disp('==> compiling ''poseToTransMatMex.cpp'' (1 out of 4)');
mex COMPFLAGS='/openmp' OPTIMFLAGS='/O2' poseToTransMatMex.cpp

disp('==> compiling ''createSetMex.cpp'' (2 out of 4)');
mex OPTIMFLAGS='/O2' createSetMex.cpp

disp('==> compiling ''evaluateEaColorMex.cpp'' (3 out of 4)');
mex COMPFLAGS='/openmp' OPTIMFLAGS='/O2' evaluateEaColorMex.cpp

disp('==> compiling ''evaluateEaInvarMex.cpp'' (4 out of 4)');
mex COMPFLAGS='/openmp' OPTIMFLAGS='/O2' evaluateEaInvarMex.cpp

disp('==> DONE!');
fprintf('\n');

% Change back to the previous directory
cd(cur_dir);