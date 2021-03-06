function CompilePrCudaMex
% Compile PR CUDA mex file

global ocv_install_dir egn_install_dir

% Change to the current script directory
cur_dir = pwd;
[folder, ~, ~] = fileparts(mfilename('fullpath'));
cd(folder);
disp('==> compiling ''prCudaMex.cpp''');
os = computer('arch');
switch os
	case 'win64'
        myCppCompiler = mex.getCompilerConfigurations('C++','Selected');
        if ~numel(ocv_install_dir)
            temp_ocv_install_dir = 'C:\opencv\build\install';
            if exist(temp_ocv_install_dir, 'dir')
                ocv_install_dir = temp_ocv_install_dir;
            else
                ocv_install_dir = uigetdir('C:\', 'OpenCV Install Folder');
            end
        end
        ocv_inc_dir = ['-I', ocv_install_dir, '\include'];
        ocv_lib_dir = ['-L', ocv_install_dir, '\x64\vc', myCppCompiler.Version(1:end-2), '\lib'];
        lib_list = dir([ocv_lib_dir(3:end), '\*.lib']);
        [~,ocv_world_name,~] = fileparts(lib_list(1).name);
        ocv_world_lib = ['-l', ocv_world_name];
        ocv_world_dll = [ocv_world_name, '.dll'];
        if ~exist(['..\..\', ocv_world_dll], 'file')
            copyfile([ocv_lib_dir(3:end-4), '\bin\', ocv_world_dll], ['..\..\', ocv_world_dll]);
        end
        if ~numel(egn_install_dir) 
            temp_egn_install_dir = 'C:\Eigen';
            if exist(temp_egn_install_dir, 'dir')
                egn_install_dir = temp_egn_install_dir;
            else
                egn_install_dir = uigetdir('C:\', 'Eigen Install Folder');
            end
        end
        egn_inc_dir = ['-I', egn_install_dir];
        eval(['mexcuda ', ...
              ocv_inc_dir, ' ', ...
              egn_inc_dir, ' ', ...
              ocv_lib_dir, ' ', ...
              ocv_world_lib, ' ', ...
              '-llibmwocvmex -lcublas ', ...
              'prCudaMex.cpp poseRefinement.cu prPreCal.cu gradientDescentRefinement.cu prUtil.cpp']);
    case 'glnxa64'
        egn_inc_dir = '-I/usr/include/eigen3';
        eval(['mexcuda ', ...
              egn_inc_dir, ' ', ...
              '-lopencv_world -lmwocvmex -lcublas ', ...
              'prCudaMex.cpp poseRefinement.cu prPreCal.cu gradientDescentRefinement.cu prUtil.cpp']);
    case 'maci64'
        egn_inc_dir = '-I/usr/include/eigen3';
        eval(['mexcuda ', ...
              egn_inc_dir, ' ', ...
              '-lopencv_world -lmwocvmex -lcublas ', ...
              'prCudaMex.cpp poseRefinement.cu prPreCal.cu gradientDescentRefinement.cu prUtil.cpp']);
end


disp('==> DONE!');
fprintf('\n');

% Change back to the previous directory
cd(cur_dir);
