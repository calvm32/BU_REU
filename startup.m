% get root dir
project_root = fileparts(mfilename('fullpath'));

% add entire project to MATLAB path
addpath(genpath(project_root));

% optional: save path for future MATLAB sessions
savepath; 