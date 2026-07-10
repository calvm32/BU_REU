clear all; close all;

%% This scripts runs over a range of epsilon and mu0 and saves
% a history of each runs L2 data, the dominate modes over that time,
% and a history of the first five amplitudes.

%% 1. Grid Parameters
EP = logspace(-6, -2, 100);       
MU0 = linspace(-0.5, 0, 100); 

[EpsMesh, Mu0Mesh] = meshgrid(EP, MU0);
flat_EP  = EpsMesh(:);
flat_MU0 = Mu0Mesh(:);  

num_total_sims = length(flat_EP);

% Preallocate flat result arrays
flat_DMODE_HIST = cell(num_total_sims, 1);
flat_L2_HIST = cell(num_total_sims, 1);
flat_AMP_HISTS = cell(num_total_sims, 1);

%% Simulation Configuration
dt0 = 0.02; 
mass = 0; 
muf = 2.0; 
xscale = 10; 
Lx = xscale*pi; 
Nx = 2^10;

dx = 2*Lx/Nx; 
x = (-Nx/2:Nx/2-1)*dx; 

terminate_thr = 5e0;
    
kx = pi*[0:Nx/2-1 -Nx/2:-1]/Lx;
Laplacian_hat = -kx.^2; 
L_operator = -(Laplacian_hat.^2);

kx_max = max(abs(kx)); 
dealias_mask = abs(kx) <= (2/3)*kx_max;

k_j = pi * [0:10] / Lx;
uh0 = ones(size(x)); 
u0 = ifft(uh0); 
u0 = u0 - mean(u0); 
u_hat0 = fft(u0);

% Start the parallel pool
p = gcp(); 

%% --- PROGRESS BAR SETUP ---
q = parallel.pool.DataQueue;

% Define the callback function that executes on the main thread
% We use a local counter wrapped inside an anonymous function
progress_counter = 0;
fprintf('Progress:   0.0%%');

afterEach(q, @(varargin) update_parallel_progress(num_total_sims));

function update_parallel_progress(num_tasks)
    % Persistent variables retain their value between function calls
    persistent progress_counter
    
    if isempty(progress_counter)
        progress_counter = 0;
    end
    
    progress_counter = progress_counter + 1;
    percent = (progress_counter / num_tasks) * 100;
    
    % This deletes exactly 6 characters (e.g., '  0.0%') and replaces them.
    % Using %5.1f%% guarantees the string width is always exactly 6 characters.
    fprintf('\b\b\b\b\b\b%5.1f%%', percent);
    
    % Clean up when finished
    if progress_counter == num_tasks
        fprintf(' -> Done!\n');
        progress_counter = []; % Reset the counter so it works on the next run
    end
end
%% ---------------------------

%% Parallel Sweep Loop
parfor idx = 1:num_total_sims
    ep  = flat_EP(idx);
    mu0 = flat_MU0(idx);
   
    dt = dt0 * 1e-3 / ep;
   
    T = (muf - mu0)/ep;
    
    mu = @(t) mu0 + t * ep;
    
    nonlin_op = @(u_hat, t) dealias_mask .* Laplacian_hat .* ...
        fft( real(ifft(u_hat)).^3 - mu(t)*real(ifft(u_hat)) );       
    
    problem = EvolutionProblem(L_operator, nonlin_op, u_hat0, [0, T]);
    solver = ETDRK4Solver(16);
    
    monitor = BifurcationMonitorHeadless(mu, kx, Lx, Nx);
    % In case you want to save videos/plots:
    % monitor = BifurcationMonitorInMu(mu, kx, Lx, Nx, x, k_j, save_video=true, video_filename='sim_movie.avi', video_framerate=30);
    termination_event = @(u, t) monitor.l2_history(monitor.step_idx) >= terminate_thr;

    evolution_solve(problem, solver, dt, ...
                    save_every=-1, ...
                    termination_event=termination_event, ...
                    monitors=monitor);
    
    dominate_modes = monitor.dominate_mode(1:monitor.dom_mode_ind, :);

    flat_TIME_HIST(idx) = {monitor.t_grid(1:monitor.step_idx)}
    flat_DMODE_HIST(idx) =  {dominate_modes};
    flat_L2_HIST(idx) = {monitor.l2_history(1:monitor.step_idx)};
    flat_AMP_HISTS(idx) = {monitor.amp_history(:, 1:monitor.step_idx)}

    %% --- SEND UPDATE TO QUEUE ---
    send(q, idx);
end

TIME_HIST = reshape(flat_TIME_HIST, length(EP), length(MU0));
DMODE_HIST = reshape(flat_DMODE_HIST, length(EP), length(MU0));
L2_HIST = reshape(flat_L2_HIST, length(EP), length(MU0));
AMP_HISTS = reshape(flat_AMP_HISTS, length(EP), length(MU0));

%% Save Collective 2D Grid Results
save('grid_sample.mat', 'TIME_HIST', 'L2_HIST', 'DMODE_HIST', 'AMP_HISTS', 'EP', 'MU0', 'Lx', 'Nx', 'terminate_thr');