clear all; close all;

%% This scripts runs over a range of epsilon and mu0 and saves
% a history of each runs L2 data, the dominate modes over that time,
% and a history of the first five amplitudes.

%% 1. Grid Parameters
EP = logspace(-5, -2, 22);       
MU0 = linspace(-0.5, 0, 100); 

DMODE = zeros(length(EP),1);
DMODE_final = zeros(length(EP),1);
MUTHR = zeros(length(EP),1);

[EpsMesh, Mu0Mesh] = meshgrid(EP, MU0);
flat_EP  = EpsMesh(:);
flat_MU0 = Mu0Mesh(:);  

num_total_sims = length(flat_EP);

%% Simulation Configuration
dt0 = 0.02; 
mass = 0; 
muf = 2.0; 
Lx = 10*pi; 
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

%% PROGRESS BAR SETUP
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
%%------------------------

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
    
    if ep < 1e-4
        solver = ETDRK4Solver(32);
    else
        solver = ETDRK4Solver(16);
    end

    monitor = BifurcationMonitorHeadless(mu, kx, Lx, Nx);
    % In case you want to save videos/plots:
    % monitor = BifurcationMonitorInMu(mu, kx, Lx, Nx, x, k_j, save_video=true, video_filename='sim_movie.avi', video_framerate=30);
    termination_event = @(u, t) monitor.l2_history(monitor.step_idx) >= terminate_thr;

    evolution_solve(problem, solver, dt, ...
                    save_every=-1, ...
                    termination_event=termination_event, ...
                    monitors=monitor);
    
    dominate_modes = monitor.dominate_mode;

    [dom_mode_max, dom_mode_idx] = max(dominate_modes(:, 2));
    mu_thr = mu(dominate_modes(dom_mode_idx, 1));
    DMODE_final(idx) = dom_mode_max;
    MUTHR(idx) = mu_thr;

    filename = fsprintf('sample_ep=1e%.5f_mu0=%.7f_%s', log10(ep), mu0, 'Float64');
    s = struct("DMODE_HIST", dominate_modes, ...
               "L2_HIST", monitor.l2_history(1:monitor.step_idx), ...
               "AMP_HISTS", monitor.amp_history(:, 1:monitor.step_idx), ...
               "ep", ep, "mu0", mu0, "Lx", Lx, "Nx", Nx, "dt", dt, ...
               "terminate_thr", terminate_thr, 'datatype', 'Float64')

    save(filename, "-fromstruct", s);

    %% SEND UPDATE TO QUEUE
    send(q, idx);
end