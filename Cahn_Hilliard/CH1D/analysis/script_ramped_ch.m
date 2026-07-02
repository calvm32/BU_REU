clear all; close all;



% range of epsilons
EP = 10.^[-6:0.25:-2]; %10.^[-4.5:1:-4];
%%store dominant mode and mu threshold for each epsilon
DMODE = zeros(length(EP),1);
DMODE_final = zeros(length(EP),1);
MUTHR = zeros(length(EP),1);

%% Save the results of each run (including solution) individually
save_each_run = false;

%% Parameters
dt = 0.02;
mass = 0;
mu0 = -0.3;
muf = 2.0;
muf0 = muf;
xscale = 10;
Lx = xscale*pi;
Nx = 2^10;

% spatial grid
dx = 2*Lx/Nx;
x = (-Nx/2:Nx/2-1)*dx;


l2_thr = 1e0; %when to say the solution has become large amplitude nonlinear quasistationary state

%% Video paramaters
plot_dmu = 0.005; 
save_video = true;


%% Fourier wavenumbers + linear operator
kx = pi*[0:Nx/2-1 -Nx/2:-1]/Lx;
Laplacian_hat = -kx.^2;
L_operator = -(Laplacian_hat.^2);

kx_max = max(abs(kx));
dealias_mask = abs(kx) <= (2/3)*kx_max;

%% Find k_j's for initial data
k_j = pi * [0:10] / Lx;
k = k_j(10);

[MUS,KS] = critical_bifurcation([1:10], 2 * Lx, mass, mu0);
display(MUS)


%% Initial Conditions

% % Zero mean white noise
% noise = 0.2 * (rand(1, Nx) - 0.5);
% noise = noise - mean(noise);


% % Zero mean white noise
% u0 = mass + noise;


%%% Start at a pure mode
%u0 = 0.01*cos(k * x) + mass;



% %all modes on
uh0 = ones(size(x));
u0 = ifft(uh0);
u0 = u0 - mean(u0);
u_hat0 = fft(u0);


%%%%% For loop over range of epsilons
for ii = 1:length(EP)
    ep = EP(ii);
    disp(['Running simulation for epsilon = ', num2str(ep)]);
    if ep < 1e-3
            muf = 0.5;
        else muf = muf0;
    end

    T = (muf - mu0)/ep;
    %muf = mu0+ep*T;%T = (muf - mu0)/ep

    plot_every = round(plot_dmu / ep / dt); % make multiple of dt

    mu = @(t) mu0 + t * ep;

    % Define problem
    % Nonlinear operator: dealias_mask .* Laplacian_hat .* fft( u^3 - mu*u )
    nonlin_op = @(u_hat, t) dealias_mask .* Laplacian_hat .* fft( real(ifft(u_hat)).^3 - mu(t)*real(ifft(u_hat)) );       
    problem = EvolutionProblem(L_operator, nonlin_op, u_hat0, [0, T]);

    % Solver
    solver = ETDRK4Solver(16);

    % Monitor params
    params.mu = mu;
    params.x = x;
    params.kx = kx;
    params.Lx = Lx;
    params.k_j = k_j;
    params.plot_every = plot_every;
    params.save_video = save_video;
    params.video_filename = sprintf('ep=%.5f_mu0=%.2f.avi', ep, mu0);
    params.video_framerate = 30;

    % Use BifurcationMonitorNoPlot if you don't want plotting/video
    % Use BifurcationMonitorInMu if you do
    monitor = BifurcationMonitorNoPlot(params);

    % Execute
    sol = evolution_solve(problem, solver, dt, save_every=plot_every, monitors=monitor);
    
    % Find high amplitude time
    j_thr = find(monitor.l2_history > l2_thr, 1);
    t_thr = monitor.t_grid(j_thr);
    mu_thr = mu(t_thr);

    % Find dominate mode at high amplitude
    dominate_modes = monitor.dominate_mode(1:monitor.dom_mode_ind, :);
    j_dom_mode_at_thr = find(dominate_modes(:, 1) < t_thr, 1, 'last');
    time_dom_mode_at_thr = dominate_modes(j_dom_mode_at_thr, 1);
    dom_mode = dominate_modes(j_dom_mode_at_thr, 2);  
    dom_mode_final = dominate_modes(end, 2);
    
    DMODE_final(ii) = dom_mode_final;
    DMODE(ii) = dom_mode;
    MUTHR(ii) = mu_thr;
    
    % Save just this run
    if save_each_run
        save(sprintf('dom_mode_ep=%.5f_mu0=%.2f.mat', ep, mu0), 'mu_thr', 'dom_mode','dom_mode_final', 'ep', 'kx', 'Lx', 'mu0', 'muf','l2_thr','mass', 'Nx', 'dt', 'T', 'plot_dmu','u0', 'sol');

    end
end

figure(20)
plot(EP,DMODE,'-o','LineWidth',2)

% Save collective run data
save('dominate_modes.mat', 'MUTHR', 'DMODE','DMODE_final', 'EP', 'kx', 'Lx', 'mu0', 'muf','l2_thr','mass', 'Nx', 'dt', 'T', 'plot_dmu','u0');



function [MUS, KS] = critical_bifurcation(j, L, mass,mu0)
    % Computes the j'th frequencies critical mu and associated eigenvalue
    KS = 2 * pi * j./ L;
    MUS = -mu0 + 2*(3 * mass^2 + KS.^2);    
end