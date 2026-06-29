function freezeout()
    clc; clear; close all;

    %% Parameters
    epsilon = 0.01; % used for scaling mu
    blowup_time = epsilon^(-2/3);

    t0 = -2.0;
    T  = 150;
    dt = 0.1;

    scale = 100;
    Lx = scale*pi;
    Nx = 2^11;
    dx = Lx/Nx;

    plot_dt = 0.5; 
    plot_every = round(plot_dt / dt);

    % Initial condition
    sigma = 0.01;
    u0 = sigma*randn(1,Nx);
    u0_hat = fft(u0);

    % discretization
    x = (-Nx/2:Nx/2-1)*dx;

    % Fourier wavenumbers
    kx = 2*pi*[0:Nx/2-1 -Nx/2:-1]/Lx;
    Laplacian_hat = -kx.^2;
    L_operator = -(Laplacian_hat.^2);

    kx_max = max(abs(kx));
    dealias_mask = abs(kx) <= (2/3)*kx_max;

    %% 1. Define the Evolution Problem
    mu = @(t) t*epsilon;
    
    % Nonlinear operator: dealias_mask .* Laplacian_hat .* fft( u^3 - mu(t)*u )
    nonlin_op = @(u_hat, t) dealias_mask .* Laplacian_hat .* fft( real(ifft(u_hat)).^3 - mu(t)*real(ifft(u_hat)) );

    problem = EvolutionProblem(L_operator, nonlin_op, u0_hat, [t0, T]);

    %% 2. Setup Solver
    solver = ETDRK4Solver(32); % M = 32

    %% 3. Setup Monitor
    params.epsilon = epsilon;
    params.blowup_time = blowup_time;
    params.Lx = Lx;
    params.Nx = Nx;
    params.dx = dx;
    params.Laplacian_hat = Laplacian_hat;
    params.plot_every = plot_every;
    params.x = x;
    params.kx = kx;
    params.mu = mu;
    params.save_video = true;
    params.video_filename = sprintf('CH1D_freezeout_epsilon=%.2f_t0=%.2f.mp4', epsilon, t0);

    monitor = FreezeoutMonitor(params);

    %% 4. Execute the Solve
    disp('Starting integration (CH1D Freezeout)...');
    sol = evolution_solve(problem, solver, dt, save_every=plot_every, monitors=monitor);
    disp(['Integration complete. Video saved to: ', params.video_filename]);
end