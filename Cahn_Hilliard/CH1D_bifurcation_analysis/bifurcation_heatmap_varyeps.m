clc; clear; close all;

%% Parameters
% heat map params
IC_modes = 0:10;
num_runs = 20;

% simulation params
t0 = -10.0;
T  = 150;
dt = 0.1;
epsilon_list = linspace(0.001,1,20);

mass = 0;

xscale = 8;
Lx = xscale*pi;
Nx = 2^8;
dx = 2*Lx/Nx;

plot_dt = 0.2; 
plot_every = round(plot_dt / dt); % make multiple of dt

k_j = 2 * pi * j / Lx;
mu_j = 3 * mass^2 + k_j.^2; 

%% discretization
% time discretization
t = t0:dt:T;
num_time_steps = length(t);

% spatial grid (needed for patterned base state)
x = (-Nx/2:Nx/2-1)*dx;

%% Fourier wavenumbers
kx = pi*[0:Nx/2-1 -Nx/2:-1]/Lx;
Laplacian_hat = -kx.^2;

% Linear operator:
L_operator = -(Laplacian_hat.^2);

%% 2/3 dealiasing mask
kx_max = max(abs(kx));
dealias_mask = abs(kx) <= (2/3)*kx_max;

%% ETDRK4 setup
E  = exp(dt*L_operator);
E2 = exp(dt*L_operator/2);
M = 16; % no. of points for complex means
r = exp(1i*pi*((1:M)-0.5)/M); % roots of unity
Lvec = L_operator(:);
LR = dt*Lvec(:,ones(M,1)) + r(ones(numel(L_operator),1),:);

Q  = dt*real(mean((exp(LR/2)-1)./LR,2)).';
f1 = dt*real(mean((-4-LR + exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2)).';
f2 = dt*real(mean((2+LR + exp(LR).*(-2+LR))./LR.^3 ,2)).';
f3 = dt*real(mean((-4-3*LR-LR.^2 + exp(LR).*(4-LR))./LR.^3 ,2)).';

clear LR Lvec

%% loop over epsilon
for eidx = 1:length(epsilon_list)
    epsilon = epsilon_list(eidx);
    mu = @(t) epsilon * t;

    Heat = zeros(length(IC_modes), length(IC_modes), length(epsilon_list));

    for ic = 1:length(IC_modes)
        k0 = IC_modes(ic);

        u0 = mass + cos(2*pi*k0*x/(2*Lx)); % + 0.01*(rand(1,Nx)-0.5);

        u_hat = fft(u0);
        u = u0;

        mean_hist = zeros(1, num_time_steps);
        mean_hist(1) = trapz(u0) * dx * 0.5 / Lx;

        dominate_mode = zeros(1, num_time_steps);
        [~, max_index] = max(abs(fft(u0 - mass)));
        dominate_mode(1) = kx(max_index);

        for run = 1:num_runs
                
            %% Time loop
            for n = 2:num_time_steps
                % Setup intermediate times for RK4 non-autonomous evaluation
                t_prev = t(n-1);
                t_half = t_prev + dt/2;
                t_curr = t(n);
                
                % Stage 1 (Evaluated at t_{n-1})
                u3_nonlinear = u.^3 - mu(t_prev)*u;
                Nu_hat = dealias_mask .* Laplacian_hat .* fft(u3_nonlinear);
                
                % Stage 2 (Evaluated at midpoint)
                a_hat = E2.*u_hat + Q.*Nu_hat;
                a = real(ifft(a_hat));
                Na_hat = dealias_mask .* Laplacian_hat .* fft(a.^3 - mu(t_half)*a);
                
                % Stage 3 (Evaluated at midpoint)
                b_hat = E2.*u_hat + Q.*Na_hat;
                b = real(ifft(b_hat));
                Nb_hat = dealias_mask .* Laplacian_hat .* fft(b.^3 - mu(t_half)*b);
                
                % Stage 4 (Evaluated at t_n)
                c_hat = E2.*a_hat + Q.*(2*Nb_hat - Nu_hat);
                c = real(ifft(c_hat));
                Nc_hat = dealias_mask .* Laplacian_hat .* fft(c.^3 - mu(t_curr)*c);
                
                % Final Time Step Combination
                u_hat = E.*u_hat + f1.*Nu_hat + 2*f2.*(Na_hat + Nb_hat) + f3.*Nc_hat;
                u = real(ifft(u_hat));

                mean_hist(n) = trapz(u) * dx * 0.5 / Lx;
                [~, max_index] = max(abs(fft(u - mass)));
                dominate_mode(n) = kx(max_index);

            end

            [~,max_index] = max(abs(fft(u-mass)));
            final_mode = round(kx(max_index));

            row = find(IC_modes==final_mode);

            if ~isempty(row)
                Heat(row,ic,eidx) = Heat(row,ic) + 1;
            end

            disp(fprintf('done w/ run=%.0f/%.0f; IC tested=%.0f/%.0f', run, num_runs, ic, length(IC_modes)));
        end
    end
end

%% plotting
v = VideoWriter('bifurcation_heatmap.avi');
v.FrameRate = length(epsilon_list);
open(v);

for eidx = 1:length(epsilon_list)
    frame = getframe(gcf); 

    imagesc(IC_modes,IC_modes,Heat(:,:,eidx)/num_runs)
    axis xy
    xlabel('IC mode number')
    ylabel('Eq, mode number')

    % display info
    title('Likelihood of equilibrium modes reached for various initial conditions');
    text(0.5, -0.1, sprintf('Parameters: T = %.0f,  t0 = %.0f, \\epsilon = %.3f\nEach IC ran for %.0f trials', T, t0, epsilon, num_runs), 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontSize', 10);

    % grid lines = integers
    xticks(IC_modes(1)-0.5:1:IC_modes(end)+0.5);
    yticks(IC_modes(1)-0.5:1:IC_modes(end)+0.5);
    grid('on')
    GridColor = [0.5 0.5 0.5];

    % custom color map
    levels = num_runs + 1;
    darkBlue = [0.02 0.12 0.55]; 
    white = [1 1 1];

    cmap = [linspace(darkBlue(1),white(1),levels)', ...
            linspace(darkBlue(2),white(2),levels)', ...
            linspace(darkBlue(3),white(3),levels)'];

    colormap(flipud(cmap))
    clim([0 1])
    colorbar
    
    writeVideo(v, frame);
end

close(v);   