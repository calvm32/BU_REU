clc; clear; close all;

%% Parameters
epsilon = 0.01; % used for scaling mu
blowup_time = epsilon^(-2/3);

t0 = -2.0;
T = 150;
dt = 0.1;

scale = 100;
Lx = scale*pi;

Nx = 2^11;
dx = Lx/Nx;

plot_dt = 0.5; 
plot_every = round(plot_dt / dt); % make multiple of dt

%% functions
function mu_val = mu(t, epsilon)
    mu_val = t*epsilon;
end

function P = power_spec(t, Laplacian_k, u_hat, epsilon)
    ksq = -Laplacian_k;
    P = abs(u_hat).^2 .* exp(t^2*ksq*epsilon - 2*t*ksq.^2);
end

function n = domainwall_density_theory(t, Laplacian_k, u_hat, epsilon)
    P = power_spec(t, Laplacian_k, u_hat, epsilon);
    ksq = -Laplacian_k;

    num = sum(ksq.*P,'all');
    den = sum(P,'all');

    n = (1/pi)*sqrt(num/den);
end

function n = domainwall_density_computed(u, domain_length)
    walls = sum(u .* circshift(u,-1) < 0);
    n = walls / domain_length;
end

%% Initial condition
sigma = 0.01;
u0 = sigma*randn(1,Nx);

u_hat = fft(u0);
u = u0;

%% discretization
% time discretization
t = t0:dt:T;
num_time_steps = length(t);

% spatial grid
x = (-Nx/2:Nx/2-1)*dx;

%% Fourier wavenumbers
kx = 2*pi*[0:Nx/2-1 -Nx/2:-1]/Lx;
Laplacian_hat = -kx.^2;

% Linear operator:
L_operator = -(Laplacian_hat.^2);

%% 2/3 dealiasing mask
kx_max = max(abs(kx));
dealias_mask = abs(kx) <= (2/3)*kx_max;

%% ETDRK4 setup
E  = exp(dt*L_operator);
E2 = exp(dt*L_operator/2);
M = 32; % no. of points for complex means
r = exp(1i*pi*((1:M)-0.5)/M); % roots of unity
Lvec = L_operator(:);
LR = dt*Lvec(:,ones(M,1)) + r(ones(numel(L_operator),1),:);

Q  = dt*real(mean((exp(LR/2)-1)./LR,2)).';
f1 = dt*real(mean((-4-LR + exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2)).';
f2 = dt*real(mean((2+LR + exp(LR).*(-2+LR))./LR.^3 ,2)).';
f3 = dt*real(mean((-4-3*LR-LR.^2 + exp(LR).*(4-LR))./LR.^3 ,2)).';

clear LR Lvec

%% Plot setup
fig = figure('Position',[100 100 1100 1100]);
set(fig,'Resize','off');

%% Pre-allocate tracking metrics
l2_hist = zeros(1, num_time_steps);
l2_hist(1) = l2_norm_periodic_1D(u_hat, Lx);

computed_density = zeros(1, num_time_steps);
computed_density(1) = domainwall_density_computed(u, Lx);

theory_density = zeros(1, num_time_steps);
theory_density(1) = domainwall_density_theory(t0,Laplacian_hat, u_hat, epsilon);

%% Subplot 1: Solution
ax1 = subplot(2,2,[1,2]);
u_line = plot(ax1, x, u, 'LineWidth', 1.5);

ylim(ax1, [-1.2 1.2]); 
xlabel(ax1, 'x'); 
ylabel(ax1, 'u(x)');
hTitle1 = title(ax1, sprintf('t = %.3f, \\mu = %.3f, \\epsilon = %.2f', t0, mu(t0, epsilon), epsilon));
hTitle1.Interpreter = 'tex';
grid(ax1, 'on');

%% Subplot 2: Domain Wall Density
ax3 = subplot(2,2,3);

theory_density_line = plot(ax3, t(1), theory_density(1), 'LineWidth',2,'DisplayName','theoretical');
hold(ax3, 'on');
computed_density_line = plot(ax3, t(1), computed_density(1), 'LineWidth',2,'DisplayName','computed');

freeze_points = plot(ax3,NaN,NaN,...
    'ro',...
    'MarkerSize',8,...
    'LineWidth',2,...
    'Color', 'black', ...
    'DisplayName','$\hat{t}$');

eq_points = plot(ax3,NaN,NaN,...
    'rs',...
    'MarkerSize',8,...
    'LineWidth',2,...
    'Color', 'black', ...
    'DisplayName','$\hat{t}_{eq}$');

title(ax3, sprintf('Domain Wall Densities'));
xlabel(ax3, 't'); 
ylabel(ax3, 'n(t)');
lgd = legend(ax3,'show','Location','northwest');
lgd.Interpreter = 'latex';
fontsize(lgd, scale=1.4);
grid(ax3, 'on');

%% Subplot 3: Fourier Modes
ax4 = subplot(2,2,4);

% shifted spectrum for plotting
u_hat_shift = fftshift(u_hat);
k_shift = fftshift(kx);
[~,idx] = max(abs(u_hat_shift).^2);

fourier_line = semilogy(ax4, k_shift, abs(u_hat_shift).^2, 'LineWidth', 1.5);
fourier_line.HandleVisibility = 'off';
hold(ax4,'on')

peak_point_comput = plot(ax4, k_shift(idx), abs(u_hat_shift(idx)).^2, ...
    'ro','MarkerFaceColor','r', 'DisplayName','computed mode');

k_theory = epsilon^(1/6);
[~, idx_theory] = min(abs(k_shift - k_theory));

peak_point_theory = plot(ax4, k_shift(idx_theory), abs(u_hat_shift(idx_theory)).^2, ...
    'o', 'MarkerFaceColor', 'g', 'DisplayName', 'theoretical mode');
hold(ax4,'off')

dummy = plot(ax3, NaN, NaN, 'w', 'HandleVisibility', 'off');
xlabel(ax4,'k');
ylabel(ax4, '$\|\hat{u}(k)\|^2$', 'Interpreter', 'latex');
title(ax4,'Fourier Spectrum and Dominant Modes');
lgd = legend(ax3, ...
    [theory_density_line, computed_density_line, freeze_points, eq_points], ...
    {'theoretical','computed','$\hat{t}$','$\hat{t}_{eq}$'}, ...
    'Location','northwest');
lgd.Interpreter = 'latex';
lgd.NumColumns = 2;
lgd2 = legend(ax4, 'show', 'Location', 'northwest');
lgd2.Interpreter = 'latex';
fontsize(lgd2, scale=1.4);
grid(ax4,'on');

drawnow limitrate

%% Video setup
save_video = true;
video_filename = sprintf('CH1D_freezeout_epsilon=%.2f_t0=%.2f', epsilon, t0);

if save_video
    v = VideoWriter(video_filename, 'Motion JPEG AVI');
    v.FrameRate = 15;  % Target frames per second
    v.Quality = 100;
    open(v);
end

%% Time loop

for n = 2:num_time_steps
    % Setup intermediate times for RK4 non-autonomous evaluation
    t_prev = t(n-1);
    t_half = t_prev + dt/2;
    t_curr = t(n);
    
    % Stage 1 (Evaluated at t_{n-1})
    u3_nonlinear = u.^3 - mu(t_prev, epsilon)*u;
    Nu_hat = dealias_mask .* Laplacian_hat .* fft(u3_nonlinear);
    
    % Stage 2 (Evaluated at midpoint)
    a_hat = E2.*u_hat + Q.*Nu_hat;
    a = real(ifft(a_hat));
    Na_hat = dealias_mask .* Laplacian_hat .* fft(a.^3 - mu(t_half, epsilon)*a);
    
    % Stage 3 (Evaluated at midpoint)
    b_hat = E2.*u_hat + Q.*Na_hat;
    b = real(ifft(b_hat));
    Nb_hat = dealias_mask .* Laplacian_hat .* fft(b.^3 - mu(t_half, epsilon)*b);
    
    % Stage 4 (Evaluated at t_n)
    c_hat = E2.*a_hat + Q.*(2*Nb_hat - Nu_hat);
    c = real(ifft(c_hat));
    Nc_hat = dealias_mask .* Laplacian_hat .* fft(c.^3 - mu(t_curr, epsilon)*c);
    
    % Final Time Step Combination
    u_hat = E.*u_hat + f1.*Nu_hat + 2*f2.*(Na_hat + Nb_hat) + f3.*Nc_hat;
    u = real(ifft(u_hat));
    
    % Update data arrays
    computed_density(n) = domainwall_density_computed(u,Lx);
    theory_density(n) = domainwall_density_theory(t_curr, Laplacian_hat, u_hat, epsilon);
    l2_hist(n) = l2_norm_periodic_1D(u_hat, Lx);

    % Update figures
    if mod(n - 1, plot_every) == 0
        u_line.YData = u;
        hTitle1.String = sprintf('t = %.3f, \\mu = %.3f, \\epsilon = %.2f', t(n), mu(t(n), epsilon), epsilon);
        hTitle1.Interpreter = 'tex';

        blowup_mask = t(1:n) >= blowup_time;

        theory_density_line.XData = t(blowup_mask);
        theory_density_line.YData = theory_density(blowup_mask);

        computed_density_line.XData = t(blowup_mask);
        computed_density_line.YData = computed_density(blowup_mask);

        axis(ax3, 'tight');
        
        u_hat_shift = fftshift(u_hat);
        fourier_line.XData = k_shift;
        fourier_line.YData = abs(u_hat_shift).^2;
        
        spectrum = abs(u_hat_shift).^2;
        spectrum(spectrum <= 0) = eps;
        
        [~,idx] = max(spectrum);
        
        peak_point_comput.XData = k_shift(idx);
        peak_point_comput.YData = spectrum(idx);
        
        k_theory = epsilon^(1/6);
        [~,idx_theory] = min(abs(k_shift-k_theory));
        
        peak_point_theory.XData = k_shift(idx_theory);
        peak_point_theory.YData = spectrum(idx_theory);

        axis(ax4,'tight')

        drawnow limitrate 
        
        if save_video
            % Capture the current high-res figure frame and write to file
            frame = getframe(fig,[0 0 1100 1100]);
            writeVideo(v, frame);
        end
    end
end

mask = t > 10;
density_masked = computed_density(mask);
max_density = max(density_masked);
tol = 1e-4*max_density;

mask_idx = find(mask);
eq_idx_local = find(density_masked >= max_density - tol,1,'first');
eq_idx = mask_idx(eq_idx_local);
eq_time = t(eq_idx);

eq_points.XData = eq_time;
eq_points.YData = computed_density(eq_idx);

blowup_mask = t(1:n) == blowup_time;
[~, freeze_idx] = min(abs(t - blowup_time));

freeze_points.XData = t(freeze_idx);
freeze_points.YData = theory_density(freeze_idx);

axis(ax4,'tight')

drawnow limitrate 
        
if save_video
    % Capture the current high-res figure frame and write to file
    frame = getframe(fig,[0 0 1100 1100]);
    writeVideo(v, frame);

    close(v);
end

%% L2 norm
function val = l2_norm_periodic_1D(u_hat,Lx)
    Nx = length(u_hat);
    val = sqrt(Lx)* sqrt(sum(abs(u_hat).^2))/Nx;
end