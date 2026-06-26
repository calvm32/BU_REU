clc; clear; close all;

function [mu_j, k_j] = critical_bifurcation(j, L, mass)
    % Computes the j'th frequencies critical mu and associated eigenvalue
    k_j = 2 * pi * j / L;
    mu_j = 3 * mass^2 + k_j^2;    
end

%% Parameters
A0 = 1e-1;
t0 = 0.0;
T  = 100;
dt = 0.001;

mass = 0;

xscale = 6;
Lx = xscale*pi;
Nx = 2^12;

[mu, ~] = critical_bifurcation(5, 2 * Lx, mass);

%% Video paramaters
plot_dt = 5.0; 
plot_every = round(plot_dt / dt); % make multiple of dt
save_video = false;

%% Find k_j's for initial data
k_j = pi * [0:7] / Lx;

%% discretization
% time discretization
t = t0:dt:T;
num_time_steps = length(t);

% spatial grid
dx = 2*Lx/Nx;
x = (-Nx/2:Nx/2-1)*dx;

% Fourier wavenumbers
kx = pi*[0:Nx/2-1 -Nx/2:-1]/Lx;
Laplacian_hat = -kx.^2;

%% Linear operator:
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


%% Initial condition

k = k_j(2);

% Zero mean white noise
noise = 0.02 * (rand(1, Nx) - 0.5);
noise = noise - mean(noise);

% Zero mean white noise
%u0 = mass + noise;

% Start at k_bif
u0 = A0 * cos(k * x) + mass;


video_filename = sprintf('k=%.3f_T=%.0f.mp4', k, T);

% Restart initial conditions
u_hat = fft(u0);
u = u0;

%% Plot setup
fig = figure('Position',[100 100 1200 700], 'Resize', 'off');

%% Pre-allocate tracking metrics
% mean_hist = zeros(1, num_time_steps);
% mean_hist(1) = trapz(u) * dx * 0.5 / Lx;

dominate_mode = NaN(20, 2);
[~, max_index] = max(abs(fft(u - mass)));
dominate_mode(1, :) = [t(1)+1e-2, kx(max_index)];
dom_mode_ind = 1;

%% Subplot 1: Solution (Spans the entire top row)
ax1 = subplot(2,2,[1 2]);
hLine1 = plot(ax1, x, u, 'LineWidth', 1.5);

ylim(ax1, 1.2 * [-sqrt(mu) sqrt(mu)]); 
xlabel(ax1, 'x'); 
ylabel(ax1, 'u');
hTitle1 = title(ax1, sprintf('t = %.3f', t0));

grid(ax1, 'on');

%% Subplot 2: Mean value (Bottom Left)
% ax2 = subplot(2,3,4);
% hLine2 = plot(ax2, t(1), mean_hist(1), 'LineWidth', 2);
% 
% xlabel(ax2, 't'); 
% ylabel(ax2, 'Mass');
% 
% grid(ax2, 'on');

%% Subplot 3: Dominate mode (Bottom Center)
ax3 = subplot(2,2,3);

hLine3 = stairs(ax3, dominate_mode(1, 1), dominate_mode(1, 2), 'LineWidth', 2);

labels = cellstr("k_" + (0:numel(k_j)-1));
yline(ax3, k_j, '--r', labels, 'LineWidth', 2);    

xlabel(ax3, 't'); 
ylabel(ax3, 'Wavenumber (k)');

ylim(ax3, [0, k_j(end)]); 
xlim(ax3, [1e-10, T]);

grid(ax3, 'on');
    
%% Subplot 4: Fourier Spectrum (Bottom Right)
ax4 = subplot(2,2,4);
hFreq = plot(ax4, ifftshift(kx), ifftshift(abs(u_hat)), 'LineWidth', 2);

xlabel(ax4, 'k'); 
ylabel(ax4, '$|\hat{u}|$', 'Interpreter', 'latex');

set(ax4, 'YScale', 'log');
xlim(ax4, [-10, 10])
grid(ax4, 'on');

drawnow limitrate

%% GIF setup    
if save_video
    % Initialize the MP4 writer using the H.264 codec
    v = VideoWriter(video_filename, 'MPEG-4');
    v.FrameRate = 30;  % Target frames per second
    v.Quality = 100;   % Highest crispness, lowest compression artifacts
    open(v);

    frame = getframe(fig);
    writeVideo(v, frame);
end

%% Time loop
for n = 2:num_time_steps
    % Stage 1
    u3_nonlinear = u.^3 - mu*u;
    Nu_hat = dealias_mask .* Laplacian_hat .* fft(u3_nonlinear);
    
    % Stage 2
    a_hat = E2.*u_hat + Q.*Nu_hat;
    a = real(ifft(a_hat));
    Na_hat = dealias_mask .* Laplacian_hat .* fft(a.^3 - mu*a);
    
    % Stage 3
    b_hat = E2.*u_hat + Q.*Na_hat;
    b = real(ifft(b_hat));
    Nb_hat = dealias_mask .* Laplacian_hat .* fft(b.^3 - mu*b);
    
    % Stage 4
    c_hat = E2.*a_hat + Q.*(2*Nb_hat - Nu_hat);
    c = real(ifft(c_hat));
    Nc_hat = dealias_mask .* Laplacian_hat .* fft(c.^3 - mu*c);
    
    % Final Time Step Combination
    u_hat = E.*u_hat + f1.*Nu_hat + 2*f2.*(Na_hat + Nb_hat) + f3.*Nc_hat;
    u = real(ifft(u_hat));

    mean_hist(n) = trapz(u) * dx * 0.5 / Lx;
    [~, max_index] = max(abs(fft(u - mass)));
    dom_mode = kx(max_index);

    if dominate_mode(dom_mode_ind, 2) ~= dom_mode
        dom_mode_ind = dom_mode_ind + 1;
        dominate_mode(dom_mode_ind, :) = [t(n), dom_mode];
    end


    % Update figures
    if mod(n - 1, plot_every) == 0
        hLine1.YData = u;
        hTitle1.String = sprintf('t = %.4f', t(n));
        
        % hLine2.XData = t(1:n);
        % hLine2.YData = mean_hist(1:n);
        % axis(ax2, 'tight');
        
        hLine3.XData = [dominate_mode(1:dom_mode_ind, 1); t(n)];
        hLine3.YData = [dominate_mode(1:dom_mode_ind, 2); dominate_mode(dom_mode_ind, 2)];
        %axis(ax3, 'tight');

        hFreq.YData = ifftshift(abs(u_hat));
       
        drawnow limitrate; 
        
        if save_video
            % Capture the current high-res figure frame and write to file
            frame = getframe(fig);
            writeVideo(v, frame);
        end
    end
end

if save_video
    close(v);
    disp(['Video saved to: ', video_filename])
end

fprintf('Surpassing time: %.4f \n', dominate_mode(2, 1))

k1 = k_j(2);
k3 = k_j(4);
lambda_1 = mu * k1^2 - k1^4;
lambda_3 = mu * k3^2 - k3^4;
pred_t_surpass = log( A0^2 * k3^2 / (4 * lambda_3 - 12 * lambda_1) ) / ( lambda_1 - lambda_3)
