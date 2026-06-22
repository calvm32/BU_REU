function CH1D_bifurcation_varying_mu()
    clc; clear; close all;

    function [mu_j, k_j] = critical_bifurcation(j, L, mass)
        % Computes the j'th frequencies critical mu and associated eigenvalue
        k_j = 2 * pi * j / L
        mu_j = 3 * mass^2 + k_j^2;    
    end

   
    %% Parameters
    save_video = true;

    t0 = -10.0;
    T  = 150;
    dt = 0.1;
    epsilon = 0.05;
    mu = @(t) epsilon * t;

    mass = 0;

    xscale = 2;

    Lx = xscale*pi;

    Nx = 2^12;

    dx = 2*Lx/Nx;

    plot_dt = 0.2; 
    plot_every = round(plot_dt / dt); % make multiple of dt


    %% Initial condition
    % spatial grid (needed for patterned base state)
    x = (-Nx/2:Nx/2-1)*dx;

    % Zero mean white noise
    u0 = mass + 0.01 * (rand(1, Nx) - 0.5);

    u_hat = fft(u0);
    u = u0;

    %% discretization
    % time discretization
    t = t0:dt:T;
    num_time_steps = length(t);

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

    %% Plot setup
    fig = figure('Position',[100 100 1200 700], 'Resize', 'off');

    %% Pre-allocate tracking metrics
    mean_hist = zeros(1, num_time_steps);
    mean_hist(1) = trapz(u) * dx * 0.5 / Lx;

    dominate_mode = zeros(1, num_time_steps);
    [~, max_index] = max(abs(fft(u - mass)));
    dominate_mode(1) = kx(max_index);

    %% Subplot 1: Solution (Spans the entire top row)
    ax1 = subplot(2,3,[1 3]);
    hLine1 = plot(ax1, x, u, 'LineWidth', 1.5);

    xlabel(ax1, 'x'); 
    ylabel(ax1, 'u');
    hTitle1 = title(ax1, sprintf('t = %.3f', t0));

    grid(ax1, 'on');
    
    %% Subplot 2: Mean value (Bottom Left)
    ax2 = subplot(2,3,4);
    hLine2 = plot(ax2, t(1), mean_hist(1), 'LineWidth', 2);

    xlabel(ax2, 't'); 
    ylabel(ax2, 'Mass');

    grid(ax2, 'on');
    
    %% Subplot 3: Dominate mode (Bottom Center)
    ax3 = subplot(2,3,5);
    hLine3 = plot(ax3, t(1), dominate_mode(1), 'LineWidth', 2);

    xlabel(ax3, 't'); 
    ylabel(ax3, 'Wavenumber (k)');

    ylim(ax3, [0, 4]); 

    grid(ax3, 'on');

    drawnow limitrate

    %% Subplot 4: Fourier Spectrum (Bottom Right)
    ax4 = subplot(2,3,6);
    hFreq = plot(ax4, kx, abs(u_hat), 'LineWidth', 2);

    xlabel(ax4, 'k'); 
    ylabel(ax4, '$|\hat{u}|$', 'Interpreter', 'latex');

    set(ax4, 'YScale', 'log');
    xlim(ax4, [-20, 20]); 

    grid(ax4, 'on');

    drawnow limitrate

    %% GIF setup
    video_filename = sprintf('CH_slow_mu_epsilon=%.3f_t0=%0.2f.mp4', epsilon, t0);
    
    if save_video
        % Initialize the MP4 writer using the H.264 codec
        v = VideoWriter(video_filename, 'MPEG-4');
        v.FrameRate = 45;  % Target frames per second
        v.Quality = 100;   % Highest crispness, lowest compression artifacts
        open(v);
    end

    %% Time loop

    for n = 2:num_time_steps
        % Stage 1
        u3_nonlinear = u.^3 - mu(t(n))*u;
        Nu_hat = dealias_mask .* Laplacian_hat .* fft(u3_nonlinear);
        
        % Stage 2
        a_hat = E2.*u_hat + Q.*Nu_hat;
        a = real(ifft(a_hat));
        Na_hat = dealias_mask .* Laplacian_hat .* fft(a.^3 - mu(t(n))*a);
        
        % Stage 3
        b_hat = E2.*u_hat + Q.*Na_hat;
        b = real(ifft(b_hat));
        Nb_hat = dealias_mask .* Laplacian_hat .* fft(b.^3 - mu(t(n))*b);
        
        % Stage 4
        c_hat = E2.*a_hat + Q.*(2*Nb_hat - Nu_hat);
        c = real(ifft(c_hat));
        Nc_hat = dealias_mask .* Laplacian_hat .* fft(c.^3 - mu(t(n))*c);
        
        % Final Time Step Combination
        u_hat = E.*u_hat + f1.*Nu_hat + 2*f2.*(Na_hat + Nb_hat) + f3.*Nc_hat;
        u = real(ifft(u_hat));

        mean_hist(n) = trapz(u) * dx * 0.5 / Lx;
        [~, max_index] = max(abs(fft(u - mass)));
        dominate_mode(n) = kx(max_index);


        % Update figures
        if mod(n - 1, plot_every) == 0
            hLine1.YData = u;
            hTitle1.String = sprintf('t = %.4f, mu = %.4f', t(n), mu(t(n)));
            axis(ax2, 'tight');

            
            hLine2.XData = t(1:n);
            hLine2.YData = mean_hist(1:n);
            axis(ax2, 'tight');
            
            hLine3.XData = t(1:n);
            hLine3.YData = dominate_mode(1:n);

            hFreq.YData = abs(u_hat);
            %axis(ax4, 'tight');

           
            drawnow; 
            
            if save_video
                % Capture the current high-res figure frame and write to file
                frame = getframe(fig);
                writeVideo(v, frame);
            end
        end
    end

    disp(['Video saved to: ', video_filename])
end


