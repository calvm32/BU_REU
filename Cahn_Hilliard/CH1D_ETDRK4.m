function CH1D_ETDRK4()
    clc; clear; close all;

    %% Parameters
    epsilon = 0.05; % used for scaling mu
    blowup_time = epsilon^(-2/3);

    t0 = -2.0;
    T  = blowup_time + 100.0;
    dt = 0.1;

    scale = 100;
    Lx = scale*pi;

    Nx = 2^11;
    dx = Lx/Nx;

    plot_dt = 0.5; 
    plot_every = round(plot_dt / dt); % make multiple of dt


    %% functions
    function mu_val = mu(t)
        mu_val = t*epsilon;
    end

    function P = power_spec(t, Laplacian_k, u_hat)
        ksq = -Laplacian_k;
        P = abs(u_hat).^2 .* exp(t^2*ksq*epsilon - 2*t*ksq.^2);
    end

    function n = domainwall_density_theory(t, Laplacian_k, u_hat)
        P = power_spec(t, Laplacian_k, u_hat);
        ksq = -Laplacian_k;

        num = sum(ksq.*P,'all');
        den = sum(P,'all');

        n = (1/pi)*sqrt(num/den);
    end

    function density = domainwall_density_computed(u, domain_length)
        walls = sum(u .* circshift(u,-1) < 0);
        density = walls / domain_length;
    end

    function F = free_energy(u, mu, dx)
        ux = gradient(u, dx);
        density = 0.25*u.^4 -0.5*mu*u.^2 +0.5*ux.^2;
        F = sum(density)*dx;
    
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
    fig = figure('Position',[100 100 1400 450]);

    %% Pre-allocate tracking metrics
    l2_hist = zeros(1, num_time_steps);
    l2_hist(1) = l2_norm_periodic_1D(u_hat, Lx);
    
    computed_density = zeros(1, num_time_steps);
    computed_density(1) = domainwall_density_computed(u, Lx);

    theory_density = zeros(1, num_time_steps);
    theory_density(1) = domainwall_density_theory(t0,Laplacian_hat, u_hat);

    energy_hist = zeros(1, num_time_steps);
    energy_hist(1) = free_energy(u0, mu(t0), dx);
    
    %% Subplot 1: Solution
    ax1 = subplot(2,2,1);
    u_line = plot(ax1, x, u, 'LineWidth', 1.5);

    ylim(ax1, [-1.2 1.2]); 
    xlabel(ax1, 'x'); 
    ylabel(ax1, 'u');

    hTitle1 = title(ax1, sprintf('t = %.3f, \\mu = %.3f', t0, mu(t0)));
    hTitle1.Interpreter = 'tex';

    grid(ax1, 'on');

    %% Subplot 2: Mean value
    ax2 = subplot(2,2,2);
    l2_line = plot(ax2, t(1), l2_hist(1), 'LineWidth', 2);

    xlabel(ax2, 't'); 
    ylabel(ax2, '||u||_{L^2}');
    title(ax2, sprintf('L2 Norm of u'));

    grid(ax2, 'on');

    %% Subplot 3: Fourier Spectrum
    ax3 = subplot(2,2,3);

    theory_density_line = plot(ax3, t(1), theory_density(1), ...
        'LineWidth',2,'DisplayName','theoretical');

    hold(ax3, 'on');

    computed_density_line = plot(ax3, t(1), computed_density(1), ...
        'LineWidth',2,'DisplayName','computed');

    title(ax3, sprintf('Domain Wall Densities'));
    grid(ax3, 'on');
    legend(ax3, 'show');

    %% Subplot 4: Free Energy
    ax4 = subplot(2,2,4);
    
    energy_line = plot(ax4, t(1), energy_hist(1), 'LineWidth',2,'DisplayName');

    title(ax3, sprintf('Free Energy'));
    grid(ax4, 'on');
    legend(ax4, 'show');

    drawnow limitrate

    %% GIF setup
    save_gif = true;
    gif_filename = sprintf('cahn_hilliard_1D_epsilon=%.2f_t0=%.2f.gif', epsilon, t0);
    if save_gif && exist(gif_filename, 'file') % delete file if it exists
        delete(gif_filename); 
    end

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
        
        % Update data arrays
        computed_density(n) = domainwall_density_computed(u,Lx);
        theory_density(n) = domainwall_density_theory(t_curr, Laplacian_hat, u_hat);
        l2_hist(n) = l2_norm_periodic_1D(u_hat, Lx);

        % Update figures
        if mod(n - 1, plot_every) == 0
            u_line.YData = u;
            
            hTitle1.String = sprintf('t = %.3f, \\mu = %.3f', t(n), mu(t(n)));
            hTitle1.Interpreter = 'tex';
            
            l2_line.XData = t(1:n);
            l2_line.YData = l2_hist(1:n);
            axis(ax2, 'tight');

            blowup_mask = t(1:n) >= blowup_time;

            theory_density_line.XData = t(blowup_mask);
            theory_density_line.YData = theory_density(blowup_mask);
  
            computed_density_line.XData = t(blowup_mask);
            computed_density_line.YData = computed_density(blowup_mask);

            energy_hist(n) = free_energy(u, mu(t(n)), dx);
            energy_line.XData = t(1:n);
            energy_line.YData = energy_hist(1:n);

            axis(ax3, 'tight');
            
            drawnow limitrate 
            
            if save_gif
                exportgraphics(fig, gif_filename, 'Append', true);
            end
        end
    end

    disp(['GIF saved to: ', gif_filename])

end

%% L2 norm

function val = l2_norm_periodic_1D(u_hat,Lx)

    Nx = length(u_hat);

    val = sqrt(Lx) * ...
          sqrt(sum(abs(u_hat).^2))/Nx;

end