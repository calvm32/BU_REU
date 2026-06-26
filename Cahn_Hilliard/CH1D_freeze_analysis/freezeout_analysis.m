clc; clear; close all;

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Description and configuration
% Find how domain wall density corresponds with different values (freezeout time, largest modes, etc.)
% and how theoretical values compare with computed values
% finally, vary epsilon and see how all of those stats change

% 1: density and L2 ( plot L2 norm VS. time, plot domain wall density VS. time, plot solution VS. time )
% 2: density and fourier modes ( plot fourier modes VS. time, plot domain wall density VS. time, plot solution VS. time )
% 3: analyzing averages for diff. epsilon ( plot theoretical vs. computed values of density, freezout time, and largest wave mode )

type = 3;

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Parameters
epsilon = 0.025; % type == 1 or 2
epsilon_list = linspace(0.1, 5, 50); % type == 3

plot_diff = false; % valid for type == 3

t0 = -2.0;
dt = 0.1;

scale = 100;
Lx = scale*pi;

Nx = 2^12;
dx = Lx/Nx;

% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------

%% Initial condition
% random
sigma = 0.01;
u0 = sigma*randn(1,Nx);

u = u0;
u_hat = fft(u0);

% gaussian
% u0 = exp(-(X.^2)/sigma);
% u_hat = fft(u0);

%% discretization
% spatial discretization
x = (-Nx/2:Nx/2-1)*dx;

%% Fourier wavenumbers
kx = 2*pi*[0:Nx/2-1 -Nx/2:-1]/Lx;
Laplacian_k = -(kx.^2);

%% 2/3 dealiasing mask
kx_max = max(abs(kx));
dealias_mask = abs(kx) <= (2/3)*kx_max;

switch type
    case 1
        plot_dt = 0.5; 
        plot_every = round(plot_dt / dt); % make multiple of dt

        % time discretization
        blowup_time = epsilon^(-2/3);
        T = blowup_time + 20.0; % used for singular eps cases

        dt = 0.1 * (dx)^2;
        t = t0:dt:T;
        num_time_steps = length(t);

        %% Plot setup
        figure('Position',[100 100 1600 500])
        %% first plot: solution
        subplot(1,3,1)

        u_phys = real(ifft(u_hat));

        hsol = plot(x,u_phys,'LineWidth',1.5);

        ylim([-1.2 1.2])

        xlabel('x')
        ylabel('u')

        title(sprintf('t = %.3f',t0))

        grid on

        %% second plot: L2 norm
        subplot(1,3,2)

        t_hist = t0;
        l2_hist = l2_norm_periodic_1D(u_hat,Lx);

        hline = plot(t_hist,l2_hist,'LineWidth',2);

        xlabel('t')
        ylabel('||u||_{L^2}')

        grid on

        %% third plot: domain wall density
        subplot(1,3,3)

        comput_density = domainwall_density_comput(u_phys,Lx);
        theory_density = domainwall_density_theory(t0,Laplacian_k,u_hat, epsilon);

        theory_density_line = plot(t_hist,theory_density,'LineWidth',2,'DisplayName','theoretical');
        hold on
        comput_density_line = plot(t_hist,comput_density,'LineWidth',2,'DisplayName','computed');

        xlabel('t')
        ylabel('density')

        legend show
        grid on

        drawnow

        %% GIF setup
        save_gif = true;
        gif_filename = ['cahn_hilliard_1D_mu(1)=' num2str(mu_calculate(1, epsilon),'%.2f') '.gif'];
        frame_count = 1;
        next_movie_time = t0;

        %% Time loop

        for n = 1:length(t)-1
            L_operator = -(Laplacian_k.^2);
            u_phys = real(ifft(u_hat));
            u3_hat = fft(u_phys.^3);

            N_hat = Laplacian_k .* dealias_mask .* u3_hat;

            % explicit nonlinear, semi-implicit linear
            u_hat = ((1 + 0.5*dt*L_operator).*u_hat + dt*N_hat) ...
                    ./ (1 - 0.5*dt*L_operator);

            u_hat = dealias_mask .* u_hat;
            u_phys = real(ifft(u_hat));

            t_hist(end+1) = t(n+1);
            l2_hist(end+1) = l2_norm_periodic_1D(u_hat,Lx);

            comput_density(end+1) = domainwall_density_comput(u_phys,Lx);
            theory_density(end+1) = domainwall_density_theory(t(n+1),Laplacian_k,u_hat, epsilon);

            if (t(n+1) >= next_movie_time) || ...
            (n+1 == length(t))

                next_movie_time = next_movie_time + plot_dt;

                u_phys = real(ifft(u_hat));

                subplot(1,3,1)
                set(hsol,'YData',u_phys)
                title(sprintf('t = %.3f',t(n+1)))

                subplot(1,3,2)
                set(hline,'XData',t_hist,'YData',l2_hist)
                axis tight

                subplot(1,3,3)
                blowup_mask = t_hist >= blowup_time;
                
                set(theory_density_line,'XData',t_hist(blowup_mask),'YData',theory_density(blowup_mask))
                set(comput_density_line,'XData',t_hist(blowup_mask),'YData',comput_density(blowup_mask))

                axis tight

                drawnow

                if save_gif

                    frame = getframe(gcf);
                    img = frame2im(frame);

                    [A,map] = rgb2ind(img,256);

                    if frame_count == 1
                        imwrite(A,map,gif_filename,'gif','LoopCount',Inf,'DelayTime',0.05);
                    else
                        imwrite(A,map,gif_filename,'gif','WriteMode','append','DelayTime',0.05);
                    end

                    frame_count = frame_count + 1;

                end
            end
        end

        disp(['GIF saved to: ', gif_filename])

    case 2
        plot_dt = 0.5; 
        plot_every = round(plot_dt / dt); % make multiple of dt

        % time discretization
        blowup_time = epsilon^(-2/3);
        T = blowup_time + 20.0; % used for singular eps cases

        dt = 0.1 * (dx)^2;
        t = t0:dt:T;
        num_time_steps = length(t);

        %% ETDRK4 setup
        L_operator = -Laplacian_k.^2;% - mu_calculate(t(0), epsilon)*Laplacian_k;
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
        fig = figure('Position',[100 100 900 900]);
        set(fig,'Resize','off');

        %% Pre-allocate tracking metrics
        l2_hist = zeros(1, num_time_steps);
        l2_hist(1) = l2_norm_periodic_1D(u_hat, Lx);

        computed_density = zeros(1, num_time_steps);
        computed_density(1) = domainwall_density_comput(u, Lx);

        theory_density = zeros(1, num_time_steps);
        theory_density(1) = domainwall_density_theory(t0,Laplacian_k, u_hat, epsilon);

        %% Subplot 1: Solution
        ax1 = subplot(2,2,[1,2]);
        u_line = plot(ax1, x, u, 'LineWidth', 1.5);

        ylim(ax1, [-1.2 1.2]); 
        xlabel(ax1, 'x'); 
        ylabel(ax1, 'u(x)');
        hTitle1 = title(ax1, sprintf('t = %.3f, \\mu = %.3f, \\epsilon = %.2f', t0, mu_calculate(t0, epsilon), epsilon));
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
        video_filename = sprintf('CH1D_freezeout_epsilon=%.2f_t0=%.2f.avi', epsilon, t0);

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
            u3_nonlinear = u.^3 - mu_calculate(t_prev, epsilon)*u;
            Nu_hat = dealias_mask .* Laplacian_k .* fft(u3_nonlinear);
            
            % Stage 2 (Evaluated at midpoint)
            a_hat = E2.*u_hat + Q.*Nu_hat;
            a = real(ifft(a_hat));
            Na_hat = dealias_mask .* Laplacian_k .* fft(a.^3 - mu_calculate(t_half, epsilon)*a);
            
            % Stage 3 (Evaluated at midpoint)
            b_hat = E2.*u_hat + Q.*Na_hat;
            b = real(ifft(b_hat));
            Nb_hat = dealias_mask .* Laplacian_k .* fft(b.^3 - mu_calculate(t_half, epsilon)*b);
            
            % Stage 4 (Evaluated at t_n)
            c_hat = E2.*a_hat + Q.*(2*Nb_hat - Nu_hat);
            c = real(ifft(c_hat));
            Nc_hat = dealias_mask .* Laplacian_k .* fft(c.^3 - mu_calculate(t_curr, epsilon)*c);
            
            % Final Time Step Combination
            u_hat = E.*u_hat + f1.*Nu_hat + 2*f2.*(Na_hat + Nb_hat) + f3.*Nc_hat;
            u = real(ifft(u_hat));
            
            % Update data arrays
            computed_density(n) = domainwall_density_comput(u,Lx);
            theory_density(n) = domainwall_density_theory(t_curr, Laplacian_k, u_hat, epsilon);
            l2_hist(n) = l2_norm_periodic_1D(u_hat, Lx);

            % Update figures
            if mod(n - 1, plot_every) == 0
                u_line.YData = u;
                hTitle1.String = sprintf('t = %.3f, \\mu = %.3f, \\epsilon = %.2f', t(n), mu_calculate(t(n), epsilon), epsilon);
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
                    frame = getframe(fig,[0 0 900 900]);
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
            frame = getframe(fig,[0 0 900 900]);
            writeVideo(v, frame);

            close(v);
        end

    case 3

        %% storage for final plots
        eps_out = epsilon_list;

        density_comput = zeros(size(epsilon_list));
        density_theory = zeros(size(epsilon_list));
        density_diff = zeros(size(epsilon_list));

        freezeout_time_comput = zeros(size(epsilon_list));
        freezeout_time_theory = zeros(size(epsilon_list));
        freezeout_time_diff = zeros(size(epsilon_list));

        k_theory = zeros(size(epsilon_list));
        k_comput = zeros(size(epsilon_list));
        k_diff = zeros(size(epsilon_list));

        %% loop over epsilon
        for eidx = 1:length(epsilon_list)

            epsilon = epsilon_list(eidx);

            x = (-Nx/2:Nx/2-1)*dx;

            kx = 2*pi*[0:Nx/2-1 -Nx/2:-1]/Lx;
            Laplacian_k = -kx.^2;
            L_operator = -(Laplacian_k.^2);

            blowup_time = (epsilon/2)^(-2/3); %theoretical_freezeout_time(epsilon,kx);
            T = min(blowup_time + 60.0, 150);

            t = t0:dt:T;
            num_time_steps = length(t);

            %% tracking average for each epsilon
            density_comput_avg = zeros(1,num_time_steps);
            density_theory_avg = zeros(1,num_time_steps);

            freezeout_time_comput_avg = 0;
            freezeout_time_theory_avg = 0;

            usq_avg = zeros(Nx,1);

            %% average runs for each epsilon
            avg_num = 10;

            for run_idx=1:avg_num

                %% initial condition
                sigma = 0.01;

                rng(100000*eidx + run_idx,'twister');

                u = sigma * randn(1,Nx);
                u_hat = fft(u);

                mu = @(t) t*epsilon;

                %% tracking once for each epsilon
                density_comput_eps = zeros(1,num_time_steps);
                density_theory_eps = zeros(1,num_time_steps);

                freezeout_time_comput_eps = 0;
                freezeout_time_theory_eps = blowup_time;

                %% ETDRK4
                E  = exp(dt*L_operator);
                E2 = exp(dt*L_operator/2);

                M = 32;
                r = exp(1i*pi*((1:M)-0.5)/M);
                Lvec = L_operator(:);
                LR = dt*Lvec(:,ones(M,1)) + r(ones(numel(L_operator),1),:);

                Q  = dt*real(mean((exp(LR/2)-1)./LR,2)).';
                f1 = dt*real(mean((-4-LR + exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2)).';
                f2 = dt*real(mean((2+LR + exp(LR).*(-2+LR))./LR.^3 ,2)).';
                f3 = dt*real(mean((-4-3*LR-LR.^2 + exp(LR).*(4-LR))./LR.^3 ,2)).';

                clear LR Lvec

                %% time loop
                for n = 2:num_time_steps

                    t_prev = t(n-1);
                    t_half = t_prev + dt/2;
                    t_curr = t(n);

                    u3 = u.^3 - mu_calculate(t_prev, epsilon)*u;
                    Nu_hat = Laplacian_k .* fft(u3);

                    a_hat = E2.*u_hat + Q.*Nu_hat;
                    a = real(ifft(a_hat));
                    Na_hat = Laplacian_k .* fft(a.^3 - mu_calculate(t_half, epsilon)*a);

                    b_hat = E2.*u_hat + Q.*Na_hat;
                    b = real(ifft(b_hat));
                    Nb_hat = Laplacian_k .* fft(b.^3 - mu_calculate(t_half, epsilon)*b);

                    c_hat = E2.*a_hat + Q.*(2*Nb_hat - Nu_hat);
                    c = real(ifft(c_hat));
                    Nc_hat = Laplacian_k .* fft(c.^3 - mu_calculate(t_curr, epsilon)*c);

                    u_hat = E.*u_hat + f1.*Nu_hat + 2*f2.*(Na_hat + Nb_hat) + f3.*Nc_hat;
                    u = real(ifft(u_hat));

                    %% diagnostics
                    density_theory_eps(n) = domainwall_density_theory(t_curr, u_hat, Laplacian_k, epsilon);
                    density_comput_eps(n) = domainwall_density_comput(u, Lx);

                    mask = t > 10;
                    density_masked = density_comput_eps(mask);

                    max_density = max(density_masked);
                    tol = 1e-4*max_density;

                    mask_idx = find(mask);
                    freeze_idx_local = find(density_masked >= max_density - tol,1,'first');

                    if ~isempty(freeze_idx_local)
                        freeze_idx = mask_idx(freeze_idx_local);
                        freezeout_time_comput_eps = t(freeze_idx);
                    end

                    usq_avg = usq_avg + abs(u_hat(:)).^2/avg_num;

                end

                %% running averages
                density_comput_avg = density_comput_avg + density_comput_eps/avg_num;
                density_theory_avg = density_theory_avg + density_theory_eps/avg_num;

                freezeout_time_comput_avg = freezeout_time_comput_avg + freezeout_time_comput_eps/avg_num;
                freezeout_time_theory_avg = freezeout_time_theory_avg + freezeout_time_theory_eps/avg_num;

                k_theory(eidx) = (epsilon/2)^(1/6);

            end

            %% POST PROCESS
            [~,cut] = min(abs(t - blowup_time));

            density_comput(eidx) = density_comput_avg(cut);
            density_theory(eidx) = density_theory_avg(cut);
            density_diff(eidx) = density_theory_avg(cut) - density_comput_avg(cut);

            freezeout_time_comput(eidx) = freezeout_time_comput_avg;
            freezeout_time_theory(eidx) = freezeout_time_theory_avg;
            freezeout_time_diff(eidx) = freezeout_time_theory_avg - freezeout_time_comput_avg;

            kshift = fftshift(kx);
            spec = fftshift(usq_avg);

            [~,idx] = max(spec);

            k_comput(eidx) = abs(kshift(idx));
            k_diff(eidx) = k_theory(eidx) - k_comput(eidx);

            fprintf("epsilon = %.3f done\n", epsilon);

        end

        %% FINAL PLOTS

        figure;

        if plot_diff

            subplot(1,3,1)
            plot(eps_out, density_diff);
            xlabel('\epsilon');
            ylabel('n(\epsilon)');
            title('Domain Wall Density at Freeze-Out');
            grid on;

            subplot(1,3,2)
            loglog(eps_out,freezeout_time_diff,'o-');
            xlabel('\epsilon');
            ylabel('t(\epsilon)');
            title('Freeze-Out Time');
            grid on;

            subplot(1,3,3)
            plot(eps_out,k_diff,'o-');
            xlabel('\epsilon');
            ylabel('k_{max}');
            title('Highest-Growing Wave Modes');
            grid on;

        else

            subplot(1,3,1)
            loglog(eps_out,density_comput,'o-'); hold on;
            loglog(eps_out,density_theory,'s-');
            xlabel('\epsilon');
            ylabel('Domain wall density');
            title('Domain Wall Density');
            legend('comput','theory', 'Location', 'northwest');
            grid on;

            subplot(1,3,2)
            loglog(eps_out,freezeout_time_comput,'o-'); hold on;
            loglog(eps_out,freezeout_time_theory,'k--');
            xlabel('\epsilon');
            ylabel('freeze-out time');
            title('Freeze-Out Time');
            legend('comput','theory');
            grid on;

            subplot(1,3,3)
            plot(eps_out,k_comput,'o-'); hold on;
            plot(eps_out,k_theory,'s-');
            xlabel('\epsilon');
            ylabel('k_{max}');
            title('Highest-Growing Wave Modes');
            legend('comput','theory');
            grid on;

        end
end


%% FUNCTIONS

function mu_val = mu_calculate(t, epsilon)
    mu_val = t*epsilon;
end

function P = power_spec(t,u_hat,Laplacian_k,epsilon)
    ksq = -Laplacian_k;
    P = abs(u_hat).^2 .* exp(t^2*ksq*epsilon - 2*t*ksq.^2);
end

function n = domainwall_density_theory(t,u_hat,Laplacian_k,epsilon)
    P = power_spec(t,u_hat,Laplacian_k,epsilon);
    ksq = -Laplacian_k;

    n = (1/pi)*sqrt(sum(ksq.*P)/sum(P));
end

function n = domainwall_density_comput(u,Lx)
    walls = sum(u .* circshift(u,-1) < 0);
    n = walls/Lx;
end

function t_hat = theoretical_freezeout_time(epsilon,kx)
    sigma = 0.01;
    P0 = sigma^2*ones(size(kx));
    dk = abs(kx(2)-kx(1));

    f = @(t) freezeout_equation(t,epsilon,kx,P0,dk);

    t_guess = 2^(2/3)*epsilon^(-2/3);
    t_hat = fzero(f,t_guess);
end

function val = freezeout_equation(t,epsilon,kx,P0,dk)
    arg = epsilon*(kx.^2)*t.^2 - 2*(kx.^4)*t;
    arg = min(arg,700);

    val = 3*sum(P0 .* exp(arg))*dk - epsilon*t;
end

function C = fit_freezeout_constant(epsilon_list,t_hat_list)
    x = epsilon_list(:).^(-2/3);
    y = t_hat_list(:);

    C = (x'*y)/(x'*x);
end

function val = l2_norm_periodic_1D(u_hat,Lx)
    Nx = length(u_hat);
    val = sqrt(Lx)* sqrt(sum(abs(u_hat).^2))/Nx;
end