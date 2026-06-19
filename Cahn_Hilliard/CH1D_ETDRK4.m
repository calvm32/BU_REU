function CH1D_ETDRK4()
    clc; clear; close all;

    %% Parameters

    epsilon = 0.05; % used for scaling mu
    blowup_time = epsilon^(-2/3);

    t0 = -2.0;
    T  = blowup_time + 20.0;

    scale = 100;
    Lx = scale*pi;

    Nx = 2^12;

    dx = Lx/Nx;

    movie_dt = 0.05;

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

    function density = domainwall_density_computed(u,Lx)
        walls = sum(u .* circshift(u,-1) < 0);
        density = walls / Lx;
    end

    %% Initial condition

    sigma = 0.01;
    u0 = sigma*randn(1,Nx);
    u_hat = fft(u0);

    %% discretization

    x = (-Nx/2:Nx/2-1)*dx;

    dt = 0.1*(dx)^2;
    t = t0:dt:T;

    %% Fourier wavenumbers

    kx = 2*pi*[0:Nx/2-1 -Nx/2:-1]/Lx;
    Laplacian_k = -(kx.^2);

    %% 2/3 dealiasing mask

    kx_max = max(abs(kx));
    dealias_mask = abs(kx) <= (2/3)*kx_max;

    %% ETDRK4 coefficients
    %
    % Use the initial linear operator. Since mu(t) varies slowly,
    % we recompute coefficients every step below.

    M = 32;
    r = exp(1i*pi*((1:M)-0.5)/M);

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

    comput_density = domainwall_density_computed(u_phys,Lx);
    theory_density = domainwall_density_theory(t0,Laplacian_k,u_hat);

    theory_density_line = plot(t_hist,theory_density,...
        'LineWidth',2,'DisplayName','theoretical');

    hold on

    comput_density_line = plot(t_hist,comput_density,...
        'LineWidth',2,'DisplayName','computed');

    xlabel('t')
    ylabel('density')

    legend show
    grid on

    drawnow

    %% GIF setup

    save_gif = true;
    gif_filename = ['cahn_hilliard_1D_mu(1)=' num2str(mu(1),'%.2f') '.gif'];

    frame_count = 1;
    next_movie_time = t0;

    %% Time loop

    for n = 1:length(t)-1

        %% Linear operator

        L_operator = -Laplacian_k.^2 ...
                     - mu(t(n))*Laplacian_k;

        %% ETDRK4 coefficients

        E  = exp(dt*L_operator);
        E2 = exp(dt*L_operator/2);

        LR = dt*L_operator(:) + r;

        Q  = dt * real(mean( ...
            (exp(LR/2)-1)./LR ,2));

        f1 = dt * real(mean( ...
            (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2));

        f2 = dt * real(mean( ...
            (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));

        f3 = dt * real(mean( ...
            (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));

        Q  = Q.';
        f1 = f1.';
        f2 = f2.';
        f3 = f3.';

        %% Nonlinear function N(u)=Delta(u^3)

        u_phys = real(ifft(u_hat));

        Nu = Laplacian_k .* ...
             fft(u_phys.^3);

        Nu = dealias_mask .* Nu;

        %% Stage a

        a = E2 .* u_hat + Q .* Nu;

        ua = real(ifft(a));

        Na = Laplacian_k .* ...
             fft(ua.^3);

        Na = dealias_mask .* Na;

        %% Stage b

        b = E2 .* u_hat + Q .* Na;

        ub = real(ifft(b));

        Nb = Laplacian_k .* ...
             fft(ub.^3);

        Nb = dealias_mask .* Nb;

        %% Stage c

        c = E2 .* a + Q .* (2*Nb - Nu);

        uc = real(ifft(c));

        Nc = Laplacian_k .* ...
             fft(uc.^3);

        Nc = dealias_mask .* Nc;

        %% ETDRK4 update

        u_hat = E .* u_hat ...
              + f1 .* Nu ...
              + 2*f2 .* (Na + Nb) ...
              + f3 .* Nc;

        u_hat = dealias_mask .* u_hat;

        %% Diagnostics

        u_phys = real(ifft(u_hat));

        t_hist(end+1) = t(n+1);

        l2_hist(end+1) = ...
            l2_norm_periodic_1D(u_hat,Lx);

        comput_density(end+1) = ...
            domainwall_density_computed(u_phys,Lx);

        theory_density(end+1) = ...
            domainwall_density_theory( ...
                t(n+1),Laplacian_k,u_hat);

        %% Plot/movie

        if (t(n+1) >= next_movie_time) || ...
           (n+1 == length(t))

            next_movie_time = ...
                next_movie_time + movie_dt;

            subplot(1,3,1)

            set(hsol,'YData',u_phys)

            title(sprintf('t = %.3f',t(n+1)))

            subplot(1,3,2)

            set(hline,...
                'XData',t_hist,...
                'YData',l2_hist)

            axis tight

            subplot(1,3,3)

            blowup_mask = ...
                t_hist >= blowup_time;

            set(theory_density_line,...
                'XData',t_hist(blowup_mask),...
                'YData',theory_density(blowup_mask))

            set(comput_density_line,...
                'XData',t_hist(blowup_mask),...
                'YData',comput_density(blowup_mask))

            axis tight

            drawnow

            if save_gif

                frame = getframe(gcf);
                img = frame2im(frame);

                [A,map] = rgb2ind(img,256);

                if frame_count == 1
                    imwrite(A,map,gif_filename,...
                        'gif',...
                        'LoopCount',Inf,...
                        'DelayTime',0.05);
                else
                    imwrite(A,map,gif_filename,...
                        'gif',...
                        'WriteMode','append',...
                        'DelayTime',0.05);
                end

                frame_count = frame_count + 1;

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