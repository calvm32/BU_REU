function CH1D_gradient_stable()
    clc; clear; close all;

    %% Parameters

    epsilon = 0.05; % used for scaling mu
    blowup_time = epsilon^(-2/3);

    t0 = -2.0;
    T  = blowup_time + 20.0;
    dt = 0.001; 

    scale = 100;
    Lx = scale*pi;

    Nx = 2^12;

    dx = Lx/Nx;

    movie_dt = 0.5;

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

    % random
    sigma = 0.01;
    u0 = sigma*randn(1,Nx);
    u_hat = fft(u0);

    % gaussian
    % u0 = exp(-(X.^2)/sigma);
    % u_hat = fft(u0);
    
    %% discretization
    % spatial discretization
    x = (-Nx/2:Nx/2-1)*dx;
    
    % time discretization
    % dt = 0.1 * (dx)^2;
    t = t0:dt:T;

    %% Fourier wavenumbers
    kx = 2*pi*[0:Nx/2-1 -Nx/2:-1]/Lx;
    Laplacian_k = -(kx.^2);

    %% 2/3 dealiasing mask
    kx_max = max(abs(kx));
    dealias_mask = abs(kx) <= (2/3)*kx_max;

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
    gif_filename = ['cahn_hilliard_1D_mu(1)=' num2str(mu(1),'%.2f') '.gif'];
    frame_count = 1;
    next_movie_time = t0;

    %% Time loop

    for n = 1:length(t)-1
        S = 2.0;
        
        u_phys = real(ifft(u_hat));
        
        u3_hat = fft(u_phys.^3);
        u3_hat = dealias_mask .* u3_hat;
        
        numerator = u_hat + dt*(Laplacian_k .* u3_hat) + dt*S*(Laplacian_k .* u_hat);
        
        denominator = 1 + dt*( ...
                        (Laplacian_k.^2) ...
                        - mu(t(n))*Laplacian_k ...
                        - S*Laplacian_k );
        
        u_hat = numerator ./ denominator;
        
        u_hat = dealias_mask .* u_hat;
        u_phys = real(ifft(u_hat));

        t_hist(end+1) = t(n+1);
        l2_hist(end+1) = l2_norm_periodic_1D(u_hat,Lx);

        comput_density(end+1) = domainwall_density_computed(u_phys,Lx);
        theory_density(end+1) = domainwall_density_theory(t(n+1),Laplacian_k,u_hat);

        if (t(n+1) >= next_movie_time) || ...
           (n+1 == length(t))

            next_movie_time = next_movie_time + movie_dt;

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

end

%% L2 norm
function val = l2_norm_periodic_1D(u_hat,Lx)
    Nx = length(u_hat);
    val = sqrt(Lx)* sqrt(sum(abs(u_hat).^2))/Nx;
end