function CH2D_domain_wall_density()
    clc; clear; close all;
    
    %% Parameters
       
    tau = 4; % used for scaling mu
    
    t0 = 0.0;
    T  = 5.0;
    
    scale=20;
    Lx = scale*pi;
    Ly = scale*pi;
    
    Nx = 2^8;
    Ny = 2^8;
    
    dx = Lx/Nx;
    dy = Ly/Ny;
    
    movie_dt = 0.05;

    %% functions
    function mu = mu(t)
        mu = t/tau;
    end

    function power_spec = power_spec(t, Laplacian_k, u_hat)
        ksq = -Laplacian_k;
        initial = integral(abs(u_hat).^2, 0, T);
        power_spec = initial .* exp( t^2*ksq/tau - 2*t*ksq.^2 );
    end

    function n = domainwall_density(t, Laplacian_k, u_hat)
        P = power_spec(t, Laplacian_k, u_hat);
        ksq = -Laplacian_k;
        num = sum(ksq.*P,'all');
        den = sum(P,'all');

        n = (pi/2)*sqrt(num/den);
    end
    
    
    %% Initial condition

    % random
    sigma = 0.01;
    u0 = sigma*randn(Nx,Ny);
    u_hat = fft2(u0);

    % gaussian
    % u0 = exp(-(X.^2 + Y.^2)/sigma);
    % u_hat = fft2(u0);
    
    %% discretization
    % spatial discretization
    x = (-Nx/2:Nx/2-1)*dx;
    y = (-Ny/2:Ny/2-1)*dy;
    
    [X,Y] = ndgrid(x,y);
    
    % time discretization
    dt = 0.1 * min(dx,dy)^2;
    t = t0:dt:T;
    
    %% Fourier wavenumbers
    
    kx = 2*pi * [0:Nx/2-1 -Nx/2:-1] / Lx;
    ky = 2*pi * [0:Ny/2-1 -Ny/2:-1] / Ly;
    
    [KX,KY] = ndgrid(kx,ky);
    
    Laplacian_k = -(KX.^2 + KY.^2);

    %% 2/3 dealiasing mask

    kx_max = max(abs(kx));
    ky_max = max(abs(ky));
    
    dealias_mask = (abs(KX) <= (2/3)*kx_max) & (abs(KY) <= (2/3)*ky_max);
    
    %% Plot setup
    figure('Position',[100 100 1800 600])
    %% first plot: solution
    subplot(1,3,1)
    
    u_phys = real(ifft2(u_hat));
    
    im = imagesc(x,y,u_phys');
    axis equal tight
    set(gca,'YDir','normal')
    
    colormap(seismic_colormap())
    caxis([-1.2 1.2])
    colorbar
    
    title(sprintf('t = %.3f',t(1)))
    xticks([-scale*pi 0 scale*pi])
    xticklabels({['-' num2str(scale,'%.0f')] ,'0', [num2str(scale,'%.0f') '\pi']})
    yticks([-scale*pi 0 scale*pi])
    yticklabels({['-' num2str(scale,'%.0f')] ,'0', [num2str(scale,'%.0f') '\pi']})

    %% second plot: L2 norm
    subplot(1,3,2)
    
    t_hist = t(1);
    l2_hist = l2_norm_periodic(u_hat,Lx,Ly);
    
    hline = plot(t_hist,l2_hist,'LineWidth',2);
    
    xlabel('t')
    ylabel('||u||_{L^2}')
    grid on
    
    %% third plot: domain wall density
    subplot(1,3,3)
    
    BW = u_phys > 0;
    perim = bwperim(BW,8);
    Lwall = sum(perim(:))*dx;
    comput_density = Lwall/(Lx*Ly);

    theory_density = domainwall_density(t0, Laplacian_k, u_hat);
    
    theory_density_line = plot(t_hist, theory_density, 'LineWidth',2, 'DisplayName', 'theoretical');
    hold on;
    comput_density_line = plot(t_hist, comput_density, 'LineWidth',2, 'DisplayName', '# minima');
    
    xlabel('t')
    ylabel('density')
    legend show

    grid on
    
    drawnow
    
    %% GIF setup
    save_gif = true;
    gif_filename = ['cahn_hilliard_2D_muIC=' num2str(mu(0),'%.2f') '.gif'];
    
    frame_count = 1;
    
    next_movie_time = t0;
    
    %% Time loop
    for n = 1:length(t)-1
        lambda = -Laplacian_k.^2 - mu(t(n))*Laplacian_k;

        u_phys = real(ifft2(u_hat));
        u3_hat = fft2(u_phys.^3);
        N_hat = Laplacian_k .* dealias_mask .* u3_hat;
    
        % explicit nonlinear, semi-explicit linear
        u_hat = ((1 + 0.5*dt*lambda).*u_hat + dt*N_hat) ...
                ./ (1 - 0.5*dt*lambda);
        u_hat = dealias_mask .* u_hat;

        t_hist(end+1) = t(n+1);
        l2_hist(end+1) = l2_norm_periodic(u_hat,Lx,Ly);
        
        % get domain wall density
        BW = u_phys > 0;
        perim = bwperim(BW,8);
        Lwall = sum(perim(:))*dx;
        comput_density(end+1) = Lwall/(Lx*Ly);
    
        theory_density(end+1) = domainwall_density(t(n+1), Laplacian_k, u_hat);

        if (t(n+1) >= next_movie_time) || (n+1 == length(t))
    
            next_movie_time = next_movie_time + movie_dt;
            u_phys = real(ifft2(u_hat));
    
            set(im,'CData',u_phys')
            subplot(1,3,1)

            subplot(1,3,2)
            set(hline,'XData',t_hist,'YData',l2_hist)
            axis tight
            
            subplot(1,3,3)
            set(theory_density_line,'XData',t_hist,'YData',theory_density)
            set(comput_density_line,'XData',t_hist,'YData',comput_density)
            
            xlabel('t')
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
function val = l2_norm_periodic(u_hat,Lx,Ly)
    [Nx,Ny] = size(u_hat);
    val = sqrt(Lx*Ly) * sqrt(sum(abs(u_hat(:)).^2)) /(Nx*Ny);
end
    
    
%% Seismic colormap    
function cmap = seismic_colormap()
    
    n = 256;
    
    r = [(0:n/2-1)/(n/2), ones(1,n/2)];
    g = [(0:n/2-1)/(n/2), (n/2-1:-1:0)/(n/2)];
    b = [ones(1,n/2), (n/2-1:-1:0)/(n/2)];
    
    cmap = [r(:),g(:),b(:)];

end