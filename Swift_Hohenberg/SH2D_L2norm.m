function SH2D_L2norm()
    clc; clear; close all;
    
    %% Parameters
    t0 = 0.0;
    T  = 5.0;
    
    xscale = 2;
    yscale = 10;
    Lx = xscale*pi;
    Ly = yscale*pi;
    
    Nx = 2^7;
    Ny = 2^9;
    
    dx = Lx/Nx;
    dy = Ly/Ny;
    
    movie_dt = 0.05;
    
    mu = 4;
    
    %% Initial condition

    % random
    sigma = 0.1;
    u0 = sigma*randn(Nx,Ny);
    u_hat = fft2(u0);

    % gaussian
    % u0 = exp(-(X.^2 + Y.^2)/sigma);
    % u_hat = fft2(u0);
    
    %% discretization
    % spatial discretization
    x = (-Nx/2:Nx/2-1)*dx;
    y = (-Ny/2:Ny/2-1)*dy;
    
    % time discretization
    dt = 0.1 * min(dx,dy);
    t = t0:dt:T;
    
    %% Fourier wavenumbers
    
    kx = pi * [0:Nx/2-1 -Nx/2:-1] / Lx;
    ky = pi * [0:Ny/2-1 -Ny/2:-1] / Ly;
    
    [KX,KY] = ndgrid(kx,ky);
    
    Laplacian_k = -(KX.^2 + KY.^2);

    %% 2/3 dealiasing mask

    kx_max = max(abs(kx));
    ky_max = max(abs(ky));
    
    dealias_mask = (abs(KX) <= (2/3)*kx_max) & (abs(KY) <= (2/3)*ky_max);
    
    %% Plot setup
    figure('Position',[100 100 1800 600])
    %% first plot: solution
    subplot(1,2,1)
    
    u_phys = real(ifft2(u_hat));
    
    im = imagesc(x,y,u_phys');
    axis equal tight
    set(gca,'YDir','normal')
    
    colormap(seismic_colormap())
    caxis([-1.2 1.2])
    colorbar
    
    title(sprintf('t = %.3f',t(1)))
    xticks([-xscale*pi 0 xscale*pi])
    xticklabels({['-' num2str(xscale,'%.0f')] ,'0', [num2str(xscale,'%.0f') '\pi']})
    yticks([-yscale*pi 0 yscale*pi])
    yticklabels({['-' num2str(yscale,'%.0f')] ,'0', [num2str(yscale,'%.0f') '\pi']})

    %% second plot: L2 norm
    subplot(1,2,2)
    
    t_hist = t(1);
    l2_hist = l2_norm_periodic(u_hat,Lx,Ly);
    
    hline = plot(t_hist,l2_hist,'LineWidth',2);
    
    xlabel('t')
    ylabel('||u||_{L^2}')
    grid on
    
    drawnow
    
    %% GIF setup
    save_gif = true;
    gif_filename = ['swift_hohenberg_2D_mu=' num2str(mu,'%.2f') '.gif'];
    
    frame_count = 1;
    
    next_movie_time = t0;
    
    %% Time loop
    for n = 1:length(t)-1
        L_operator = -(1+Laplacian_k).^2 + mu; % linear form

        u_phys = real(ifft2(u_hat));
        u3_hat = fft2(u_phys.^3);
        N_hat = -dealias_mask .* u3_hat; % nonlinear form
    
        % explicit nonlinear, semi-explicit linear
        u_hat = ((1 + 0.5*dt*L_operator).*u_hat + dt*N_hat) ...
                ./ (1 - 0.5*dt*L_operator);
        u_hat = dealias_mask .* u_hat;

        t_hist(end+1) = t(n+1);
        l2_hist(end+1) = l2_norm_periodic(u_hat,Lx,Ly);

        if (t(n+1) >= next_movie_time) || (n+1 == length(t))
    
            next_movie_time = next_movie_time + movie_dt;
            u_phys = real(ifft2(u_hat));
    
            set(im,'CData',u_phys')
            subplot(1,2,1)

            subplot(1,2,2)
            set(hline,'XData',t_hist,'YData',l2_hist)
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