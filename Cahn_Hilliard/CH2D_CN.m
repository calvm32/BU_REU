function CH2D_CN()
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
    
    % time discretization
    dt = 0.1 * min(dx,dy)^2;
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
    xticks([-scale*pi 0 scale*pi])
    xticklabels({['-' num2str(scale,'%.0f')] ,'0', [num2str(scale,'%.0f') '\pi']})
    yticks([-scale*pi 0 scale*pi])
    yticklabels({['-' num2str(scale,'%.0f')] ,'0', [num2str(scale,'%.0f') '\pi']})

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
    gif_filename = ['cahn_hilliard_2D_muIC=' num2str(mu(0),'%.2f') '.gif'];
    
    frame_count = 1;
    
    next_movie_time = t0;
    
    %% Time loop
    for n = 1:length(t)-1
        L_operator = -Laplacian_k.^2 - mu(t(n))*Laplacian_k;

        u_phys = real(ifft2(u_hat));
        u3_hat = fft2(u_phys.^3);
        N_hat = Laplacian_k .* dealias_mask .* u3_hat;
    
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