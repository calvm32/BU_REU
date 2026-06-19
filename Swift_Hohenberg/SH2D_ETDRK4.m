function SH2D_ETDRK4()
    clc; clear; close all;

    %% Parameters
    t0 = 0.0;
    T  = 800.0;

    xscale = 1;
    yscale = 10;

    Lx = xscale*pi;
    Ly = yscale*pi;

    Nx = 2^7;
    Ny = 2^9;

    dx = 2*Lx/Nx;
    dy = 2*Ly/Ny;

    movie_dt = 1.0;

    mu = 0.25;
    k  = 0.95;

    %% Initial condition

    % spatial grid (needed for patterned base state)
    x = (-Nx/2:Nx/2-1)*dx;
    y = (-Ny/2:Ny/2-1)*dy;

    [X,Y] = ndgrid(x,y);

    kappa = k^2 - 1;

    up = sqrt((4/3)*(mu-kappa^2))*cos(k*X);

    sigma = 0.1;
    u0 = up + sigma*(rand(Nx,Ny)-0.5);

    u_hat = fft2(u0);

    %% discretization

    % time discretization
    dt = 0.1;
    t = t0:dt:T;

    %% Fourier wavenumbers

    kx = pi*[0:Nx/2-1 -Nx/2:-1]/Lx;
    ky = pi*[0:Ny/2-1 -Ny/2:-1]/Ly;

    [KX,KY] = ndgrid(kx,ky);

    % anisotropic SH operator
    k2 = k^2*KX.^2 + KY.^2;

    % Linear operator:
    %
    % u_t = -(1-2k2+k2^2)u + mu*u - u^3
    %
    L_operator = -(1 - 2*k2 + k2.^2) + mu;

    %% 2/3 dealiasing mask

    kx_max = max(abs(kx));
    ky_max = max(abs(ky));

    dealias_mask = ...
        (abs(KX) <= (2/3)*kx_max) & ...
        (abs(KY) <= (2/3)*ky_max);

    %% ETDRK4 setup

    E  = exp(dt*L_operator);
    E2 = exp(dt*L_operator/2);

    M = 16; % no. of points for complex means
    r = exp(1i*pi*((1:M)-0.5)/M); % roots of unity

    Lvec = L_operator(:);
    LR = dt*Lvec(:,ones(M,1)) + r(ones(numel(L_operator),1),:);

    Q  = dt*real(mean((exp(LR/2)-1)./LR,2));

    f1 = dt*real(mean((-4-LR + exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2));
    f2 = dt*real(mean((2+LR + exp(LR).*(-2+LR))./LR.^3 ,2));
    f3 = dt*real(mean((-4-3*LR-LR.^2 + exp(LR).*(4-LR))./LR.^3 ,2));

    Q  = reshape(Q ,Nx,Ny);
    f1 = reshape(f1,Nx,Ny);
    f2 = reshape(f2,Nx,Ny);
    f3 = reshape(f3,Nx,Ny);

    clear LR Lvec

    %% Plot setup

    figure('Position',[100 100 1800 600])

    %% first plot: solution
    subplot(1,2,1)

    u_phys = real(ifft2(u_hat));
    im = imagesc(x,y,u_phys');

    axis equal tight
    set(gca,'YDir','normal')

    colormap(seismic_colormap())
    colorbar
    caxis([-1 1])

    title(sprintf('t = %.3f',t0))

    xticks([-xscale*pi 0 xscale*pi])
    xticklabels({['-' num2str(xscale)],'0',[num2str(xscale) '\pi']})

    yticks([-yscale*pi 0 yscale*pi])
    yticklabels({['-' num2str(yscale)],'0',[num2str(yscale) '\pi']})

    xlabel('x')
    ylabel('y')

    %% second plot: cross section
    subplot(1,2,2)
    xh = round(Nx/2);
    hline = plot(y,u_phys(xh,:),'LineWidth',2);

    xlabel('y')
    ylabel('u(x=0,y)')
    grid on

    ylim([-1.2 1.2])
    xlim([-Ly Ly])

    drawnow

    %% Data storage

    ycross = y;
    ucross = u_phys(xh,:);

    %% GIF setup
    save_gif = true;
    gif_filename = sprintf('SH_ETDRK4_mu=%.2f_k=%.2f.gif',mu,k);
    frame_count = 1;
    next_movie_time = t0;

    %% Time loop

    for n = 1:length(t)-1

        % ETDRK4

        u = real(ifft2(u_hat));

        Nu = -u.^3;
        Nu_hat = dealias_mask .* fft2(Nu);

        a_hat = E2.*u_hat + Q.*Nu_hat;

        a = real(ifft2(a_hat));
        Na_hat = dealias_mask .* fft2(-a.^3);

        b_hat = E2.*u_hat + Q.*Na_hat;

        b = real(ifft2(b_hat));
        Nb_hat = dealias_mask .* fft2(-b.^3);

        c_hat = E2.*a_hat + Q.*(2*Nb_hat - Nu_hat);

        c = real(ifft2(c_hat));
        Nc_hat = dealias_mask .* fft2(-c.^3);

        u_hat = E.*u_hat + f1.*Nu_hat + 2*f2.*(Na_hat + Nb_hat) + f3.*Nc_hat;
        u_hat = dealias_mask .* u_hat;

        % -----------------

        u_phys = real(ifft2(u_hat));

        ucross(end+1,:) = u_phys(xh,:);

        if (t(n+1) >= next_movie_time) || (n+1 == length(t))

            next_movie_time = next_movie_time + movie_dt;

            subplot(1,2,1)

            set(im,'CData',u_phys')

            title(sprintf('t = %.3f',t(n+1)))

            subplot(1,2,2)

            set(hline,...
                'XData',y,...
                'YData',u_phys(xh,:))

            title(sprintf('Cross section at x = %.3f',x(xh)))

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

    %% Save data

    save(sprintf('SH_mu=%.2f_k=%.2f.mat',mu,k),...
        'mu','k','Nx','Ny','Lx','Ly',...
        'dt','T','ycross','ucross')

    disp(['GIF saved to: ', gif_filename])

end


%% Seismic colormap
function cmap = seismic_colormap()

    n = 256;

    r = [(0:n/2-1)/(n/2), ones(1,n/2)];
    g = [(0:n/2-1)/(n/2), (n/2-1:-1:0)/(n/2)];
    b = [ones(1,n/2), (n/2-1:-1:0)/(n/2)];

    cmap = [r(:),g(:),b(:)];

end