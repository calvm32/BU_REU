clear; clc;

graph = 4; % valid: 1, 2, 3, 4

mu0 = -4;
m = 1;

function mu = mu_bif(mu0, k, L, m) 
    try
        mu = -mu0 + 2*(2*pi*k./L).^2 + 6*m^2;
    catch
        mu = -mu0 + 2*(2*pi*k/L).^2 + 6*m^2;
    end
end

mu = @(epsilon, t, mu0) epsilon*t + mu0;

switch graph
    case 1 
        % plots the exact point of bifurcation for varying L, k
        epsilon = 0.05;
        t = 2;
        
        [L,K] = meshgrid(linspace(2,100,300),0:20);
        diff = mu(epsilon, t, mu0) - mu_bif(mu0, K, L, m);
        
        figure; hold on
        surf(L,K,diff)
        shading interp
        view(45,30)

        xlabel('L')
        ylabel('k')
        zlabel('\mu-\mu_{bif}')

    case 2 
        % plots simply the threshold that mu would need to cross 
        % in order to bifurcate for varying L, k
        [L,K] = meshgrid(linspace(2,100,300),0:20);
        
        figure; hold on
        for k = 0:10
            plot(L,mu_bif(mu0, k, L, m),'LineWidth',2)
        end
        
        xlabel('L')
        ylabel('\mu')
        title('Instability boundaries')

    case 3 
        % uses the bifurcation regime mu - mu_bif to find the largest unstable mode k 
        % for varying mu just taken in a linear space
        Lvec  = linspace(2,100,300);
        muvec = linspace(0,20,300);
        [L,Mu] = meshgrid(Lvec,muvec);
        
        kmax = 30;
        k_unstable = NaN(size(L));
        
        % calculate unstable modes for each length L
        for i = 1:numel(L)
        
            L0  = L(i);
            mu_now = Mu(i);
        
            unstable = [];
            for k = 0:kmax
        
                diff = mu_now - mu_bif(mu0, k, L0, m);
        
                if diff > 0
                    unstable(end+1) = k;
                end
            end
        
            if ~isempty(unstable)
                k_unstable(i) = max(unstable);
            end
        end
        
        figure
        imagesc(Lvec,muvec,k_unstable)
        set(gca,'YDir','normal')
        xlabel('L')
        ylabel('\mu')
        title('Largest unstable mode')
        colorbar

    case 4 
        % uses the bifurcation regime mu - mu_bif to find the largest unstable mode k 
        % for varying epsilon, as time changes
        Lvec  = linspace(2,100,200);
        epsvec = linspace(0,5,200);
        [L,Eps] = meshgrid(Lvec,epsvec);
        
        kmax = 30;
        k_unstable = NaN(size(L));
        k_fast = NaN(size(L));

        % time discretization
        t0 = -2.0;
        T = 150;
        dt = 0.1;
        t = t0:dt:T;
        num_time_steps = length(t);
                
        % Video setup
        save_video = true;
        video_filename = sprintf('CH1D_parameter_bif');
        
        if save_video
            v = VideoWriter(video_filename, 'Motion JPEG AVI');
            v.FrameRate = 15;  % Target frames per second
            v.Quality = 100;
            open(v);
        end

        % figure setup
        fig = figure('Position',[100 100 600 600]);
        set(fig,'Resize','off');
        
        unstable_modes = imagesc(Lvec,epsvec,k_unstable);
        hold on
        % fastest_modes = contour(L,Eps,k_fast,0:1:kmax,'k','LineWidth',1.5);
        set(gca,'YDir','normal')
        xlabel('L')
        ylabel('\epsilon')
        mu_min = Eps(1)*t0 + mu0;
        mu_max = Eps(end)*t0 + mu0;

        hTitle = title(sprintf('Largest unstable mode,   t = %.3f,   %.3f < \\mu <   %.3f', t0, mu_min, mu_max));
        colorbar
                
        % calculate unstable modes for each length L
        for n = 1:num_time_steps
            t_curr = t(n);
            mu_min = Eps(1)*t_curr + mu0;
            mu_max = Eps(end)*t_curr + mu0;

            hTitle = title(sprintf('Largest unstable mode,   t = %.3f,   %.3f < \\mu <   %.3f', t_curr, mu_min, mu_max));

            for i = 1:numel(L)
            
                L0  = L(i);
                eps = Eps(i);
                mu_now = eps*t_curr + mu0;

                % all unstable modes (later take max)
                unstable = [];
                for k = 0:kmax
            
                    diff = mu_now - mu_bif(mu0, k, L0, m);
            
                    if diff > 0
                        unstable(end+1) = k;
                    end
                end
            
                if ~isempty(unstable)
                    k_unstable(i) = max(unstable);
                end

                % fastest-growing mode
                if mu_now > 3*m^2
                    k_fast(i) = L0/(2*pi) * sqrt((mu_now - 3*m^2)/2);
                end

            end

            % update plotting
            unstable_modes.CData = k_unstable;

            % k_fast_int = round(k_fast);
            % delete(fastest_modes)
            % fastest_modes = contour(L,Eps,k_fast_int,0:1:kmax,'k','LineWidth',1.5);
            drawnow limitrate 
            
            if save_video
                % Capture the current high-res figure frame and write to file
                frame = getframe(fig,[0 0 600 600]);
                writeVideo(v, frame);
            end

        end

    otherwise
        error('Invalid graph selection');
end

grid on