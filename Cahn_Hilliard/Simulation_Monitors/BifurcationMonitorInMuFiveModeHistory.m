classdef BifurcationMonitorInMuFiveModeHistory < BifurcationMonitorHeadless
    %BIFURCATIONMONITORINMU A monitor for bifurcation and mode surpass analysis.
    % Displays the solution, L2 norm, dominant Fourier mode history, and 
    % amplitude history of the first five modes.

    properties (SetAccess = private)
        x (1, :) double
        k_j (1, :) double
        plot_every (1, 1) double
        save_video (1, 1) logical
        video_filename string
        video_framerate (1, 1) double
        local_mass (1, 1) double

        % Plotting handles
        fig
        hLine1
        l2_line
        amp_plots
        crit_lines
        hLine3  
        ax1
        ax2
        ax3
        ax4
        hTitle1

        % Video
        v
    end

    methods
        function obj = BifurcationMonitorInMuFiveModeHistory(mu, kx, Lx, Nx, x, k_j, options)
            arguments
                mu
                kx (1, :) double
                Lx (1, 1) double
                Nx (1, 1) double
                x (1, :) double
                k_j (1, :) double
                options.plot_every (1, 1) double = 1
                options.save_video (1, 1) logical = false
                options.video_filename string = "bifurcation_movie.avi"
                options.video_framerate (1, 1) double = 30
                options.subtract_mass logical = false
                options.local_mass double = 0
            end

            obj = obj@BifurcationMonitorHeadless(mu, kx, Lx, Nx, 'subtract_mass', options.subtract_mass);
            obj.x = x;
            obj.k_j = k_j;
            obj.plot_every = options.plot_every;
            obj.save_video = options.save_video;
            obj.video_filename = options.video_filename;
            obj.video_framerate = options.video_framerate;
            obj.local_mass = options.local_mass;
        end

        function initialize(obj, u0_hat, t_grid)    
            
            if obj.subtract_mass
                u0_hat(1) = 0;
            end
            
            % Plot setup
            obj.fig = figure('Position',[100 100 1200 700], 'Resize', 'off');

            % Subplot 1: Solution (Spans the entire top row)
            obj.ax1 = subplot(2,3,[1 2 3]);
            u0 = real(ifft(u0_hat));
            obj.hLine1 = plot(obj.ax1, obj.x, u0, 'LineWidth', 1.5);
            
            % Set limits depending on the value of mu at the last time.
            %ylim(obj.ax1, 1.2 * [-sqrt(obj.mu(t_grid(end))) sqrt(obj.mu(t_grid(end)))]);
            mu_val = max(0, obj.mu(t_grid(end)));
            try
                ylim(obj.ax1, 1.2 * [-sqrt(mu_val) sqrt(mu_val)]);
            catch
                ylim(obj.ax1, 1.2 * [-1 1]);
            end
            
            xlabel(obj.ax1, 'x'); 
            ylabel(obj.ax1, 'u - m');
            obj.hTitle1 = title(obj.ax1, sprintf('t = %.3f', t_grid(1)));
            grid(obj.ax1, 'on');

            % Subplot 2: L2 norm
            obj.ax2 = subplot(2,3,4);
            obj.l2_line = semilogy(obj.ax2, 0, NaN, 'LineWidth', 2);
            xlabel(obj.ax2, '\mu'); 
            ylabel(obj.ax2, '||u(\mu, \cdot) - m||_{L^2}');
            title(obj.ax2, 'L2 Norm of u');
            grid(obj.ax2, 'on');

            % Subplot 3: Dominant mode
            obj.ax3 = subplot(2,3,5);
            obj.hLine3 = stairs(obj.ax3, 0, 0, 'LineWidth', 2);
            labels = cellstr("k_" + (0:numel(obj.k_j)-1));
            yline(obj.ax3, obj.k_j, '--r', labels, 'LineWidth', 2);    
            xlabel(obj.ax3, '\mu'); 
            ylabel(obj.ax3, 'Wavenumber (k)');
            xlim(obj.ax3, obj.mu(t_grid([1, end])));
            ylim(obj.ax3, [0, obj.k_j(end)]); 
            grid(obj.ax3, 'on');

            % Subplot 4: Fourier amplitude histories
            obj.ax4 = subplot(2,3,6);
            hold(obj.ax4, 'on'); 
            obj.amp_plots = gobjects(1, 5); 
            
            for k = 1:5
                obj.amp_plots(k) = semilogy(obj.ax4, NaN, NaN, 'LineWidth', 2);
            end

            obj.crit_lines = gobjects(1,20);
            for k = 1:20
                mu_k_c = 3*obj.local_mass^2 + (pi*k/obj.Lx)^2;
                obj.crit_lines(k) = xline(obj.ax4, mu_k_c, ':', sprintf('\\mu_{%d}^c', k), ...
                    'LineWidth', 1.5, 'Color', [0.5 0.5 0.5], ...
                    'LabelVerticalAlignment','middle', ...
                    'Visible','off', 'HandleVisibility','off');
                    
            end
            
            labels = arrayfun(@(k) sprintf('u_%d', k), 1:5, 'UniformOutput', false);
            legend(obj.ax4, obj.amp_plots, labels, 'Location', 'southwest', 'Interpreter', 'tex');
            xlabel(obj.ax4, '\mu'); 
            ylabel(obj.ax4, '$|u_k - m|$', 'Interpreter', 'latex');
            title(obj.ax4, 'Amplitude over \mu');
            grid(obj.ax4, 'on');
            set(obj.ax4, 'YScale', 'log');

            % Video setup
            if obj.save_video
                obj.v = VideoWriter(obj.video_filename, 'Motion JPEG AVI');
                obj.v.FrameRate = obj.video_framerate;
                obj.v.Quality = 100;
                open(obj.v);
            end
            
            initialize@BifurcationMonitorHeadless(obj, u0_hat, t_grid);
        end

        function update(obj, u_hat, t)

            if obj.subtract_mass
                u_hat(1) = 0;
            end
            
            update@BifurcationMonitorHeadless(obj, u_hat, t);

            u = real(ifft(u_hat));

            % Update plots
            if mod(obj.step_idx - 1, obj.plot_every) == 0
                obj.hLine1.YData = u;
                obj.hTitle1.String = sprintf('mu = %.4f, t = %.1f', obj.mu(t), t);

                obj.l2_line.XData = obj.mu_grid(1:obj.step_idx);
                obj.l2_line.YData = obj.l2_history(1:obj.step_idx);
                axis(obj.ax2, 'tight');

                valid_modes = obj.dominate_mode(1:obj.dom_mode_ind, :);
                set(obj.hLine3, 'XData', obj.mu([valid_modes(:, 1); t]), 'YData', [valid_modes(:, 2); valid_modes(end, 2)]);

                for k = 1:5
                    set(obj.amp_plots(k), ...
                        'XData', obj.mu_grid(1:obj.step_idx), ...
                        'YData', obj.amp_history(k, 1:obj.step_idx));
                end
                
                for k = 1:numel(obj.crit_lines)
                    mu_k_c = 3*obj.local_mass^2 + (pi*k/obj.Lx)^2;

                    if obj.mu(t) >= mu_k_c
                        obj.crit_lines(k).Visible = 'on';
                    else
                        obj.crit_lines(k).Visible = 'off';
                    end
                end

                axis(obj.ax4, 'tight');

                drawnow;

                if obj.save_video && ~isempty(obj.v) && isgraphics(obj.fig)
                    frame = getframe(obj.fig);
                    writeVideo(obj.v, frame);
                end
            end
        end

        function finalize(obj)
            if obj.save_video && ~isempty(obj.v)
                close(obj.v);
            end
            if isgraphics(obj.fig)
                close(obj.fig);
            end
        end
    end
end
