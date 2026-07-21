classdef BifurcationMonitorInMu < BifurcationMonitorHeadless
    %BIFURCATIONMONITORINMU A monitor for bifurcation and mode surpass analysis.
    % Displays the solution, L2 norm, dominant Fourier mode history, and the Fourier spectrum.

    properties (SetAccess = private)
        x (1, :) double
        k_j (1, :) double
        plot_every (1, 1) double
        save_video (1, 1) logical
        video_filename string
        video_framerate (1, 1) double

        % Plotting handles
        fig
        hLine1
        l2_line
        hLine3
        hFreq
        ax1
        ax2
        ax3
        ax4
        hTitle1

        % Video
        v
    end

    methods
        function obj = BifurcationMonitorInMu(mu, kx, Lx, Nx, x, k_j, options)
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
            end

            obj = obj@BifurcationMonitorHeadless(mu, kx, Lx, Nx);
            obj.x = x;
            obj.k_j = k_j;
            obj.plot_every = options.plot_every;
            obj.save_video = options.save_video;
            obj.video_filename = options.video_filename;
            obj.video_framerate = options.video_framerate;
        end

        function initialize(obj, u0_hat, t_grid)
            % Plot setup
            obj.fig = figure('Position',[100 100 1200 700], 'Resize', 'off');

            % Subplot 1: Solution (Spans the entire top row)
            obj.ax1 = subplot(2,3,[1 2 3]);
            u0 = real(ifft(u0_hat));
            obj.hLine1 = plot(obj.ax1, obj.x, u0, 'LineWidth', 1.5);
            
            % Set limits depending on the value of mu at the last time.
            ylim(obj.ax1, 1.2 * [-sqrt(obj.mu(t_grid(end))) sqrt(obj.mu(t_grid(end)))]);
            
            xlabel(obj.ax1, 'x'); 
            ylabel(obj.ax1, 'u');
            obj.hTitle1 = title(obj.ax1, sprintf('t = %.3f', t_grid(1)));
            grid(obj.ax1, 'on');

            % Subplot 2: L2 norm
            obj.ax2 = subplot(2,3,4);
            obj.l2_line = semilogy(obj.ax2, 0, NaN, 'LineWidth', 2);
            xlabel(obj.ax2, '\mu'); 
            ylabel(obj.ax2, '||u(\mu, \cdot)||_{L^2}');
            title(obj.ax2, 'L2 Norm of u');
            grid(obj.ax2, 'on');

            % Subplot 3: Dominant mode
            obj.ax3 = subplot(2,3,5);
            obj.hLine3 = stairs(obj.ax3, 0, 0, 'LineWidth', 2);
            labels = cellstr("k_{" + (0:numel(obj.k_j)-1)+"}");
            yline(obj.ax3, obj.k_j, '--r', labels, 'LineWidth', 2);    
            xlabel(obj.ax3, '\mu'); 
            ylabel(obj.ax3, 'Wavenumber (k)');
            xlim(obj.ax3, obj.mu(t_grid([1, end])));
            ylim(obj.ax3, [0, obj.k_j(end)]); 
            grid(obj.ax3, 'on');

            % Subplot 4: Fourier Spectrum (semilogy: handles zeros cleanly)
            obj.ax4 = subplot(2,3,6);
            obj.hFreq = semilogy(obj.ax4, ifftshift(obj.kx), ifftshift(abs(u0_hat)), 'LineWidth', 2);
            xlabel(obj.ax4, 'k'); 
            ylabel(obj.ax4, '$|\widehat{u(\mu)}|$', 'Interpreter', 'latex');
            xlim(obj.ax4, [-obj.kx(64), obj.kx(64)]);
            ylim(obj.ax4, [1e-25, 1e5]);
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

                obj.hFreq.YData = ifftshift(abs(u_hat) / obj.Nx);

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