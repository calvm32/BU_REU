classdef BifurcationMonitorInMu < BifurcationMonitorHeadless
    %BIFURCATIONMONITORINMU A monitor for bifurcation and mode surpass analysis.
    % Displays the solution, L2 norm, dominant Fourier mode history, and the Fourier spectrum.

    properties (SetAccess = private)
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
        function initialize(obj, u0_hat, t_grid)
            % Plot setup
            obj.fig = figure('Position',[100 100 1200 700], 'Resize', 'off');

            % Subplot 1: Solution (Spans the entire top row)
            obj.ax1 = subplot(2,3,[1 2 3]);
            u0 = real(ifft(u0_hat));
            obj.hLine1 = plot(obj.ax1, obj.params.x, u0, 'LineWidth', 1.5);
            
            % Set limits depending on the value of mu at the last time.
            ylim(obj.ax1, 1.2 * [-sqrt(obj.params.mu(t_grid(end))) sqrt(obj.params.mu(t_grid(end)))]);
            
            xlabel(obj.ax1, 'x'); 
            ylabel(obj.ax1, 'u');
            obj.hTitle1 = title(obj.ax1, sprintf('t = %.3f', t_grid(1)));
            grid(obj.ax1, 'on');

            % Subplot 2: L2 norm
            obj.ax2 = subplot(2,3,4);
            obj.l2_line = semilogy(obj.ax2, 0, NaN, 'LineWidth', 2);
            xlabel(obj.ax2, '\mu'); 
            ylabel(obj.ax2, '||u||_{L^2}');
            title(obj.ax2, 'L2 Norm of u');
            grid(obj.ax2, 'on');

            % Subplot 3: Dominant mode
            obj.ax3 = subplot(2,3,5);
            obj.hLine3 = stairs(obj.ax3, 0, 0, 'LineWidth', 2);
            labels = cellstr("k_" + (0:numel(obj.params.k_j)-1));
            yline(obj.ax3, obj.params.k_j, '--r', labels, 'LineWidth', 2);    
            xlabel(obj.ax3, '\mu'); 
            ylabel(obj.ax3, 'Wavenumber (k)');
            xlim(obj.ax3, obj.params.mu(t_grid([1, end])));
            ylim(obj.ax3, [0, obj.params.k_j(end)]); 
            grid(obj.ax3, 'on');

            % Subplot 4: Fourier Spectrum (semilogy: handles zeros cleanly)
            obj.ax4 = subplot(2,3,6);
            obj.hFreq = semilogy(obj.ax4, ifftshift(obj.params.kx), ifftshift(abs(u0_hat)), 'LineWidth', 2);
            xlabel(obj.ax4, 'k'); 
            ylabel(obj.ax4, '$|\hat{u}|$', 'Interpreter', 'latex');
            xlim(obj.ax4, [-obj.params.kx(32), obj.params.kx(32)]);
            ylim(obj.ax4, [1e-25, 1e5]);
            grid(obj.ax4, 'on');

            % Video setup
            if isfield(obj.params, 'save_video') && obj.params.save_video
                obj.v = VideoWriter(obj.params.video_filename, 'Motion JPEG AVI');
                obj.v.FrameRate = obj.params.video_framerate;
                obj.v.Quality = 100;
                open(obj.v);
            end
            
            initialize@BifrucationMonitorHeadless(obj, u0_hat, t_grid);
        end

        function update(obj, u_hat, t)
            update@BifrucationMonitorHeadless(obj, u_hat, t);

            u = real(ifft(u_hat));

            % Update plots
            if mod(obj.step_idx - 1, obj.params.plot_every) == 0
                obj.hLine1.YData = u;
                obj.hTitle1.String = sprintf('mu = %.4f, t = %.1f', obj.params.mu(t), t);

                obj.l2_line.XData = obj.params.mu(obj.t_grid(1:obj.step_idx));
                obj.l2_line.YData = obj.l2_history(1:obj.step_idx);
                axis(obj.ax2, 'tight');

                valid_modes = obj.dominate_mode(1:obj.dom_mode_ind, :);
                set(obj.hLine3, 'XData', obj.params.mu([valid_modes(:, 1); t]), 'YData', [valid_modes(:, 2); valid_modes(end, 2)]);

                obj.hFreq.YData = ifftshift(abs(u_hat));

                drawnow;

                if isfield(obj.params, 'save_video') && obj.params.save_video && ~isempty(obj.v) && isgraphics(obj.fig)
                    frame = getframe(obj.fig);
                    writeVideo(obj.v, frame);
                end
            end
        end

        function finalize(obj)
            if isfield(obj.params, 'save_video') && obj.params.save_video && ~isempty(obj.v)
                close(obj.v);
            end
            if isgraphics(obj.fig)
                close(obj.fig);
            end
        end
    end
end
