classdef GradStableIMEXSolver < Solver
    %GRADSTABLEIMEXSOLVER A stabilized IMEX Euler time stepper for the 1D Cahn-Hilliard equation.

    properties
        S double
        Laplacian_k double
        mu_func
        denominator
    end

    methods
        function obj = GradStableIMEXSolver(S, Laplacian_k, mu_func)
            arguments
                S double
                Laplacian_k double
                mu_func (1,1) function_handle
            end
            obj.S = S;
            obj.Laplacian_k = Laplacian_k;
            obj.mu_func = mu_func;
        end

        function obj = init_problem(obj, problem, dt)            
            arguments
                obj
                problem EvolutionProblem 
                dt double
            end
            obj.problem = problem;
            obj.denominator = 1 + dt * (obj.Laplacian_k.^2 - obj.S * obj.Laplacian_k);
        end

        function [up] = step(obj, u, t, dt)
            arguments
                obj
                u 
                t double
                dt double
            end

            if isempty(obj.problem)
                error("The problem has not been set");
            end

            % The nonlinear operator evaluates fft(u^3) dealiased
            u3_hat = obj.problem.nonlinearOperator(u, t);

            numerator = u ...
                + dt * (obj.mu_func(t) + obj.S) * obj.Laplacian_k .* u ...
                - dt * obj.Laplacian_k .* u3_hat;
            
            up = numerator ./ obj.denominator;
        end
    end
end
