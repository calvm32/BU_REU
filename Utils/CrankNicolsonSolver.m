classdef CrankNicolsonSolver < Solver
    %CRANKNICOLSONSOLVER An explicit for nonlinear, semi-explicit linear 
    % time stepper for solving odes/evolution pdes.
    %   One may specify a dealias_mask to be applied at the end of each step 

    properties
        problem EvolutionProblem % TODO: Generalize 
        % TODO: add dealias_mask option
    end

    methods
        function obj = init_problem(obj, problem, dt)            
            arguments
                obj
                problem EvolutionProblem 
                dt double
            end

            obj.problem = problem;
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

            for n = 2:length(t)
                % explicit nonlinear, semi-explicit linear
                up = ((1 + 0.5 * dt * obj.problem.linearOperator).*u ...
                    + dt*obj.problem.nonlinearOperator(u, t)) ...
                    ./ (1 - 0.5*dt*L_operator);
            
                % TODO: Support dealias_mask
                u_hat = dealias_mask .* u_hat;

            end
        end
    end
end


