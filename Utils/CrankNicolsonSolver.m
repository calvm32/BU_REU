classdef CrankNicolsonSolver < Solver
    %CRANKNICOLSONSOLVER An explicit for nonlinear, semi-explicit linear 
    % time stepper for solving odes/evolution pdes.
    %   Supports both static and time-varying linear operators.

    properties
        numerator_coeff
        denominator
    end

    methods
        function obj = init_problem(obj, problem, dt)            
            arguments
                obj
                problem EvolutionProblem 
                dt double
            end

            obj.problem = problem;

            % If linearOperator is static, we can precompute the coefficients
            if ~isa(problem.linearOperator, 'function_handle')
                obj.numerator_coeff = 1 + 0.5 * dt * problem.linearOperator;
                obj.denominator = 1 - 0.5 * dt * problem.linearOperator;
            end
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

            % If linearOperator is a function handle, evaluate it dynamically
            if isa(obj.problem.linearOperator, 'function_handle')
                L = obj.problem.linearOperator(t);
                num_coeff = 1 + 0.5 * dt * L;
                den_coeff = 1 - 0.5 * dt * L;
            else
                num_coeff = obj.numerator_coeff;
                den_coeff = obj.denominator;
            end

            up = (num_coeff .* u + dt * obj.problem.nonlinearOperator(u, t)) ./ den_coeff;
        end
    end
end
