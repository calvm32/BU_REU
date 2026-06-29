classdef (Abstract) Solver < handle
    %SOLVER Abstract class for evolution equation type solvers

    properties
        problem EvolutionProblem % TODO: Generalize 
    end

    methods (Abstract)
        obj = init_problem(obj, problem, dt)
        up = step(obj, u, t, dt)
    end
end