%% Generate and Recover Analysis Script

% This script runs generate and recover analysis for the model parameters.
% It uses the fitted model parameters obtained from ModelFit.mat file.
% The analysis involves simulating data based on the fitted parameters and
% then optimizing the simulated data to recover the parameters using
% maximum likelihood estimation.
% 
% Dependencies:
%   - Optimization Toolbox (fmincon)
%   - Global Optimization Toolbox
%   - Parallel Computing Toolbox (for parallel execution)
%
% Author: Jing-Jing Li (jl3676@berkeley.edu)
% Last Modified: 5/28/2023

%% Set up

% Clear all variables in the workspace
clear all;

% Specify the model to analyze
model = 'dynamic_model';

% Load the fitted model parameters from ModelFit.mat
load('ModelFit.mat')
model_ind = find(strcmp(cellfun(@(x) x.name, Ms(:), 'UniformOutput', false), model));
fitted_params = All_Params{model_ind};

%% Initialization

% Get the names of the model parameters
names = Ms{model_ind}.pnames;

% Define the minimum and maximum values for the parameters
pmin = Ms{model_ind}.pMin;
pmax = Ms{model_ind}.pMax;

% Get the prior distribution functions for the parameters
pdfs = Ms{model_ind}.pdfs;

% Get the number of iterations for analysis
n_iters = size(fitted_params, 1);

% Create a cell array to store the generate and recover results
genrec = cell(n_iters, 1);

%% Iterate

parfor it = 1:n_iters
    s = subjects(it);
    T = find(Alldata.ID == s & Alldata.phase == 0);
    realData = [Alldata.stim(T) Alldata.choice(T) Alldata.cor(T) Alldata.ns(T) Alldata.learningblock(T)];
    realData = realData(realData(:, 2) > 0, :);

    % Get the fitted parameters for this iteration
    theta = fitted_params(it, 1:end-1);
    realK = fitted_params(it, end);

    % Initialize parameter values for optimization
    par = zeros(length(pmin), 1);
    for p_ind = 1:length(pmin)
        par(p_ind) = pdfs{p_ind}(0); % Use an arbitrary value for initialization
    end

    % Simulate data based on the fitted parameters
    this_data = feval(model, theta, realK, realData);
    this_data(isnan(this_data(:,1)),:) = [];

    sofar = [];

    % Define the objective function for optimization
    for K = 2:5
        myfitfun = @(p) feval([model, '_llh'], p, K, this_data);
        rng default % For reproducibility
        opts = optimoptions(@fmincon,'Algorithm','sqp');
        problem = createOptimProblem('fmincon','objective', myfitfun, 'x0', par, 'lb', pmin, 'ub', pmax, 'options', opts);
        gs = GlobalSearch;
        [p, fval] = run(gs, problem);
        sofar(K-1, :) = [p', K, fval];
    end

    [llh, i] = min(sofar(:, end));
    param = sofar(i(1), 1:end-1);

    % Store the true and recovered parameters for this iteration
    theta = [theta realK];
    genrec{it} = [theta' param'];
end

%% Plot Generate and Recover

% Create a figure for the plot
figure('Position', [300 300 900 400])

% Plot each parameter's generate and recover results
for p = 1:length(names)
    subplot(2, 4, p)
    scatter(cell2mat(cellfun(@(x) x(p, 1), genrec, 'uni', 0)), cell2mat(cellfun(@(x) x(p, 2), genrec, 'uni', 0)), 30, '.')
    lsline
    title(names{p}, 'Interpreter', 'none')
    xlabel('True')
    ylabel('Recovered')
    hold on
    plot(xlim, xlim, 'r')
end

% Set the title of the figure
h = sgtitle(model);
h.Interpreter = 'none';

% Save the figure as PNG and SVG files
saveas(gcf, ['../plots/genrec_', model, '.png'])
saveas(gcf, ['../plots/genrec_', model, '.svg'])
