% This script runs the "generate and recover" process for model parameters.
% It generates simulated data using the static model and recovers the model 
% parameters using optimization methods with the static model.

clear all;

%% Define Models
% Set options for the optimization process
options = optimoptions('fmincon','Display','off');

% Create a cell array to store the models
Ms = [];

% Define the static model and its parameter limits
curr_model = [];
curr_model.name = 'static_model';
curr_model.pMin = [1e-6, -1, 1e-6];
curr_model.pMax = [1, 1, 1];
curr_model.pnames = {'alpha', 'stick', 'epsilon'};

% Add the static model to the list of models
Ms{1} = curr_model;

% Define the dynamic model and its parameter limits
curr_model = [];
curr_model.name = 'dynamic_model';
curr_model.pMin = [1e-6, -1, 1e-6, 1e-6];
curr_model.pMax = [1, 1, 1, 1];
curr_model.pnames = {'alpha', 'stick', 'lapse', 'recover'};

% Add the dynamic model to the list of models
Ms{2} = curr_model;

%% Simulate Data with Model
num_subjects = 500;
pmin = Ms{1}.pMin;
pmax = Ms{1}.pMax;
names = Ms{1}.pnames;

alpha_sample = @(x) betarnd(3, 10);
stick_sample = @(x) normrnd(0, .1);
eps_sample = @(x) betarnd(1, 15);

sampling_funcs = {alpha_sample, stick_sample, eps_sample};

% Iterate over subjects to generate and recover data
genrec = cell(num_subjects, 1);

parfor it = 1:num_subjects
    % Simulate data
    theta = zeros(length(pmin), 1);
    for p = 1:length(sampling_funcs)
        theta(p) = sampling_funcs{p}(0);
    end
    data = static_model(theta);

    % Find optimal parameters
    par = pmin + rand(1, length(pmin)) .* (pmax - pmin);
    myfitfun = @(p) static_model_llh(p, data);

    rng shuffle
    fmincon_opts = optimoptions(@fmincon, 'Algorithm', 'sqp');
    problem = createOptimProblem('fmincon', 'objective', ...
        myfitfun, 'x0', par, 'lb', pmin, 'ub', pmax, 'options', fmincon_opts);
    gs = GlobalSearch;
    [param, llh] = run(gs, problem);

    genrec{it} = [theta param'];
end

%% Plot Generate and Recover Results
figure('Position', [300 300 900 300])

for p = 1:length(names)
    subplot(1, 3, p)
    x1 = cell2mat(cellfun(@(x) x(p, 1), genrec, 'uni', 0));
    x2 = cell2mat(cellfun(@(x) x(p, 2), genrec, 'uni', 0));
    scatter(x1, x2, 'Marker', '.')
    hold on
    
    if p == 3 || p ==  4
        x1 = x1 + .001;
        x2 = x2 + .001;
        set(gca, 'XScale', 'log')
        set(gca, 'YScale', 'log')
        b = polyfit(log(x1), log(x2), 1);
        fit = exp(b(2)) .* x1.^b(1);
        plot(x1, fit, 'Color', [.75 .75 .75])
    else
        lsline
    end
    
    title(names{p}, 'Interpreter', 'none')
    xlabel('True')
    ylabel('Recovered')
    hold on
    plot(ylim, ylim, 'r')
    xlim([min(x1) max(x1)])
    ylim([min(x2) max(x2)])
end

% Set the title of the overall figure
h = sgtitle('Static data recovered by static model');
h.Interpreter = 'none';

% Save the figure as PNG and SVG files
saveas(gcf, '../plots/genrec_static.png')
saveas(gcf, '../plots/genrec_static.svg')
