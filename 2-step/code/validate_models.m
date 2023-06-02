% Clear all variables and functions from the workspace
clear all

% Load model fit data
load("ModelFit.mat")

% Define the names of the models
model1 = 'static_model';
model2 = 'dynamic_model';

% Find the indices of the models in the model data
model_ind1 = find(strcmp(cellfun(@(x) x.name, Ms(:), 'UniformOutput', false), model1));
model_ind2 = find(strcmp(cellfun(@(x) x.name, Ms(:), 'UniformOutput', false), model2));

%% Plot behavior

% Create a figure
figure('Position', [300 300 600 400])

% Compute stay probabilities for each condition
stay_probs = zeros(4, length(data));
for s = 1:length(data)
    this_data = data(s).data;

    money = this_data.money;
    money_prev = [-1; money(1:end-1)];

    transition = double(strcmp(this_data.transition, 'common'));
    transition_prev = [-1; transition(1:end-1)];

    choice1 = this_data.choice1;
    choice1_prev = [0; choice1(1:end-1)];

    stay = choice1_prev == choice1;

    stay_probs(1, s) = sum(stay(transition_prev == 1 & money_prev == 1)) / sum(transition_prev == 1 & money_prev == 1);
    stay_probs(2, s) = sum(stay(transition_prev == 0 & money_prev == 1)) / sum(transition_prev == 0 & money_prev == 1);
    stay_probs(3, s) = sum(stay(transition_prev == 1 & money_prev == 0)) / sum(transition_prev == 1 & money_prev == 0);
    stay_probs(4, s) = sum(stay(transition_prev == 0 & money_prev == 0)) / sum(transition_prev == 0 & money_prev == 0);
end

% Compute mean and standard error of stay probabilities
stay_probs_mean = squeeze(nanmean(stay_probs, 2));
stay_probs_sem = squeeze(nanstd(stay_probs, 1, 2)) / sqrt(length(data));

% Define bar plot data
y = [stay_probs_mean(1) stay_probs_mean(2); stay_probs_mean(3) stay_probs_mean(4)];
err = [stay_probs_sem(1) stay_probs_sem(2); stay_probs_sem(3) stay_probs_sem(4)];

% Plot bar chart
b = bar(y, 'grouped', 'LineWidth', 1.5);
b(1).FaceColor = [0 .5 .5];
b(1).EdgeColor = [0 .9 .9];
b(2).FaceColor = [.5 .4 .8];
b(2).EdgeColor = [.8 .6 .9];
ylim([0.5 0.85])
hold on

% Compute x positions for error bars
[ngroups, nbars] = size(y);
x = nan(nbars, ngroups);
for i = 1:nbars
    x(i, :) = b(i).XEndPoints;
end

% Add error bars to the plot
errorbar(x', y, err, 'k', 'LineStyle', 'none', 'LineWidth', 1)
hold on

%% Plot simulations of model1

% Set the number of iterations for model simulations
niters = 100;
stay_probs = zeros(4, length(data), niters);
fitted_params = All_Params{model_ind1};

% Perform model simulations
for s = 1:length(data)
    for it = 1:niters
        theta = fitted_params(s,:);
        this_data = feval(model1, theta);

        money = this_data.money;
        money_prev = [-1; money(1:end-1)];

        transition = double(strcmp(this_data.transition, 'common'));
        transition_prev = [-1; transition(1:end-1)];

        choice1 = this_data.choice1;
        choice1_prev = [0; choice1(1:end-1)];

        stay = choice1_prev == choice1;

        stay_probs(1, s, it) = sum(stay(transition_prev == 1 & money_prev == 1)) / sum(transition_prev == 1 & money_prev == 1);
        stay_probs(2, s, it) = sum(stay(transition_prev == 0 & money_prev == 1)) / sum(transition_prev == 0 & money_prev == 1);
        stay_probs(3, s, it) = sum(stay(transition_prev == 1 & money_prev == 0)) / sum(transition_prev == 1 & money_prev == 0);
        stay_probs(4, s, it) = sum(stay(transition_prev == 0 & money_prev == 0)) / sum(transition_prev == 0 & money_prev == 0);
    end
end

% Compute mean stay probabilities across iterations
stay_probs = squeeze(nanmean(stay_probs, 3));

stay_probs_mean = squeeze(nanmean(stay_probs, 2));
stay_probs_sem = squeeze(nanstd(stay_probs, 1, 2)) / sqrt(length(data));

y = [stay_probs_mean(1) stay_probs_mean(2); stay_probs_mean(3) stay_probs_mean(4)];
err = [stay_probs_sem(1) stay_probs_sem(2); stay_probs_sem(3) stay_probs_sem(4)];

[ngroups, nbars] = size(y);
x = nan(nbars, ngroups);
for i = 1:nbars
    x(i, :) = b(i).XEndPoints - 0.05;
end

% Add error bars to the plot
errorbar(x', y, err, 'k', 'LineStyle', 'none', 'LineWidth', 1, 'Marker', '.', 'MarkerEdgeColor', [245/255 130/255 32/255], 'MarkerSize', 30)
hold on

%% Plot simulations of model2

stay_probs = zeros(4, length(data), niters);
fitted_params = All_Params{model_ind2};

% Perform model simulations
for s = 1:length(data)
    for it = 1:niters
        theta = fitted_params(s,:);
        latent_st_traj = feval(strjoin([model2, "_latent"], ""), theta, data(s).data);
        latent_st_traj = latent_st_traj(:, 1);
        this_data = feval(model2, theta, latent_st_traj);

        money = this_data.money;
        money_prev = [-1; money(1:end-1)];

        transition = double(strcmp(this_data.transition, 'common'));
        transition_prev = [-1; transition(1:end-1)];

        choice1 = this_data.choice1;
        choice1_prev = [0; choice1(1:end-1)];

        stay = choice1_prev == choice1;

        stay_probs(1, s, it) = sum(stay(transition_prev == 1 & money_prev == 1)) / sum(transition_prev == 1 & money_prev == 1);
        stay_probs(2, s, it) = sum(stay(transition_prev == 0 & money_prev == 1)) / sum(transition_prev == 0 & money_prev == 1);
        stay_probs(3, s, it) = sum(stay(transition_prev == 1 & money_prev == 0)) / sum(transition_prev == 1 & money_prev == 0);
        stay_probs(4, s, it) = sum(stay(transition_prev == 0 & money_prev == 0)) / sum(transition_prev == 0 & money_prev == 0);
    end
end

% Compute mean stay probabilities across iterations
stay_probs = squeeze(nanmean(stay_probs, 3));

stay_probs_mean = squeeze(nanmean(stay_probs, 2));
stay_probs_sem = squeeze(nanstd(stay_probs, 1, 2)) / sqrt(length(data));

y = [stay_probs_mean(1) stay_probs_mean(2); stay_probs_mean(3) stay_probs_mean(4)];
err = [stay_probs_sem(1) stay_probs_sem(2); stay_probs_sem(3) stay_probs_sem(4)];

[ngroups, nbars] = size(y);
x = nan(nbars, ngroups);
for i = 1:nbars
    x(i, :) = b(i).XEndPoints + 0.05;
end

% Add error bars to the plot
errorbar(x', y, err, 'k', 'LineStyle', 'none', 'LineWidth', 1, 'Marker', '.', 'MarkerEdgeColor', [112/255 172/255 66/255], 'MarkerSize', 30)
hold on

% Add legend and titles
hleg = legend({'Common', 'Rare', '', '', 'Static', '', 'Dynamic', ''}, 'Location', 'best');
htitle = get(hleg, 'Title');
set(htitle, 'String', 'Previous trial transition')
title('2-step model validation')

% Save the figure
saveas(gcf, '../plots/model_validation.svg')
saveas(gcf, '../plots/model_validation.png')
