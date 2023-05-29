% This script performs various analyses based on the fitted model parameters
% obtained from ModelFit.mat. It generates plots to compare different models
% and visualize relationships between parameters.
%
% Author: Jing-Jing Li (jl3676@berkeley.edu)
% Last Modified: 5/28/2023

%% Set up

% Clear all variables in the workspace
clear all;

% Load the fitted model parameters from ModelFit.mat
load('ModelFit.mat')

%% Plot parameter shifts from the static to dynamic model

% Find the index of the static model and dynamic model
model_static_ind = find(contains(cellfun(@(x) x.name, Ms, 'UniformOutput', false), 'static'));
model_dynamic_ind = find(contains(cellfun(@(x) x.name, Ms, 'UniformOutput', false), 'dynamic'));

% Sort the AIC differences
[out, idx] = sort(AICs(:, model_dynamic_ind) - AICs(:, model_static_ind));

% Create a figure for the plot
figure('Position', [300, 300, 1500, 800])

% Plot the parameter shift for each model parameter
for param = Ms{model_static_ind}.pnames(1:end-1)
    param_ind_1 = strcmp(Ms{model_static_ind}.pnames, param);
    param_ind_2 = strcmp(Ms{model_dynamic_ind}.pnames, param);
    subplot(4, 2, find(strcmp(param, Ms{model_static_ind}.pnames(1:end-1))))
    hold on
    [B1, I] = sort(All_Params{model_static_ind}(idx, param_ind_1));
    B2 = All_Params{model_dynamic_ind}(idx, param_ind_2);
    B2 = B2(I);
    for k = 1:length(B1)
        plot([k k], [B1(k) B2(k)], 'Color', 'k')
        hold on
    end
    hold on
    for k = 1:length(B1)
        color = [245/255 130/255 32/255; 104/255 73/255 3/255];
        if AICs(k, model_static_ind) > AICs(k, model_dynamic_ind)
            color = [112/255 172/255 66/255; 44/255 81/255 6/255];
        end
        plot(k, B1(k), '.', 'Color', color(1, :), 'MarkerSize', 15);
        plot(k, B2(k), '.', 'Color', color(2, :), 'MarkerSize', 10);
        hold on
    end
    xlabel("Sorted participant")
    ylim([Ms{model_static_ind}.pMin(param_ind_1) Ms{model_static_ind}.pMax(param_ind_1)])
    set(gca, 'TickLabelInterpreter', 'none')
    p = signrank(B1, B2);
    title(strjoin([param ' (p=' num2str(p, 3) ')']), 'Interpreter', 'none')
end

sgtitle('Parameter values')
set(gcf, 'Renderer', 'painters');
saveas(gcf, '../plots/param_shift.png')
saveas(gcf, '../plots/param_shift.svg')

%% Plot Epsilon and Lapse Relationship

eps_ind = find(strcmp('epsilon', Ms{model_static_ind}.pnames));
lapse_ind = find(strcmp('lapse', Ms{model_dynamic_ind}.pnames));

% Create a figure for the plot
figure

% Plot the relationship between log(epsilon) and log(lapse)
h = plot([-7 0], [-7 0]);
h.Color = 'k';
hold on
plot(log(All_Params{model_static_ind}(:, eps_ind) + .001), log(All_Params{model_dynamic_ind}(:, lapse_ind) + .001), '.', 'MarkerSize', 20, 'MarkerEdgeColor', "#EDB120")
hold on
inds = AICs(:, model_dynamic_ind) - AICs(:, model_static_ind) > 0;
[r, p] = corr(All_Params{model_static_ind}(inds, end), All_Params{model_dynamic_ind}(inds, lapse_ind), 'type', 'Kendall');
inds = AICs(:, model_dynamic_ind) - AICs(:, model_static_ind) < 0;
plot(log(All_Params{model_static_ind}(inds, eps_ind) + .001), log(All_Params{model_dynamic_ind}(inds, lapse_ind) + .001), '.', 'MarkerSize', 20, 'MarkerEdgeColor', "#77AC30")
xlabel('log(epsilon)')
ylabel('log(lapse)')
legend({'', 'static better', 'dynamic better'}, 'Location', 'best')
axis square
title(['Correlation between epsilon and lapse, r = ' num2str(r, '%.2f') ', p = ' num2str(p)])
saveas(gcf, '../plots/eps_lapse_corr.png')
saveas(gcf, '../plots/eps_lapse_corr.svg')

%% Plot 1 - Epsilon and Recover Relationship

eps_ind = find(strcmp('epsilon', Ms{model_static_ind}.pnames));
rec_ind = find(strcmp('recover', Ms{model_dynamic_ind}.pnames));

% Create a figure for the plot
figure

% Plot the relationship between 1 - epsilon and recover
h = plot([0 1], [0 1]);
h.Color = 'k';
hold on
plot(1 - All_Params{model_static_ind}(:, eps_ind), All_Params{model_dynamic_ind}(:, rec_ind), '.', 'MarkerSize', 20, 'MarkerEdgeColor', "#EDB120")
hold on
[r, p] = corr(1 - All_Params{model_static_ind}(:, eps_ind), All_Params{model_dynamic_ind}(:, rec_ind), 'type', 'Kendall');
inds = AICs(:, model_dynamic_ind) - AICs(:, model_static_ind) < 0;
plot(1 - All_Params{model_static_ind}(inds, eps_ind), All_Params{model_dynamic_ind}(inds, rec_ind), '.', 'MarkerSize', 20, 'MarkerEdgeColor', "#77AC30")
xlabel('1 - epsilon')
ylabel('rec')
legend({'', 'static better', 'dynamic better'}, 'Location', 'best')
axis square
title(['Correlation between 1 - epsilon and recover, r = ' num2str(r, '%.2f') ', p = ' num2str(p)])
saveas(gcf, '../plots/eps_rec_correlation.png')
saveas(gcf, '../plots/eps_rec_correlation.svg')

%% Plot Violins for Lapse - Epsilon

eps_ind = find(strcmp('epsilon', Ms{model_static_ind}.pnames));
lapse_ind = find(strcmp('lapse', Ms{model_dynamic_ind}.pnames));

% Create a figure for the plot
figure('Position', [300, 300, 500, 400])

% Set colors for the violins
colors = [245/255 130/255 32/255; 112/255 172/255 66/255];

plot([0 3], [0, 0], '--', 'Color', 'k')
hold on
inds = (AICs(:, model_dynamic_ind) - AICs(:, model_static_ind) >= 0);
to_plot_1 = ((All_Params{model_dynamic_ind}(inds, lapse_ind)) - (All_Params{model_static_ind}(inds, eps_ind)));
hold on
inds = (AICs(:, model_dynamic_ind) - AICs(:, model_static_ind) < 0);
to_plot_2 = ((All_Params{model_dynamic_ind}(inds, lapse_ind)) - (All_Params{model_static_ind}(inds, eps_ind)));
p = signrank(to_plot_2, 0, 'tail', 'left');
disp(['p = ' num2str(p) ' (dynamic better)'])
hold on
g = [zeros(length(to_plot_1), 1); ones(length(to_plot_2), 1)];
violinplot(([to_plot_1; to_plot_2]), g, 'ViolinColor', colors);
xticks([1 2])
xticklabels({'static better', 'dynamic better'})
ylabel('lapse - epsilon')
title('lapse')
saveas(gcf, '../plots/lapse_violins.png')
saveas(gcf, '../plots/lapse_violins.svg')

%% Plot Violins for Recover - (1 - Epsilon)

eps_ind = find(strcmp('epsilon', Ms{model_static_ind}.pnames));
rec_ind = find(strcmp('recover', Ms{model_dynamic_ind}.pnames));

% Create a figure for the plot
figure('Position', [300, 300, 500, 400])

% Set colors for the violins
colors = [245/255 130/255 32/255; 112/255 172/255 66/255];

plot([0 3], [0, 0], '--', 'Color', 'k')
hold on
inds = (AICs(:, model_dynamic_ind) - AICs(:, model_static_ind) >= 0);
to_plot_1 = ((All_Params{model_dynamic_ind}(inds, rec_ind)) - 1 + (All_Params{model_static_ind}(inds, eps_ind)));
hold on
inds = (AICs(:, model_dynamic_ind) - AICs(:, model_static_ind) < 0);
to_plot_2 = ((All_Params{model_dynamic_ind}(inds, rec_ind)) - 1 + (All_Params{model_static_ind}(inds, eps_ind)));
p = signrank(to_plot_2, 0, 'tail', 'left');
disp(['p = ' num2str(p) ' (dynamic better)'])
hold on
g = [zeros(length(to_plot_1), 1); ones(length(to_plot_2), 1)];
violinplot(([to_plot_1; to_plot_2]), g, 'ViolinColor', colors);
xticks([1 2])
xticklabels({'static better', 'dynamic better'})
ylabel('recover - (1 - epsilon)')
title('recover')
saveas(gcf, '../plots/rec_violins.png')
saveas(gcf, '../plots/rec_violins.svg')
