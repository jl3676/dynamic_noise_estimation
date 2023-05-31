%% Validates the static and dynamic models against behavior

% Clear workspace
clear all;

% Load fitted model parameters
load('ModelFit.mat')

% Define model names
model1 = 'static_model';
model2 = 'dynamic_model';

% Find indices of the models in the parameter array
model_ind1 = find(strcmp(cellfun(@(x) x.name, Ms(:), 'UniformOutput', false), model1));
model_ind2 = find(strcmp(cellfun(@(x) x.name, Ms(:), 'UniformOutput', false), model2));

% Define set sizes for simulations
set_sizes = [3 6];

%% Run simulations for model 1
fitted_params = All_Params{model_ind1};
learning_curves_sim1 = zeros(length(set_sizes), length(fitted_params), 13);

for k = 1:length(fitted_params)
    s = subjects(k);
    params = fitted_params(k, 1:end-1);
    K = fitted_params(k, end);

    subj_learning_curves_sim_3 = zeros(13, 1);
    subj_learning_curves_sim_6 = zeros(13, 1);

    % Fetch real data for the subject
    T = find(Alldata.ID == s & Alldata.phase == 0);
    realData = [Alldata.stim(T) Alldata.choice(T) Alldata.corchoice(T) Alldata.ns(T) Alldata.learningblock(T)];
    realData = realData(realData(:, 2) > 0, :);

    % Count the number of unique set sizes in the real data
    ns_counts = zeros(length(set_sizes), 1);
    for b = unique(realData(:, 5))'
        ns = realData(find(realData(:, 5) == b, 1), 4);
        ns_counts(set_sizes == ns) = ns_counts(set_sizes == ns) + 1;
    end

    niters = 1000;
    parfor it = 1:niters
        % Simulate data using the model
        data = feval(model1, params, K, realData);

        % Compute learning curves for each block
        for block = unique(data(:, 5))'
            block_data = data(data(:, 5) == block, :);
            set_size = block_data(1, 4);
            this_lc = arrayfun(@(i) mean(block_data(i:i + set_size - 1, 3)), 1:set_size:size(block_data, 1) - set_size + 1)';
            if set_size == 3
                subj_learning_curves_sim_3 = subj_learning_curves_sim_3 + this_lc;
            elseif set_size == 6
                subj_learning_curves_sim_6 = subj_learning_curves_sim_6 + this_lc;
            end
        end
    end

    % Compute average learning curves for each set size
    learning_curves_sim1(1, k, :) = subj_learning_curves_sim_3 / (niters * ns_counts(1));
    learning_curves_sim1(2, k, :) = subj_learning_curves_sim_6 / (niters * ns_counts(2));
end

%% Run simulations for model 2
fitted_params = All_Params{model_ind2};
learning_curves_sim2 = zeros(length(set_sizes), length(fitted_params), 13);

for k = 1:length(fitted_params)
    s = subjects(k);
    params = fitted_params(k, 1:end-1);
    K = fitted_params(k, end);

    subj_learning_curves_sim_3 = zeros(13, 1);
    subj_learning_curves_sim_6 = zeros(13, 1);

    % Fetch real data for the subject
    T = find(Alldata.ID == s & Alldata.phase == 0);
    realData = [Alldata.stim(T) Alldata.choice(T) Alldata.corchoice(T) Alldata.ns(T) Alldata.learningblock(T)];
    realData = realData(realData(:, 2) > 0, :);

    % Count the number of unique set sizes in the real data
    ns_counts = zeros(length(set_sizes), 1);
    for b = unique(realData(:, 5))'
        ns = realData(find(realData(:, 5) == b, 1), 4);
        ns_counts(set_sizes == ns) = ns_counts(set_sizes == ns) + 1;
    end

    niters = 1000;
    parfor it = 1:niters
        % Simulate data using the model
        data = feval(model2, params, K, realData);

        % Compute learning curves for each block
        for block = unique(data(:, 5))'
            block_data = data(data(:, 5) == block, :);
            set_size = block_data(1, 4);
            this_lc = arrayfun(@(i) mean(block_data(i:i + set_size - 1, 3)), 1:set_size:size(block_data, 1) - set_size + 1)';
            if set_size == 3
                subj_learning_curves_sim_3 = subj_learning_curves_sim_3 + this_lc;
            elseif set_size == 6
                subj_learning_curves_sim_6 = subj_learning_curves_sim_6 + this_lc;
            end
        end
    end

    % Compute average learning curves for each set size
    learning_curves_sim2(1, k, :) = subj_learning_curves_sim_3 / (niters * ns_counts(1));
    learning_curves_sim2(2, k, :) = subj_learning_curves_sim_6 / (niters * ns_counts(2));
end

%% Plot behavior
learning_curves = cell(length(set_sizes), 1);
for set_size = set_sizes
    learning_curves{set_size == set_sizes} = NaN(1, 13);
end

for k = 1:length(subjects)
    s = subjects(k);

    % Fetch real data for the subject
    T = find(Alldata.ID == s & Alldata.phase == 0);
    data = [Alldata.stim(T) Alldata.choice(T) Alldata.cor(T) Alldata.ns(T) Alldata.learningblock(T)];
    data = data(data(:, 2) > 0, :);

    subj_learning_curves = cell(length(set_sizes), 1);
    for set_size = set_sizes
        subj_learning_curves{set_size == set_sizes} = NaN(1, 13);
    end

    % Compute learning curves for each block
    for block = unique(data(:, 5))'
        block_data = data(data(:, 5) == block, :);
        set_size = block_data(1, 4);
        temp = arrayfun(@(i) mean(block_data(i:i + set_size - 1, 3)), 1:set_size:size(block_data, 1) - set_size + 1)';
        subj_learning_curves{set_sizes == set_size}(end + 1, 1:length(temp)) = temp;
    end

    % Compute average learning curves for each set size
    for set_size = set_sizes
        learning_curves{set_size == set_sizes}(end + 1, :) = nanmean(subj_learning_curves{set_size == set_sizes}(2:end, :), 1);
    end
end

%% Plot
figure('Position', [300 300 1000 500])
for set_size = set_sizes
    subplot(1, 2, find(set_size == set_sizes))
    x = 1:13;
    y = nanmean(learning_curves{set_size == set_sizes}, 1);
    err = nanstd(learning_curves{set_size == set_sizes}, 1) / sqrt(length(subjects));
    errorbar(x, y, err, 'Color', [0.4660 0.6740 0.1880])
    hold on
    y_sim1 = nanmean(squeeze(learning_curves_sim1(set_size == set_sizes, :, :)), 1);
    err_sim1 = nanstd(squeeze(learning_curves_sim1(set_size == set_sizes, :, :)), 1) / sqrt(length(subjects));
    plot(x, y_sim1, 'Color', [0.4940 0.1840 0.5560])
    hold on
    y_sim2 = nanmean(squeeze(learning_curves_sim2(set_size == set_sizes, :, :)), 1);
    err_sim2 = nanstd(squeeze(learning_curves_sim2(set_size == set_sizes, :, :)), 1) / sqrt(length(subjects));
    plot(x, y_sim2, 'Color', [0.8500 0.3250 0.0980])
    xlim([1 x(end)])
    xlabel('Iteration')
    ylabel('Reward')
    legend({'Human', model1, model2}, 'Interpreter', 'none', 'Location', 'best')
    title(['NS = ' num2str(set_size)])
end

% Save the plot as an image and SVG file
saveas(gcf, '../plots/model_validation.png')
saveas(gcf, '../plots/model_validation.svg')