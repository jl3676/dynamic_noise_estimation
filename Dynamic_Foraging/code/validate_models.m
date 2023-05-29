% Validates the static and dynamic models against behavior.
%
% Dependencies:
%   - Parallel Computing Toolbox (for parallel execution)
%
% Author: Jing-Jing Li (jl3676@berkeley.edu)
% Last Modified: 5/29/2023

clear all
load ModelFit.mat

%% Data Processing

% Set the number of trials to plot and the number of trials before a switch
ntrials_to_plot = 15;
n_trials_before = 0;

% Find the index of the static model and dynamic model in the cell array 'Ms'
model_static_ind = find(contains(cellfun(@(x) x.name, Ms, 'UniformOutput', false), 'static'));
model_dynamic_ind = find(contains(cellfun(@(x) x.name, Ms, 'UniformOutput', false), 'dynamic'));

% Get the names of the static and dynamic models
model_static = Ms{model_static_ind}.name;
model_dynamic = Ms{model_dynamic_ind}.name;

% Create a cell array to store reward structures for each subject
rewardStructs = cell(length(subjects), 1);

% Process data for each subject
for i = 1:length(subjects)
    subject_data = data.(subjects{i});
    sess = fieldnames(subject_data);
    nsessions = length(sess);

    % Create a reward structure for the current subject
    rewardStruct = struct();

    % Iterate over sessions for the current subject
    for s = 1:nsessions
        % Extract reward probabilities from the session data
        rewardProbL = [subject_data.(sess{s}).rewardProbL]';
        rewardProbR = [subject_data.(sess{s}).rewardProbR]';

        % Store the reward probabilities in the reward structure
        rewardStruct.(sess{s}) = struct('rewardProbL', rewardProbL, 'rewardProbR', rewardProbR);
    end

    % Store the reward structure for the current subject
    rewardStructs{i} = rewardStruct;
end

%% Plot Behavior

% Create a figure to display the behavior plots
figure('Position', [300 300 900 250])

% Initialize arrays to store behavior data
hllh_data = NaN(1, ntrials_to_plot * 2 + 1);
mllh_data = NaN(1, ntrials_to_plot * 2 + 1);

% Iterate over subjects
for s = subjects'
    this_data = data.(s{1});

    sessions = fieldnames(this_data);

    % Iterate over sessions for the current subject
    for sess = sessions'
        this_sess_data = this_data.(sess{1});

        % Exclude trials with CSminus and missing reward information
        this_sess_data(strcmp({this_sess_data.trialType}, "CSminus")) = [];
        this_sess_data(isnan([this_sess_data.rewardTime])) = [];
        this_sess_data(isnan([this_sess_data.rewardR]) & isnan([this_sess_data.rewardL])) = [];

        % Extract reward probabilities and rewards from the session data
        rewardProbL = [this_sess_data.rewardProbL];
        rewardProbR = [this_sess_data.rewardProbR];
        rewardL = [this_sess_data.rewardL];
        rewardR = [this_sess_data.rewardR];

        % Calculate changes in reward probabilities
        diffRewardProbL = diff(rewardProbL);
        diffRewardProbR = diff(rewardProbR);

        % Find indices of changes from low to high reward probability (L-to-H)
        changesL = find(diffRewardProbL ~= 0);
        % Find indices of changes from high to low reward probability (H-to-L)
        changesR = find(diffRewardProbR ~= 0);

        % Find high-low to low-high (HLLH) transitions
        hllh = [];
        mllh = [];
        for c = changesL
            % Exclude trials near the beginning or end of the session
            if c <= ntrials_to_plot || c >= length(rewardL) - ntrials_to_plot
                continue
            end

            % Check if there is a corresponding L-to-H change within a certain range
            if sum(changesR - c <= 10 & changesR - c >= 0) == 0
                continue
            end

            % Check the pattern of reward probability changes
            if diffRewardProbL(c) == 80
                c2 = changesR(changesR - c <= 10 & changesR - c >= 0);
                c2 = c2(1);

                if diffRewardProbR(c2) == -40
                    mllh(end+1) = c;
                elseif diffRewardProbR(c2) == -80
                    hllh(end+1) = c;
                end
            end
        end

        % Find medium-low to low-high (MLLH) transitions
        for c = changesR
            % Exclude trials near the beginning or end of the session
            if c <= ntrials_to_plot || c >= length(rewardL) - ntrials_to_plot
                continue
            end

            % Check if there is a corresponding H-to-L change within a certain range
            if sum(changesL - c <= 10 & changesL - c >= 0) == 0
                continue
            end

            % Check the pattern of reward probability changes
            if diffRewardProbR(c) == 80
                c2 = changesL(changesL - c <= 10 & changesL - c >= 0);
                c2 = c2(1);

                if diffRewardProbL(c2) == -40
                    mllh(end+1) = c;
                elseif diffRewardProbL(c2) == -80
                    hllh(end+1) = c;
                end
            end
        end

        % Store the behavioral data for HLLH transitions
        for point = hllh
            trials = point - ntrials_to_plot + 1 : point + ntrials_to_plot + 1;

            if rewardProbL(point + 1) == 90
                hllh_data(end + 1, :) = ~isnan(rewardR(trials));
            elseif rewardProbR(point + 1) == 90
                hllh_data(end + 1, :) = ~isnan(rewardL(trials));
            end
        end

        % Store the behavioral data for MLLH transitions
        for point = mllh
            trials = point - ntrials_to_plot + 1 : point + ntrials_to_plot + 1;
            if rewardProbL(point + 1) == 90
                mllh_data(end + 1, :) = ~isnan(rewardR(trials));
            elseif rewardProbR(point + 1) == 90
                mllh_data(end + 1, :) = ~isnan(rewardL(trials));
            end
        end
    end
end

% Plot the behavior data
subplot(1, 3, 1)
x = -ntrials_to_plot : ntrials_to_plot;
xline(0, 'k--')
hold on
moving_avg = 1;
y1 = movmean(nanmean(hllh_data, 1), moving_avg);
y1_err = movmean(nanstd(hllh_data, 1) / sqrt(sum(~isnan(hllh_data), 1)), moving_avg);
plot(x, y1, 'Color', 'k', 'LineWidth', 1.5)
hold on
patch([x fliplr(x)], [y1 fliplr(y1 + y1_err)], 'k', 'FaceAlpha', 0.1, 'Edgecolor', 'none')
patch([x fliplr(x)], [y1 fliplr(y1 - y1_err)], 'k', 'FaceAlpha', 0.1, 'Edgecolor', 'none')

y2 = movmean(nanmean(mllh_data, 1), moving_avg);
y2_err = movmean(nanstd(mllh_data, 1) / sqrt(sum(~isnan(mllh_data), 1)), moving_avg);
patch([x fliplr(x)], [y2 fliplr(y2 + y2_err)], [.5 .5 .5], 'FaceAlpha', 0.1, 'Edgecolor', 'none')
patch([x fliplr(x)], [y2 fliplr(y2 - y2_err)], [.5 .5 .5], 'FaceAlpha', 0.1, 'Edgecolor', 'none')
plot(x, y2, 'Color', [.5 .5 .5], 'LineWidth', 1.5)
xlim([-17, 17])
ylim([0 1])
xlabel('Trials from switch')
ylabel('Choice average')
legend({'', 'High-low to low-high transition', '', '', '', '', 'Medium-low to low-high transition'}, 'Location', 'best')
title('Mice')

%% Plot Simulations of Models

% Define the models and their corresponding indices
models = {model_static, model_dynamic};
model_inds = [model_static_ind, model_dynamic_ind];

% Iterate over the models
for modeli = 1:length(models)
    model = models{modeli};
    model_ind = model_inds(modeli);

    % Initialize arrays to store simulation results
    hllh_data = zeros(1, 31);
    mllh_data = zeros(1, 31);
    hllh_n = 0;
    mllh_n = 0;
    fitted_params = All_Params{model_ind};

    % Iterate over iterations
    parfor iter = 1:1000
        it = 0;

        % Iterate over subjects
        for s = subjects'
            it = it + 1;
            theta = fitted_params(it,:);
            this_data = feval(model, theta, rewardStructs{it});

            sessions = fieldnames(this_data)';

            % Iterate over sessions for the current subject
            for sess_ind = 1:length(sessions)
                sess = sessions(sess_ind);
                this_sess_data = this_data.(sess{1});
                rewardProbL = [this_sess_data.rewardProbL];
                rewardProbR = [this_sess_data.rewardProbR];
                rewardL = [this_sess_data.rewardL];
                rewardR = [this_sess_data.rewardR];

                % Calculate changes in reward probabilities
                diffRewardProbL = diff(rewardProbL);
                diffRewardProbR = diff(rewardProbR);

                % Find indices of changes from low to high reward probability (L-to-H)
                changesL = find(diff(rewardProbL) ~= 0);
                % Find indices of changes from high to low reward probability (H-to-L)
                changesR = find(diff(rewardProbR) ~= 0);

                % Find high-low to low-high (HLLH) transitions
                hllh = [];
                mllh = [];
                for c = changesL
                    % Exclude trials near the beginning or end of the session
                    if c <= ntrials_to_plot || c >= length(rewardL) - ntrials_to_plot
                        continue
                    end
                    % Check if there is a corresponding L-to-H change within a certain range
                    if sum(changesR - c <= 10 & changesR - c >= 0) == 0
                        continue
                    end
                    % Check the pattern of reward probability changes
                    if diffRewardProbL(c) == 80
                        c2 = changesR(changesR - c <= 10 & changesR - c >= 0);
                        if diffRewardProbR(c2) == -40
                            mllh(end+1) = c;
                        elseif diffRewardProbR(c2) == -80
                            hllh(end+1) = c;
                        end
                    end
                end

                % Find medium-low to low-high (MLLH) transitions
                for c = changesR
                    % Exclude trials near the beginning or end of the session
                    if c <= ntrials_to_plot || c >= length(rewardL) - ntrials_to_plot
                        continue
                    end
                    % Check if there is a corresponding H-to-L change within a certain range
                    if sum(changesL - c <= 10 & changesL - c >= 0) == 0
                        continue
                    end
                    % Check the pattern of reward probability changes
                    if diffRewardProbR(c) == 80
                        c2 = changesL(changesL - c <= 10 & changesL - c >= 0);
                        if diffRewardProbL(c2) == -40
                            mllh(end+1) = c;
                        elseif diffRewardProbL(c2) == -80
                            hllh(end+1) = c;
                        end
                    end
                end

                % Update the simulation results for HLLH transitions
                for point = hllh
                    trials = point - ntrials_to_plot + 1 : point + ntrials_to_plot + 1;
                    if rewardProbL(point + 1) == 90
                        hllh_data = hllh_data + ~isnan(rewardR(trials));
                    elseif rewardProbR(point + 1) == 90
                        hllh_data = hllh_data + ~isnan(rewardL(trials));
                    end
                    hllh_n = hllh_n + 1;
                end

                % Update the simulation results for MLLH transitions
                for point = mllh
                    trials = point - ntrials_to_plot + 1 : point + ntrials_to_plot + 1;
                    if rewardProbL(point + 1) == 90
                        mllh_data = mllh_data + ~isnan(rewardR(trials));
                    elseif rewardProbR(point + 1) == 90
                        mllh_data = mllh_data + ~isnan(rewardL(trials));
                    end
                    mllh_n = mllh_n + 1;
                end
            end
        end
    end

    % Plot the simulation results
    subplot(1, 3, modeli + 1)
    x = -ntrials_to_plot : ntrials_to_plot;
    xline(0, 'k--')
    hold on
    moving_avg = 1;
    y1 = movmean(hllh_data / hllh_n, moving_avg);
    plot(x, y1, 'Color', 'k', 'LineWidth', 1.5)
    hold on

    y2 = movmean(mllh_data / mllh_n, moving_avg);
    plot(x, y2, 'Color', [.5 .5 .5], 'LineWidth', 1.5)
    xlim([-17, 17])
    ylim([0 1])
    xlabel('Trials from switch')
    ylabel('Choice average')
    title(model, 'Interpreter', 'none')
end

%% Save Plots

% Save the figure as PNG and SVG files
saveas(gcf, '../plots/model_validation.png')
saveas(gcf, '../plots/model_validation.svg')
