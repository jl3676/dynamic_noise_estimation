% This script performs model identification and evaluation for the Dynamic Foraging task.
% It loads precomputed model fits and behavioral data, generates simulated data for each
% fitted model, and compares the models using the Akaike Information Criterion (AIC) and
% Bayesian Information Criterion (BIC). The results are visualized using confusion
% matrices to assess the accuracy of model identification.
%
% The script consists of the following main steps:
%
% 1. Load necessary data:
%    - Load precomputed model fits from 'ModelFit.mat'.
%    - Load behavioral data for the Dynamic Foraging task.
%
% 2. Prepare reward structure:
%    - Create a reward structure for each subject based on the behavioral data.
%
% 3. Model identification and evaluation:
%    - Iterate over the models and subjects.
%    - Generate simulated data for each subject using the fitted parameters.
%    - Fit each model to the simulated data and calculate the log-likelihood, AIC, and BIC.
%    - Identify the best model for each subject based on AIC and BIC.
%
% 4. Reshape and save results:
%    - Reshape the results for analysis and visualization.
%    - Save the results in 'ModelID.mat'.
%
% 5. Visualize model identification results:
%    - Plot confusion matrices showing the accuracy of model identification using AIC and BIC.
%    - Save the resulting plots as 'modelID.png' and 'modelID.svg'.
%
% Required Files:
%   - 'ModelFit.mat': Precomputed model fits.
%
% Output:
%   - 'ModelID.mat': Saved results of model identification and evaluation.
%   - 'modelID.png' and 'modelID.svg': Confusion matrices visualizing the model identification results.
%
% Dependencies:
%   - Optimization Toolbox (fmincon)
%   - Global Optimization Toolbox
%   - Parallel Computing Toolbox (for parallel execution)
%
% Author: Jing-Jing Li (jl3676@berkeley.edu)
% Last Modified: 5/28/2023

%% Set up
clear all
load('ModelFit.mat') % Load precomputed model fits

% Create a reward structure for each subject
rewardStructs = cell(length(subjects),1); % Cell array to store reward structure for each subject
for i=1:length(subjects)
    subject_data = data.(subjects{i}); % Behavioral data for current subject
    sess = fieldnames(subject_data); % Session names for current subject
    nsessions = length(sess); % Number of sessions for current subject

    rewardStruct = struct(); % Reward structure for current subject
    for s=1:nsessions
        rewardProbL = [subject_data.(sess{s}).rewardProbL]'; % Left reward probabilities for current session
        rewardProbR = [subject_data.(sess{s}).rewardProbR]'; % Right reward probabilities for current session
        rewardStruct.(sess{s}) = struct('rewardProbL', rewardProbL, 'rewardProbR', rewardProbR); % Store reward probabilities in reward structure
    end

    rewardStructs{i} = rewardStruct; % Store reward structure for current subject
end

%% Run model identification
% Iterate over models
num_subjects = length(subjects); % Total number of subjects
Ytest = cell(num_subjects,2); % True model names (for testing)
Ypredicted_AIC = cell(num_subjects,2); % Predicted model names using AIC
Ypredicted_BIC = cell(num_subjects,2); % Predicted model names using BIC
for M1=1:size(Ms,2) % Iterate over models
    model1 = Ms{M1}.name; % Name of generative model
    model_ind = find(strcmp(cellfun(@(x) x.name, Ms(:), 'UniformOutput', false), model1)); % Index of current model
    fitted_params = All_Params{model_ind}; % Fitted parameters for generative model

    llh = zeros(num_subjects,size(Ms,2)); % Log-likelihood matrix
    AIC = zeros(num_subjects,size(Ms,2)); % AIC matrix
    BIC = zeros(num_subjects,size(Ms,2)); % BIC matrix
    winner = zeros(num_subjects,size(Ms,2)); % Model winner matrix

    parfor subject_idx=1:num_subjects % Iterate over subjects in parallel
        % Repeat for 10 times per subject
        for rep=1:10
            % Generate data
            theta = fitted_params(subject_idx,:); % Fitted parameters for current subject
            this_data = feval(model1, theta, rewardStructs{subject_idx}); % Simulated data based on the current model and parameters
    
            llh_M2 = zeros(size(Ms,2),1); % Log-likelihood vector for each model
            AIC_M2 = zeros(size(Ms,2),1); % AIC vector for each model
            BIC_M2 = zeros(size(Ms,2),1); % BIC vector for each model
            
            for M2=1:size(Ms,2) % Iterate over models for comparison
                name = Ms{M2}.name; % Name of test model
                pmin = Ms{M2}.pMin; % Minimum parameter values for test model
                pmax = Ms{M2}.pMax; % Maximum parameter values for test model
                pdfs = Ms{M2}.pdfs; % Sampling functions for model parameters
    
                % Find optimal parameters
                myfitfun = @(p) feval([name,'_llh'],p,this_data); % Objective function for parameter fitting
                
                % Initialize parameter values
                par  = zeros(length(pmin),1);
                for p_ind=1:length(pmin)
                    par(p_ind) = pdfs{p_ind}(0); % Initialize parameters by sampling from the uniform distribution (0 is the random seed)
                end

                % Set starting values of dynamic model parameters to the best fit
                % static model parameters
                if contains(name, 'dynamic') % Check if the model is dynamic
                    dynamic_model = Ms{M2}; % Dynamic model for comparison
                    static_model_ind = find(contains(cellfun(@(x) x.name, Ms, 'UniformOutput', false), 'static')); % Index of corresponding static model
                    static_model = Ms{static_model_ind}; 
                    for z = 1:length(dynamic_model.pnames)
                        this_p = dynamic_model.pnames{z}; % Current parameter name
                        if sum(strcmp(this_p, static_model.pnames)) > 0 % Check if the parameter is present in the static model
                            par(strcmp(this_p, dynamic_model.pnames)) = All_Params{static_model_ind}(subject_idx, strcmp(this_p, Ms{static_model_ind}.pnames)); % Set dynamic parameter to the value of the corresponding static parameter
                        end
                    end
                    par(strcmp('lapse', dynamic_model.pnames)) = All_Params{static_model_ind}(subject_idx, strcmp('epsilon', static_model.pnames)); % Set dynamic lapse parameter to the value of the static epsilon parameter
                    par(strcmp('recover', dynamic_model.pnames)) = 1 - All_Params{static_model_ind}(subject_idx, strcmp('epsilon', static_model.pnames)); % Set dynamic recover parameter to (1 - static epsilon)
                end
    
                rng default % For reproducibility
                opts = optimoptions(@fmincon,'Algorithm','sqp');
                problem = createOptimProblem('fmincon','objective',...
                    myfitfun,'x0',par,'lb',pmin,'ub',pmax,'options',opts);
                gs = GlobalSearch;
                [param,this_llh] = run(gs,problem); % Find optimal parameters using global search
                
                ntrials = sum(cellfun(@(x) numel(x), struct2cell(this_data))); % Total number of trials
            
                llh_M2(M2) = this_llh; % Store log-likelihood for current model
                AIC_M2(M2) = 2 * this_llh + 2 * length(pmin); % Calculate AIC for current model
                BIC_M2(M2) = 2 * this_llh + log(ntrials) * length(pmin); % Calculate BIC for current model
            end
            
            llh(subject_idx,:) = llh(subject_idx,:) + llh_M2'; % Accumulate log-likelihood across repetitions
            AIC(subject_idx,:) = AIC(subject_idx,:) + AIC_M2'; % Accumulate AIC across repetitions
            BIC(subject_idx,:) = BIC(subject_idx,:) + BIC_M2'; % Accumulate BIC across repetitions
        end

        [~,ind] = min(AIC(subject_idx,:)); % Find the model index with the minimum AIC for current subject
        Ytest{subject_idx,M1} = Ms{M1}.name; % True model name (for testing)
        Ypredicted_AIC{subject_idx,M1} = Ms{ind}.name; % Predicted model name using AIC
        [~,ind] = min(BIC(subject_idx,:)); % Find the model index with the minimum BIC for current subject
        Ypredicted_BIC{subject_idx,M1} = Ms{ind}.name; % Predicted model name using BIC
    end
end

% Reshape the results for analysis
Ytest = reshape(Ytest, 2*num_subjects, 1); % Reshape true model names
Ypredicted_AIC = reshape(Ypredicted_AIC, 2*num_subjects, 1); % Reshape predicted model names using AIC
Ypredicted_BIC = reshape(Ypredicted_BIC, 2*num_subjects, 1); % Reshape predicted model names using BIC

% Save the results
save('ModelID')

%% Plot the confusion matrices
figure('Position',[300,300,1200,500])
subplot(1,2,1)
cm = confusionchart(Ytest,Ypredicted_AIC,'RowSummary','row-normalized'); % Confusion matrix for model identification using AIC
cm.Title = 'AIC';

subplot(1,2,2)
cm = confusionchart(Ytest,Ypredicted_BIC,'RowSummary','row-normalized'); % Confusion matrix for model identification using BIC
cm.Title = 'BIC';

sgtitle('Model identification (Dynamic Foraging)')
saveas(gcf,'../plots/modelID.png') % Save the plot as an image
saveas(gcf,'../plots/modelID.svg') % Save the plot as an SVG file
