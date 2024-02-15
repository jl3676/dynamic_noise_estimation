function nllh = static_model_llh(theta, data)
% Computes the negative log-likelihood of data given parameters (theta) for a static model.
%
% Inputs:
%   - theta: Model parameters
%         theta(1): alpha - learning rate
%         theta(2): beta_mb - inverse softmax temperature parameter for
%         model-based learning
%         theta(3): beta_mf - inverse softmax temperature parameter for
%         model-free learning
%         theta(4): beta - inverse softmax temperature parameter for
%         the second stage
%         theta(5): lambda - discount factor between stages
%         theta(6): stickiness - choice stickiness
%         theta(7): epsilon - uniform level of noise
%   - data (struct)
%
% Output:
%   - nllh: Negative log-likelihood of the data given the model parameters
%
% Author: Jing-Jing Li (jl3676@berkeley.edu)
% Last Modified: 5/28/2023

% Parameters:
alpha = theta(1);           % learning rate
beta_mb = theta(2);         % softmax inverse temperature for model-based
beta_mf = theta(3);         % softmax inverse temperature for model-free
beta = theta(4);            % softmax inverse temperature for second stage
lambda = theta(5);          % discount factor
stickiness = theta(6);      % choice stickiness
epsilon = theta(7);         % uniform level of noise
lapse = 0;                  % lapse - not used in the static model
recover = 1;                % recover - not used in the static model

nA = 2;                     % number of actions

%% Preprocess data

% Start model-fitting at choice 10
choice1short = data.choice1(10:end);
choice2short = data.choice2(10:end);
moneyshort = data.money(10:end);
stateshort = data.state(10:end);

% Remove trials with missing responses
noanswertrials = find(choice1short == 0 | choice2short == 0 | stateshort == 0);
choice1short(noanswertrials) = [];
choice2short(noanswertrials) = [];
moneyshort(noanswertrials) = [];
stateshort(noanswertrials) = [];

% Define the prior choice
choice10 = data.choice1(9);
if choice10 == 0
    choice10 = data.choice1(8);
end

%% Initialize variables

Qd = zeros(3, 2);           % Q values for each state-action pair
Tm = [.7, .3; .3, .7];      % transition matrix for first-stage states
T = [1 - recover, lapse; recover, 1 - lapse];  % transition matrix for latent states

llh = 0;
N = length(choice1short);
p = [0, 1];                 % initialize p(engaged)
latent = zeros(N, 2);       % latent variables

%% Loop through each trial

for i = 1:N
    priorchoice = [0, 0];
    if i == 1
        priorchoice(choice10) = 1;
    else
        priorchoice(choice1short(i - 1)) = 1;
    end

    MaxQ = max(Qd, [], 2);
    MaxQS2 = MaxQ(2:end);
    
    % Compute model-based and model-free Q values
    Qmb = [sum(MaxQS2 .* Tm(:, 1)), sum(MaxQS2 .* Tm(:, 2))];
    Qmf = Qd(1, :);
    
    % Softmax function for choice 1
    numerator = exp(beta_mf * Qmf(1, choice1short(i)) + beta_mb * Qmb(1, choice1short(i)) + stickiness * priorchoice(choice1short(i)));
    denominator = sum(exp(beta_mf * Qmf(1, :) + beta_mb * Qmb(1, :) + (stickiness * priorchoice)));
    lik_choice1 = numerator / denominator;
    
    % Softmax function for choice 2
    numerator = exp(beta * Qd(stateshort(i), choice2short(i)));
    denominator = sum(exp(beta * Qd(stateshort(i), :)));
    lik_choice2 = numerator / denominator;
    
    lt = log(epsilon / (nA * nA) + (1 - epsilon) * lik_choice1 * lik_choice2);
    latent(i, :) = [p(2), exp(lt)];
    
    llh = llh + lt;

    % Update p(att)
    p = ((1/nA) * (1/nA) * p(1) * T(:, 1) + lik_choice1 * lik_choice2 * p(2) * T(:, 2)) / exp(lt);
    
    % Compute Reward Prediction Error (RPE) for model-free and model-based updates
    tdQ = [0, 0];
    tdQ(1) = Qd(stateshort(i), choice2short(i)) - Qd(1, choice1short(i));
    tdQ(2) = moneyshort(i) - Qd(stateshort(i), choice2short(i));
    
    % Model-free update
    Qd(1, choice1short(i)) = Qd(1, choice1short(i)) + alpha * tdQ(1) + lambda * alpha * tdQ(2);
    
    % Model-based update
    Qd(stateshort(i), choice2short(i)) = Qd(stateshort(i), choice2short(i)) + alpha * tdQ(2);
end

nllh = -llh;  % Return negative log-likelihood
end
