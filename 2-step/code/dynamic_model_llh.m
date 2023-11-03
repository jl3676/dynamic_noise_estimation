function nllh = dynamic_model_llh(theta, data)
% dynamic_model_llh computes the negative log likelihood (nllh) of the dynamic model given the parameters (theta) and data.

% Parameters:
alpha = theta(1);           % softmax inverse temperature
beta_mb = theta(2);         % learning rate
beta_mf = theta(3);         % eligibility trace decay
beta = theta(4);            % mixing weight
lambda = theta(5);          % stimulus stickiness
stickiness = theta(6);      % response stickiness
lapse = theta(7);           % lapse
recover = theta(8);         % recover

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
T = [1 - recover, lapse; recover, 1 - lapse];  % transition matrix for attention state

llh = 0;
N = length(choice1short);
p = [(1+lapse-recover)/2 (1-lapse+recover)/2];     % initialize p(att)
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
    lt = log(lik_choice1);
    
    % Softmax function for choice 2
    numerator = exp(beta * Qd(stateshort(i), choice2short(i)));
    denominator = sum(exp(beta * Qd(stateshort(i), :)));
    lik_choice2 = numerator / denominator;
    
    lt = lt + log(lik_choice2);
    latent(i, :) = [p(2), exp(lt)];
    lt = log((1/nA) * (1/nA) * p(1) + exp(lt) * p(2));
    
    llh = llh + lt;
    
    % Update p(att)
    p = ((1/nA) * (1/nA) * p(1) * T(:, 1) + lik_choice1 * lik_choice2 * p(2) * T(:, 2)) / exp(lt);
    
    % Compute Reward Prediction Error (RPE)
    tdQ = [0, 0];
    tdQ(1) = Qd(stateshort(i), choice2short(i)) - Qd(1, choice1short(i));
    tdQ(2) = moneyshort(i) - Qd(stateshort(i), choice2short(i));
    
    % Model-free update
    Qd(1, choice1short(i)) = Qd(1, choice1short(i)) + alpha * tdQ(1) + lambda * alpha * tdQ(2);
    
    % Model-based update
    Qd(stateshort(i), choice2short(i)) = Qd(stateshort(i), choice2short(i)) + alpha * tdQ(2);
end

nllh = -llh;  % return negative log likelihood
