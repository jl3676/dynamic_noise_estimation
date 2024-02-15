function data = static_model(theta, payoff, num_trials, p_latent)
% Generates data for the Iowa Gambling Task (IGT) using the static model.
%
% Inputs:
%   - theta: Parameter values for the dynamic model. It is a vector with the following elements:
%     - theta(1): Learning rate (alpha)
%     - theta(2): Inverse temperature (beta)
%     - theta(3): Sensitivity parameter (sensitivity)
%     - theta(4): Decay rate (decay)
%     - theta(5): Exploration rate (phi)
%     - theta(6): Uniform level of noise (epsilon)
%   - payoff: Payoff schedule identifier.
%   - num_trials: Number of trials to generate.
%   - p_latent (optional): Probability of latent state = 1 (0 or 1) for each trial.
%
% Output:
%   - data: Generated data matrix of size (num_trials x 3) containing choice, gain, and loss for each trial.
%     - data(:, 1): Choices made by the subject
%     - data(:, 2): Rewards received by the subject
%     - data(:, 3): Losses received by the subject
%
% Author: Jing-Jing Li (jl3676@berkeley.edu)
% Last Modified: 5/28/2023

alpha = theta(1);     % Learning rate
beta = theta(2);      % Inverse temperature
sensitivity = theta(3);     % Sensitivity
decay = theta(4);     % Decay rate
phi = theta(5);       % Exploration rate
epsilon = theta(6);   % Uniform level of noise
lapse = 0;            % Lapse rate - not used in static model
recover = 1;          % Recover rate - not used in static model

engaged = 1;
nA = 4; % Number of available actions

Q = ones(nA,1) / nA;
explore = ones(nA,1) / nA;
exploit = ones(nA,1) / nA;

data = zeros(num_trials, 3);

payoff_schedule = readtable(['../data/payoff_schedule_' num2str(payoff) '.csv']);

for k = 1:num_trials

    data(k, 4) = engaged;

    if nargin < 4
        if engaged
            b = epsilon / nA + (1 - epsilon) * exp(beta * Q - logsumexp(beta * Q));
            if rand < lapse
                engaged = 0;
            end
        else
            b = ones(nA,1) / nA;
            if rand < recover
                engaged = 1;
            end
        end
    else
        if rand < p_latent(k)
            b = epsilon / nA + (1 - epsilon) * exp(beta * Q - logsumexp(beta * Q));
        else
            b = ones(nA,1) / nA;
        end
    end

    choice = randsample(1:4, 1, true, b);

    [gain, loss] = IGT_gain_loss(choice, payoff_schedule, k);
    
    exploit = decay * exploit;
    exploit(choice) = exploit(choice) + (gain ^ sensitivity - abs(loss) ^ sensitivity);

    explore = explore + alpha * (phi - explore);
    explore(choice) = 0;
    
    Q = explore + exploit;
    
    data(k, 1:3) = [choice, gain, loss];
    data(k, 5) = b(choice);
end

end

function [gain, loss] = IGT_gain_loss(choice, payoff_schedule, trial)
% IGT_GAIN_LOSS returns the gain and loss for the given deck choice in the Iowa Gambling Task.
%
% Inputs:
%   - choice: Selected deck choice (1, 2, 3, or 4).
%   - payoff_schedule: Table containing the payoff schedule.
%   - trial: Current trial index.
%
% Outputs:
%   - gain: Gain value for the selected deck.
%   - loss: Loss value for the selected deck.

it = rem(trial, length(payoff_schedule.Trial));
if it == 0
    it = length(payoff_schedule.Trial);
end
gain_values = [payoff_schedule.winA(it) payoff_schedule.winB(it) payoff_schedule.winC(it) payoff_schedule.winD(it)];
loss_values = [payoff_schedule.lossA(it) payoff_schedule.lossB(it) payoff_schedule.lossC(it) payoff_schedule.lossD(it)];

gain = gain_values(choice);
loss = loss_values(choice);
end
