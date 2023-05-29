function data = dynamic_model(theta, payoff, num_trials, p_latent)
% Generates data for the Iowa Gambling Task (IGT) using the dynamic model.
%
% Inputs:
%   - theta: Model parameters [alpha, beta, sensitivity, decay, phi, lapse, recover].
%   - payoff: Payoff schedule identifier.
%   - num_trials: Number of trials to generate.
%   - p_latent (optional): Probability of latent attentional state (0 or 1) for each trial.
%
% Output:
%   - data: Generated data matrix of size (num_trials x 3) containing choice, gain, and loss for each trial.

alpha = theta(1);
beta = theta(2);
sensitivity = theta(3);
decay = theta(4);
phi = theta(5);
epsilon = 0;
lapse = theta(6);
recover = theta(7);

engaged = 1;
nA = 4;

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
