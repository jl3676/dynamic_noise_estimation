function data = dynamic_model(theta, payoff, num_trials, p_latent)
% Generates data for the Iowa Gambling Task (IGT) using the same model whose likelihood is computed by the VSE_llh function.
% Inputs:
%   - theta: parameters vector (same as in VSE_llh)
%   - payoff: payoff scheme (1, 2, or 3)
%   - num_trials: number of trials to generate data for
%   - p_latent: the probability that the participant is in the enaged state
%   over trials
% Output:
%   - data: generated data (num_trials x 3 matrix, with columns for choices, gains, and losses)

alpha = theta(1);
beta = theta(2);
sensitivity = theta(3);
decay = theta(4);
phi = theta(5);
epsilon = 0;
lapse = theta(6);
rec = theta(7);

engaged = 1;
nA = 4;

Q = ones(nA,1) / nA;
explore = ones(nA,1) / nA;
exploit = ones(nA,1) / nA;

data = zeros(num_trials, 3);

payoff_schedule = readtable(['../data/payoff_schedule_' num2str(payoff) '.csv']);

for k = 1:num_trials

    if nargin < 4
        if engaged
            b = epsilon / nA + (1 - epsilon) * exp(beta * Q - logsumexp(beta * Q));
            if rand < lapse
                engaged = 0;
            end
        else
            b = ones(nA,1) / nA;
            if rand < rec
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
    
    data(k, :) = [choice, gain, loss];
end

end

function [gain, loss] = IGT_gain_loss(choice, payoff_schedule, trial)
% Returns the gain and loss for the given deck choice in the Iowa Gambling Task.
it = rem(trial, length(payoff_schedule.Trial));
if it == 0
    it = length(payoff_schedule.Trial);
end
gain_values = [payoff_schedule.winA(it) payoff_schedule.winB(it) payoff_schedule.winC(it) payoff_schedule.winD(it)];
loss_values = [payoff_schedule.lossA(it) payoff_schedule.lossB(it) payoff_schedule.lossC(it) payoff_schedule.lossD(it)];

gain = gain_values(choice);
loss = loss_values(choice);
end