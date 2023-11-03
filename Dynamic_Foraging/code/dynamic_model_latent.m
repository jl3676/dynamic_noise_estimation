function latent = dynamic_model_latent(theta, data)
% Computes the full negative log likelihood of data given parameters (theta)
% and returns the latent state probability and choice probability
% trajectories.
%
% Input:
%   - theta: Model parameters (1x9 vector)
%     - theta(1:2): Alpha parameters (alpha- and alpha+)
%     - theta(3): Beta parameter
%     - theta(4): Forget parameter
%     - theta(5): Bias parameter
%     - theta(6): Alpha_v parameter
%     - theta(7): Phi parameter
%     - theta(8): Lapse parameter
%     - theta(9): Recover parameter
%   - data: Struct containing behavioral data
%
% Output:
%   - latent: matrix containing latent state and choice probability
%   trajectories
%
% Author: Jing-Jing Li (jl3676@berkeley.edu)
% Last Modified: 5/28/2023

% Extract parameter values
alpha = theta(1:2);      % Alpha parameters (alpha- and alpha+)
beta = theta(3);        % Beta parameter
forget = theta(4);      % Forget parameter
bias = theta(5);        % Bias parameter
alpha_v = theta(6);     % Alpha_v parameter
phi = theta(7);         % Phi parameter
lapse = theta(8);       % Lapse parameter
recover = theta(9);     % Recover parameter

T = [1 - recover, lapse; recover, 1 - lapse]; % Transition probability matrix for latent policy states
llh = 0;
nA = 2; % Number of actions

sessions = fieldnames(data)';
counter = 0; 

% Iterate over sessions
for s = sessions
    this_data = data.(s{1});

    Q = ones(1, nA) / nA;   % Initial action values
    p = [(1+lapse-recover)/2 (1-lapse+recover)/2]; % Probability that the latent state == 0 and 1 in the previous trial
    err = 0;

    % Iterate over trials
    for t = 1:size(this_data, 2)
        if (isnan(this_data(t).rewardL) && isnan(this_data(t).rewardR)) % Skip trials with no feedback
            continue;
        elseif ~isnan(this_data(t).rewardL) % Chose L
            choice = 1;
            r = this_data(t).rewardL;
        else
            choice = 2;
            r = this_data(t).rewardR;
        end

        counter = counter + 1;

        % Calculate choice probabilities using softmax
        b(2) = 1 / (1 + exp(beta * (Q(1) - Q(2) + bias)));
        b(1) = 1 - b(2);
           
        lt = log((1/nA) * p(1) + b(choice) * p(2)); % Log-likelihood contribution of the trial
        llh = llh + lt; % Accumulate log-likelihood

        % Update probability for latent state == 0 and 1 in the current trial using Bayes' rule
        p = ((1/nA) * p(1) * T(:, 1) + b(choice) * p(2) * T(:, 2)) / exp(lt);

        latent(counter,:) = [p(2) b(choice)];

        % Calculate prediction error and update the learning rate for value updates
        v = abs(r - Q(choice)) - err;
        err = err + alpha_v * v;

        % Update alpha- based on prediction error
        if r - Q(choice) < 0
            alpha(1) = phi * (v + theta(1)) + (1 - phi) * alpha(1);
            alpha(1) = max(alpha(1), 0);
        end

        % Update Q values based on reward prediction error
        Q(choice) = Q(choice) + alpha(r + 1) * (r - Q(choice)) * (1 - err);

        % Forget the value of the unchosen action
        Q(3 - choice) = forget * Q(3 - choice);
    end
end

nllh = -llh; % Return the negative log-likelihood

end
