function nllh = static_model_llh(theta, data)
% Computes the full negative log-likelihood of data given parameters (theta).
%
% Input:
%   - theta: Model parameters (1x8 vector)
%     - theta(1:2): Alpha parameters (alpha- and alpha+)
%     - theta(3): Beta parameter
%     - theta(4): Forget parameter
%     - theta(5): Bias parameter
%     - theta(6): Alpha_v parameter
%     - theta(7): Phi parameter
%     - theta(8): Epsilon parameter
%   - data: Struct containing behavioral data
%
% Output:
%   - nllh: Negative log-likelihood of the data given the parameters
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
epsilon = theta(8);     % Epsilon parameter   

llh = 0;
nA = 2; % Number of actions

sessions = fieldnames(data)';

% Iterate over sessions
for s = sessions
    this_data = data.(s{1});

    Q = ones(1, nA) / nA;   % Initial action values
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

        % Calculate choice probabilities using softmax
        b(2) = 1 / (1 + exp(beta * (Q(1) - Q(2) + bias)));
        b(1) = 1 - b(2);
        b = epsilon / nA + (1 - epsilon) * b;
           
        lt = log(b(choice)); % Log-likelihood contribution of the trial
        llh = llh + lt; % Accumulate log-likelihood

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
