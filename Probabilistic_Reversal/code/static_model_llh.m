function nllh = static_model_llh(theta, data)
% Computes the negative log-likelihood of data given parameters (theta) for a static model.
%
% Inputs:
%   - theta(1): alpha - learning rate
%   - theta(2): stickiness - stickiness parameter
%   - theta(3): epsilon - uniform noise parameter
%   - data: Matrix containing the choices and rewards from the data
%         Column 3: Choices (1 or 2)
%         Column 5: Rewards (0 or 1)
%         Other columns are irrelevant here
%
% Output:
%   - nllh: Negative log-likelihood of the data given the model parameters
%
% Author: Jing-Jing Li (jl3676@berkeley.edu)
% Last Modified: 5/28/2023

alpha = theta(1);              % learning rate - determines the weight of new information during learning
beta = 8;                      % Inverse softmax temperature parameter, fixed 
stick = theta(2);              % Stickiness parameter - controls the tendency to repeat the previous action
epsilon = theta(3);            % Epsilon parameter - uniform level of noise
lapse = 0;                     % Probability of lapsing in an engaged state - not used in the static model
recover = 1;                   % Probability of returning to an engaged state after lapsing - not used in the static model

T = [1 - recover, lapse; recover, 1 - lapse];  % Transition probability matrix for latent state (not used in the static model)
nA = 2;                                       % Number of actions

choices = data(:, 3);    % Get choices from data 
rewards = data(:, 5);    % Get rewards from data

Q = ones(1, nA) / nA;   % Initial action values
side = 0;              % Side to stick to (1 = A1, -1 = A2)
p = [lapse, 1 - lapse];  % Probability that the latent state == 0 and 1 in the previous trial (not used in the static model)
llh = 0;               % Cumulative log-likelihood

% Iterate over trials
for k = 1:length(choices)
    choice = choices(k);   % Current choice
    r = rewards(k);        % Current reward

    b(2) = epsilon / 2 + (1 - epsilon) / (1 + exp(beta * (Q(1) - Q(2) + stick * side)));
    b(1) = 1 - b(2);

    lt = log(p(1) / nA + b(choice) * p(2));   % Log-likelihood of trial t
    llh = llh + lt;                           % Increment total log-likelihood

    % Update probability for latent state == 0 and 1 in the current trial
    p = (p(1) * T(:, 1) / nA + b(choice) * p(2) * T(:, 2)) / exp(lt);

    % Update Q values
    Q(choice) = Q(choice) + alpha * (r - Q(choice));

    if choice == 1
        side = 1;
    else
        side = -1;
    end
end

nllh = -llh;  % Return the negative log-likelihood

end
