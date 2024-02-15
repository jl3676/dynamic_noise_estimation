function latent = dynamic_model_latent(theta, data)
% Computes the latent probability trajectory for a dynamic model with given
% fitted parameters.
%
% Inputs:
%   - theta(1): alpha - learning rate
%   - theta(2): stickiness - stickiness parameter
%   - theta(3): lapse - probability of lapsing into the random state
%   - theta(4): recover - probability of returning to an engaged state after lapsing
%   - data: Matrix containing the choices and rewards from the data
%         Column 3: Choices (1 or 2)
%         Column 5: Rewards (0 or 1) 
%         Other columns are irrelevant here
%
% Output:
%   - latent: matrix containing estimated latent state and choice
%   probability trajectories
%
% Author: Jing-Jing Li (jl3676@berkeley.edu)
% Last Modified: 5/28/2023

alpha = theta(1);              % learning rate - determines the weight of new information during learning
beta = 8;                      % Inverse softmax temperature parameter, fixed 
stick = theta(2);              % Stickiness parameter - controls the tendency to repeat the previous action
epsilon = 0;                   % Epsilon parameter - not used in the model
lapse = theta(3);              % Probability of lapsing while in an engaged state 
recover = theta(4);            % Probability of returning to an engaged state after lapsing - the probability of recovering from a lapse

T = [1 - recover, lapse; recover, 1 - lapse];  % Transition probability matrix for latent states
nA = 2;                                        % Number of actions

choices = data(:, 3);    % Get choices from data
rewards = data(:, 5);    % Get rewards from data

Q = ones(1, nA) / nA;    % Initial action values
side = 0;                % Side to stick to (1 = A1, -1 = A2)
p = [lapse, 1 - lapse];  % Initial latent state occupancy probabilities

latent = zeros(length(choices), 2); % Initialize matrix to store latent information

% Iterate over trials
for k = 1:length(choices)
    choice = choices(k);   % Current choice
    r = rewards(k);        % Current reward

    b(2) = epsilon / 2 + (1 - epsilon) / (1 + exp(beta * (Q(1) - Q(2) + stick * side)));
    b(1) = 1 - b(2);

    lt = log(p(1) / nA + b(choice) * p(2));   % Log-likelihood of trial t

    % Update probability for latent state == 0 and 1 in the current trial
    p = (p(1) * T(:, 1) / nA + b(choice) * p(2) * T(:, 2)) / exp(lt);

    latent(k,:) = [p(2), b(choice)];

    % Update Q values
    Q(choice) = Q(choice) + alpha * (r - Q(choice));

    if choice == 1
        side = 1;
    else
        side = -1;
    end
end
end
