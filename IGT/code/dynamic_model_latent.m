function latent = dynamic_model_latent(theta, data)
% Computes the full negative log likelihood of data given parameters (theta).
%
% Inputs:
%   - theta: Parameter values for the dynamic model. It is a vector with the following elements:
%     - theta(1): Learning rate (alpha)
%     - theta(2): Inverse temperature (beta)
%     - theta(3): Sensitivity parameter (sensitivity)
%     - theta(4): Decay rate (decay)
%     - theta(5): Exploration rate (phi)
%     - theta(6): Lapse rate (lapse)
%     - theta(7): Recover rate (recover)
%   - data: Experimental data. It is an N-by-3 matrix, where N is the number of trials. Each row contains the following information:
%     - data(:, 1): Choices made by the subject
%     - data(:, 2): Rewards received by the subject
%     - data(:, 3): Losses received by the subject
%
% Output:
%   - latent: latent state and choice probability trajectories 

% Extract parameter values
alpha = theta(1);     % Learning rate
beta = theta(2);      % Inverse temperature
sensitivity = theta(3);     % Sensitivity
decay = theta(4);     % Decay rate
phi = theta(5);       % Exploration rate
epsilon = 0;          % Epsilon parameter for static noise
lapse = theta(6);     % Lapse rate
recover = theta(7);   % Recover rate

% Transition probability matrix for latent attentional state
T = [1 - recover, lapse; recover, 1 - lapse];

choices = data(:, 1);    % Choices from data
gains = data(:, 2);      % Rewards from data
losses = data(:, 3);     % Losses from data

nA = 4;    % Number of available actions

explore = ones(nA, 1) / nA;
exploit = ones(nA, 1) / nA;
llh = 0;    % Cumulative log-likelihood
p = [lapse, 1 - lapse]; % Probability that the latent state == 0 and 1 in the previous trial

% Iterate over trials
for k = 1:length(choices)
    choice = choices(k);
    gain = gains(k);
    loss = losses(k);

    % Update Q values
    Q = explore + exploit;
    b = epsilon / nA + (1 - epsilon) * exp(beta * Q - logsumexp(beta * Q));
    
    lt = log(p(1) / nA + b(choice) * p(2));
    llh = llh + lt;
    
    % Update probability for latent state == 0 and 1 in the current trial
    p = p(1) * T(:, 1) / nA + b(choice) * p(2) * T(:, 2);
    p = p / exp(lt);

    exploit = decay * exploit;
    exploit(choice) = exploit(choice) + (gain ^ sensitivity - abs(loss) ^ sensitivity);
    explore = explore + alpha * (phi - explore);
    explore(choice) = 0;

    latent(k,:) = [p(2), b(choice)];
end

end
