# dynamic_noise_estimation
 Public repository for the dynamic noise estimation method paper.

## Citation
 Li, J. J., Shi, C., Li, L., & Collins, A. G. (2023). Dynamic noise estimation: A generalized method for modeling noise fluctuations in decision-making. bioRxiv, 2023-06.

## File structure

### Simulations (Probabilistic_Reversal/)
- code/simulate_lapses.m: analysis code for simulating data with lapses of attention and comparing the static and dynamic models (Fig 2)
- code/validate_models.m: validation analysis for both models against data simulated by the dynamic model (Fig 3)


### Empirical data ([task_name]/data/)
- Dynamic_Foraging (Grossman et al., 2022)
    - Public data repository: https://datadryad.org/stash/dataset/doi:10.5061/dryad.cz8w9gj4s
- IGT (Steingroever et al., 2015)
    - Public data repository: https://osf.io/8t7rm/
    - payoff_lookup.csv, payoff_schedule_1.csv, payoff_schedule_2.csv, payoff_schedule_3.csv were created by Jing-Jing Li according to the payoff schedules described in the paper
- RLWM (Collins 2018)
    - Public data repository: https://osf.io/5gbr3/ 
- 2-step (Nussenbaum et al., 2020)
    - Public data repository: https://osf.io/we89v/
    - data_processing_scripts/concatenate_mats.m was created by Jing-Jing Li to reorganize the data structure

### Modeling scripts for empirical datasets ([task_name]/code/)
- static_model_llh.m and dynamic_model_llh.m: functions to compute the negative log likelihoods of data given the static and dynamic model parameters
- static_model.m and dynamic_model.m: functions to generate data using the static and dynamic models
- fit_models.m: model fitting code for the static and dynamic models (Fig 4)
- compare_params.m: compares the same parameters between the best-fit values of the static and dynamic models (Fig 5, Fig 6, Fig A11)
- identify_models.m: model identification analysis for both the static and dynamic models (Fig A8)
- recover_params.m: generate and recover analysis for parameters of the dynamic model (Fig A10)
- recover_latent_probs.m: recovery analysis of p(Engaged) trajectory (Fig A10)
- validate_models.m: validation analysis against behavior for both models (Fig A9)

### Plots ([task_name]/plots/)
- Output plots for all figures in .png and .svg

