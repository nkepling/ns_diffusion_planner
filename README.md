# ns_diffusion_planner
Diffusion based planner for solving NS-MDPs

## Set up:

Pip install `ns_gym` package for non-stationary Gymnasium style MDPs. Package dependencies should set us up for this project.

```bash
python3.10 -m venv env
source env/bin/activate
pip install git+https://github.com/scope-lab-vu/ns_gym.git
```

## TODOs:

- [ ] Finish Data Gen file and generate dataset
- [ ] Set up Diffusion model to sample value maps given differenet configs fo FrozenLake obstacle configurations. 
- [ ] Set up training pipeline.
- [ ] Look at resutls for sationary value maps. 


## A Note:

I think given the stochtistic nature of Frozenlake rather than "denoising" state-action trajectories we learn to sample value "Value Map Images" conditioned on different FrozenLake obstacle configurations. Once we get the stationary value maps going we can see how well they work. 





