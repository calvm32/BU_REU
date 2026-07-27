# Project Description

For $u \in [-L,L] \times [t_0, \infty)$, we consider the following 1D Cahn-Hilliard equation (CH):
$$u_t = -\nabla^2 ( \nabla^2 u + \mu u - u^3), \qquad u(x, t_0) = u_0(x), \qquad u(-L, t) = u(L, t),$$    \
where $\mu$ is a parameter slowly varying with time and $t_0 < 0$. This is derived from the following 2D Swift-Hohenberg equation (SH):
$$w_t = -(1+ \nabla^2)^2w + \mu w - w^3, \qquad w(x, t_0) = w_0(x), \qquad w(-L, t) = w(L, t).$$    

We simulate these with spectral methods and multiple time steppers, predominately in MATLAB but with occasional Julia files to ensure higher precision.

---
# How to run

## Running locally
1. To ensure all of your `.m` MATLAB files are on path, edit `startup.m` to include YOUR project root.
2. Next, ensure you have the following: 
    - MATLAB toolboxes: Image Processing, Symbolic Math
    - Julia packages: MultiFloats, GenericFFT, AbstractFFTs, LinearAlgebra, Statistics, Plots, Printf, Colors, Distributions, Measures, ColorSchemes, LaTeXStrings


## Repository structure

To navigate the repository, note the following layout:    \
.    \
├── Equation $$\color{red}\text{e.g. Cahn-Hilliard or Swift-Hohenberg}$$    \
│   &emsp;&emsp;├── Equation1D $$\color{red}\text{Subfolder for 1D or 2D}$$    \
│   &emsp;&emsp;│   &emsp;&emsp;├── 0_simulations    \
│   &emsp;&emsp;│   &emsp;&emsp;│   &emsp;&emsp;├── $$\color{blue}\text{YOUR SIMULATIONS GO HERE}$$    \
│   &emsp;&emsp;│   &emsp;&emsp;│   &emsp;&emsp;└── $$\color{red}\text{NOTE: file formats } .asv \text{ and } .mp4 \text{ are UNTRACKED by Git}$$    \
│   &emsp;&emsp;│   &emsp;&emsp;├── analysis    \
│   &emsp;&emsp;│   &emsp;&emsp;│   &emsp;&emsp;├── $$\color{red}\text{THESE FILES ARE USED FOR ANALYSIS}$$    \
│   &emsp;&emsp;│   &emsp;&emsp;│   &emsp;&emsp;└── $$\color{red}\text{Plots various quantities to better understand the system}$$    \
│   &emsp;&emsp;│   &emsp;&emsp;├── Julia    \
│   &emsp;&emsp;│   &emsp;&emsp;│   &emsp;&emsp;└── $$\color{red}\text{Julia files, ALSO used for analysis}$$    \
│   &emsp;&emsp;│   &emsp;&emsp;└── timesteppers    \
│   &emsp;&emsp;│   &emsp;&emsp;├── $$\color{red}\text{BASIC TIMESTEPPERS}$$    \
│   &emsp;&emsp;│   &emsp;&emsp;└── $$\color{red}\text{These show just one run, with various quantities available to plot from Simulation Monitors}$$    \
│   &emsp;&emsp;├── Equation2D    \
│   &emsp;&emsp;│   &emsp;&emsp;└── $$\color{red}\text{Mirrors structure of Equation1D}$$    \
│   &emsp;&emsp;└── Simulation_Monitors    \
│   &emsp;&emsp;&nbsp;&nbsp;    &emsp;&emsp;└── $$\color{red}\text{Various relevant quantities, e.g. norms and time scale measurements}$$    \
├── post_processing $$\color{red}\text{Used for converting file formats, e.g} .asv \text{ to } .mp4$$    \
├── README.md $$\color{red}\text{YOU ARE HERE}$$    \
├── startup.m    \
└── Utils    \
&emsp;&emsp;&nbsp;&nbsp;└── $$\color{red}\text{ACTUAL TIMESTEP SOLVERS}$$

## Library structure
The `Utils` folder provides several generic time steppers that can be used used to solve any PDE of the form
$$u_t = Lu + N(u).$$
For now, we provide ETDRK4 and Crank-Nicolson, with an addition gradient flow solver speicifically for the fixed parameter Cahn-Hilliard equation. To see a simple example on how these solvers can be used, check out [`solve_CH1D_ETDRK4.m`](https://github.com/calvm32/BU_REU/blob/main/Cahn_Hilliard/CH1D/timesteppers/solve_CH1D_ETDRK4.m).

`SimulationMonitors` is an abstract class allowing users to process the solution at intermediate time steps for logging and plotting. The initialization is called by the solver populating it with the initial data and time grid. The update method is called every time step where the current time and solution are given to the monitor. The solver calls the finalize method at the end, useful for closing videos or figures, for instance. In the example above, [`DensityL2FreeEnergyMonitor`](https://github.com/calvm32/BU_REU/blob/main/Cahn_Hilliard/Simulation_Monitors/DensityL2FreeEnergyMonitor.m) is used to log and plot the solution, the theoritical and compute density, the L2 norm, and the free energy over time. It then saves this plot as a video.

---
# Replicating transcript code

Going section-by-section, we now describe how to replicate figures seen throughout the transcript.

## "Introduction"

- To recreate the diagram that shows bifurcations for two different values of mass $$m$$, using the method of continuation approximation, simply run `Cahn-Hilliard/CH1D/analysis/continuation_approx.m`

## "Preliminary Results"

- To recreate the diagram that compares computed and theoretical values of fastest-growing wave mode at freeze-out time and domain wall density averaged over a certain number of runs as the parameter $$\epsilon$$ varies, run `Cahn-Hilliard/CH1D/analysis/KZ_freezeout_analysis.m`. To see what these metrics look like for a single run in time, run `KZ_density_analysis.m` in the same folder

## "Slow Passage Through Bifurcations"

- The large scale survey for the small domain was obtained using `Cahn-Hilliard/CH1D/analysis/epsilon_mu0_parallel_survey.m` The data itself can be found in [this Dropbox folder](https://www.dropbox.com/scl/fo/mwq6qvvx34nkxqlz062es/AAw49QGLOELpVJZQvfll7ts?rlkey=sx62mgkswoutoome047hcxf7r&dl=0). The live script to analyze the survey data may be found in `Cahn_Hilliard/CH1D/analysis/galerkin_projections/galerking_approx_contours.mlx`. The large domain data was generated using `Cahn_Hilliard/CH1D/analysis/script_ramped_ch.m` and the data can be found [here](https://www.dropbox.com/scl/fi/lquhduipfo6x1gevjvk79/large_domain_results.mat?rlkey=jsilmx5nrjogs32vp2c91n5v0&dl=0).
