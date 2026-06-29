# Project Description

For $u \in [-L,L] \times [t_0, \infty)$, we consider the following 1D Cahn-Hilliard equation (CH):
$$u_t = -\nabla^2 ( \nabla^2 u + \mu u - u^3), \qquad u(x, t_0) = u_0(x), \qquad u(-L, t) = u(L, t),$$    \
where $\mu$ is a parameter slowly varying with time and $t_0 < 0$. This is derived from the following 2D Swift-Hohenberg equation (SH):
$$w_t = -(1+ \nabla^2)^2w + \mu w - w^3, \qquad w(x, t_0) = w_0(x), \qquad w(-L, t) = w(L, t).$$    \

We simulate these with spectral methods and multiple timesteppers, predominately in MATLAB but with occassional Julia files to ensure higher precision.

# How to run

## Running locally
1. To ensure all of your `.m` MATLAB files are on path, edit `startup.m` to include YOUR project root.
2. Next, ensure you have the following: 
    - MATLAB toolboxes: Image Processing, Symbolic Math
    - Julia packages: MultiFloats, GenericFFT, AbstractFFTs, LinearAlgebra, Statistics, Plots, Printf, Colors, Distributions, Measures, ColorSchemes, LaTeXStrings


## Repository structure

To navigate the repository, note the following layout:    \
.    \
├── $$\color{red}\text{Equation, e.g. Cahn-Hilliard or Swift-Hohenberg}$$    \
│   ├── $$\color{red}\text{Subfolder for 1D or 2D}$$    \
│   │   ├── 0_simulations    \
│   │   │   ├── $$\color{blue}\text{YOUR SIMULATIONS GO HERE}$$    \
│   │   │   └── $$\color{red}\text{NOTE: file formats } .asv \text{ and } .mp4 \text{ are UNTRACKED by Git}$$    \
│   │   ├── analysis    \
│   │   │   ├── $$\color{red}\text{THESE FILES ARE USED FOR ANALYSIS}$$    \
│   │   │   └── $$\color{red}\text{Plots various quantities to better understand the system}$$    \
│   │   ├── Julia    \
│   │   │   └── $$\color{red}\text{Julia files, ALSO used for analysis}$$    \
│   │   └── timesteppers    \
│   │   \t    ├── <span style="color: #FF0000;">BASIC TIMESTEPPERS</span>    \
│   │   \t    └── <span style="color: #FF0000;">These show juse one run, with various quantities available to plot from Simulation_Monitors</span>    \
│   ├── <span style="color: #FF0000;">2D Equation</span>    \
│   │   └──  0_simulations    \
│   │   \t   └── <span style="color: #FF0000;">Mirrors 1D Equation</span>    \
│   └── Simulation_Monitors    \
│   \t    └── <span style="color: #FF0000;">Various relevant quantities, e.g. norms and time scale measurements</span>    \
├── post_processing <span style="color: #FF0000;">Used for converting file formats, e.g `.asv` to `.mp4`</span>    \
├── README.md <span style="color: #FF0000;">YOU ARE HERE</span>    \
├── startup.m    \
└── Utils    \
\t    └── <span style="color: #FF0000;">ACTUAL TIMESTEP SOLVERS</span>    \


Additionally, note that certain files contain several different ***types of plots*** all trying to get a better idea of certain structures happening "under the hood", like for example, freezeout analysis plots a number of different quantities like domain wall density, compares those quantities for theoretical and computational, etc.

To run these files, simply read the comment beforehand that describes these different ***types***, and simply change the corresponding value to plot the ***plot type*** you're interested in. Additionally, right below the ***plot type*** description will be a list of parameters, which also should be changed as desired.
