# A Spatiotemporal Epidemic Model to Quantify The Effects of Testing, Contact Tracing and Containment

This repository contains scripts and notebooks to run the sampling algorithm of a high resolution spatiotemporal epidemic model, which can be used to predict the spread of COVID-19 under different testing & tracing strategies, social distancing measures and business restrictions in an arbitrary city/town. Details about the relevant theory and methods can be found in the [paper](https://arxiv.org/abs/2004.07641).

## Project description

We introduce a modeling framework for studying epidemics that is specifically designed to make use of fine-grained spatiotemporal data. Motivated by the availability of data from contact tracing technologies and the current COVID-19 outbreak, our model uses marked temporal point processes to represent individual mobility patterns and the course of the disease for each individual in a population.

The sampling algorithm provided in this repository can be used to predict the spread of COVID-19 under different testing & tracing strategies, social distancing measures and business restrictions, given location or contact histories of individuals. Moreover, it gives a detailed representation of the disease's effect on each individual through time, distinguishing between several known states like asymptomatic, presymptomatic, symptomatic or recovered. 

<p align="center">
<img width="33%" src="./img/run0_opt_002.png">
</p>

Additionally, an inference script based on Bayesian Optimization allows the estimation of the risk of exposure of each individual at sites, the percentage of symptomatic individuals, and the difference in transmission rate between asymptomatic and symptomatic individuals from historical longitudinal testing data.

<p align="center">
<img width="33%" src="./img/fig_exp2.png">
</p>

The notebooks in this repository are focused on real COVID-19 data and mobility patterns from Tübingen, a town in the Southwest of Germany, but can be easily parameterized and used for generating realistic mobility patterns and simulating the spread of a disease for any given city/town.

<p align="center">
<img width="33%" src="./img/population_distribution.png">
<img width="33%" src="./img/site_distribution.png">
</p>

## Dependencies

All the experiments were executed using Python 3. In order to create a virtual environment and install the project dependencies you can run the following commands:

```bash
python3 -m venv env
source env/bin/activate
pip install -r sim/requirements.txt
```

Otherwise, you can manually install the following packages:

* numpy
* pandas
* numba
* networkx
* scipy
* interlap
* seaborn
* bayesian-optimization
* joblib
* geopy
* pathos
* requests
* folium

## Code organization

In the following tables, short descriptions of notebooks and main scripts are given. The notebooks are self-explanatory and execution details can be found within them.

| Notebook              | Description                                                   |
|-----------------------|---------------------------------------------------------------|
| [town-generator.ipynb](sim/town-generator.ipynb)  | Generates population, site and mobility data for a given town. |
| [exe-inference.ipynb](sim/exe-inference.ipynb)   | Performs Bayesian optimization to infer the model parameters based on mobility patterns and reported cases of infection. |
| [experiments.ipynb](sim/experiments.ipynb)     | Performs experiments about the spread of the disease under testing, contact tracing and/or containment measures. |

| Script                | Description                                                   |
|-----------------------|---------------------------------------------------------------|
| [distributions.py](sim/lib/distributions.py) | Contains COVID-19 constants and distribution sampling functions. |
| [town_data.py](sim/lib/town_data.py)  | Contains functions for population and site generation. |
| [data.py](sim/lib/data.py)   | Contains functions for COVID-19 data collection. |
| [mobilitysim.py](sim/lib/mobilitysim.py) | Produces a **MobilitySimulator** object for generating mobility traces. |
| [dynamics.py](sim/lib/dynamics.py) | Produces a **DiseaseModel** object for simulating the spread of the disease. |
| [parallel.py](sim/lib/parallel.py) | Contains functions used for simulations on parallel threads. |
| [measures.py](sim/lib/measures.py) | Produces a **Measure** object for implementing intervention policies. |
| [inference.py](sim/lib/inference.py) | Contains functions used for Bayesian optimization. |
| [plot.py](sim/lib/plot.py) | Produces a **Plotter** object for generating plots. |
| [town_maps.py](sim/lib/plot.py) | Produces a **MapIllustrator** object for generating interactive maps. |

## Call for Contributions

This project is a work in progress. We appreciate any help to fix and/or improve the code base. Feel free to drop us an issue to discuss bugs, enhancements, feature requests, etc. We are happy to accept pull requests.

## Citation

If you use parts of the code in this repository for your own research purposes, please consider citing:

    @article{lorch2020spatiotemporal,
        title={A Spatiotemporal Epidemic Model to Quantify the Effects of Contact Tracing, Testing, and Containment},
        author={Lars Lorch and William Trouleau and Stratis Tsirtsis and Aron Szanto and Bernhard Sch\"{o}lkopf and Manuel Gomez-Rodriguez},
        journal={arXiv preprint arXiv:2004.07641},
        year={2020}
    }
