# Listening to Bluetooth Beacons for Epidemic Risk Mitigation

This repository contains the code for the experiments complementing our white paper on [PanCast](https://arxiv.org/pdf/2011.08069.pdf).
It builds on the spatio-temporal epidemic model introduced in our previous [paper](https://arxiv.org/abs/2004.07641).

## Project description

During the ongoing COVID-19 pandemic, there have been burgeoning efforts to develop and deploy smartphone apps 
to expedite contact tracing and risk notification. 
Unfortunately, the success of these apps has been limited, partly owing to poor interoperability with 
manual contact tracing, low adoption rates, and a societally sensitive trade-off between utility and privacy.
In this work, we introduce a new privacy-preserving and inclusive system for epidemic risk assessment and notification that
aims to address the above limitations.
Rather than capturing pairwise encounters between smartphones as done by existing apps, our system captures
encounters between inexpensive, zero-maintenance, small devices carried by users, and beacons placed in 
strategic locations where infection clusters are most likely to originate.
The epidemiological simulations using the agent-based model contained in this repository demonstrate 
several beneficial properties of our system. 
By achieving bidirectional interoperability with manual contact tracing, our system can help control disease spread 
already at low adoption. By utilizing the location and environmental information provided by the beacons, 
our system can provide significantly higher sensitivity and specificity than existing app-based systems.
In addition, our simulations also suggest that it is sufficient to deploy beacons in a small fraction of 
strategic locations for our system to achieve high utility.


<p align="center">
<img width="25%" src="./img/beacon-environment-GER-TU-beta_dispersion=2.0.pdf">
<img width="25%" src="./beacon-environment-GER-TU-beta_dispersion=10.0.pdf">
</p>

<p align="center">
<img width="25%" src="./img/reduction-cumu_infected-GER-TU-beacon_mode=visit_freq.pdf">
<img width="25%" src="./reduction-hosp-GER-TU-beacon_mode=visit_freq.pdf">
</p>

<p align="center">
<img width="25%" src="./img/relative-cumu_infected-heatmap-pancast.pdf">
<img width="25%" src="./relative-hosp-heatmap-pancast.pdf">
</p>


## Implementation of PanCast into the epidemiological simulator
Our simulations include manual contact tracing as well as digital contact tracing using either PanCast or 
smartphone-based pairwise encounter-based contact tracing systems (SPECTs).  

Independent of the tracing method, whenever a contact person of a diagnosed individual  gets  successfully  traced, 
the  infection  risk  is  estimated  by  taking  into  account  the  duration  of the contact and possibly 
environmental factors (only for PanCast). Individuals with infection risk above a certain threshold are quarantined 
for two weeks, get tested within the next 24 hours, and receive the out come of the test 48 hours later. We choose the infection risk threshold to correspond to a 15 minute contact with asymptomatic individual in the model, 
which is in accordance with SPECTs currently employed in Germany, Switzerland, the United Kingdom, France, and Australia.
All simulations using PanCast or SPECTs also implement manual tracing by assuming that a certain fraction of visitors leave their contact details at social, office and education sites, 
so that they can be reliably contacted, e.g., via phone.  Upon receiving a positive test result, 
we assume that every individual participates in a manual contact-tracing interview independent of their participation 
in digital tracing.  In the tracing interview, we assume a person only remembers a certain fraction of 
their visit history of the past 14 days. 

For PanCast we assume that a proportion of the population has adopted the system and is carrying dongles. We place beacons at a different proportions of the sites.  This can be done at random or strategically, 
by taking into account quantities related to the site-specific probability of infections, e.g., by ranking the 
sites according to their integrated visit time.  Whenever a person carrying a dongle gets tested positive, all 
contacts at sites with beacons who also carry dongles get traced.  In addition, the information can be used to 
trigger manual tracing action at all sites registered by the dongle of the positive-tested individual
(i.e., PanCast supports manual tracing).  Likewise, when a person who tested positive does not carry a dongle but 
participates in a manual contact interview and recalls a visit to a site with a beacon,  all individuals carrying 
dongles at this site can be traced (i.e., manual tracing supports PanCast).  In our simulations, we assume that 
individuals receive PanCast’s risk broadcast instantaneously.  In practice, this assumption maybe violated, but
PanCast's protocol for risk dissemination ensures that the time it takes
for dongles to receive risk information stays within acceptable limits.



## Dependencies

This project uses Python 3. To create a virtual environment and install the project dependencies, you can run the following commands:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```


## Reproducibility

The experimental results can be reproduced by running the following commands from the `sim` directory:

```python
python3 sim-beacon-environment.py --country GER --area TU
python3 sim-beacon-manual-baseline.py --country GER --area TU
python3 sim-beacon-manual-tracing.py --country GER --area TU
python3 sim-beacon-sparse-locations.py --country GER --area TU
```
Every script runs several experiments with different settings. Note that depending on the settings simulations can take 
up to 10 hours and require up to 1TB of RAM.
The scripts generate result summary files in the directory `sim/condensed_summaries`. 
The plots can be generated from these files using the iPython notebook `sim-plot-beacon.ipynb`.



## Code Organization

The `sim/` directory contains the entire project code, where all simulator-specific code is situated inside `sim/lib/`. The simulator operates using the following main modules and rough purpose descriptions:

| `sim/lib/`                                                   | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [dynamics.py](sim/lib/dynamics.py)                           | Simulator core; defines a **DiseaseModel** object for simulating the spread of the epidemic. |
| [measures.py](sim/lib/measures.py)                           | Containement measures; defines a **Measure** object for implementing intervention policies. |
| [mobilitysim.py](sim/lib/mobilitysim.py)                     | Mobility patterns; defines a **MobilitySimulator** object for mobility traces of individuals. |
| [distributions.py](sim/lib/distributions.py)                 | Epidemiology; contains COVID-19 constants and distribution sampling functions. |
| [calibrationFunctions.py](sim/lib/calibrationFunctions.py); [calibrationParser.py](sim/lib/calibrationParser.py); [calibrationSettings.py](sim/lib/calibrationSettings.py) | Parameter Estimation; defines **Bayesian Optimization pipeline** for estimating region-specific exposure parameters |
| [experiment.py](sim/lib/experiment.py)                       | Analysis; defines an **Experiment** object for structured analysis and plotting of scenarios |
| [plot.py](sim/lib/plot.py)                                   | Plotting; defines a **Plotter** object for generating plots. |

The `sim/` directory itself containts several scripts of the form `sim-*.py`, which run experiments and simulations as reported in our [paper](https://arxiv.org/abs/2004.07641) parallelized across CPUs. To execute a specific experiment script for an already fitted region, simply execute e.g. `sim-*.py --country GER --area TU` (here: Tübingen, Germany). 

To apply the entire framework for a new region in experiments as defined in `sim-*.py`, the following two major steps need to be performed in order beforehand and **only once**:

1. Create a new mobility file using the `sim/town-generator.ipynb` notebook, which fixes the region-specific metadata for simulation and stores it inside `sim/lib/mobility`. A downsampling factor for population and sites can be used to speed-up the simulation initially or for parameter estimation. The directory already contains all mobility metadata files we used in our simulations. 
2. Estimate the region-specific exposure parameters by executing `calibrate.py`. Before doing so, add the mobility file path, a region-specific code (e.g. `GER` and `TU`), and other details from above to `sim/lib/calibrationSettings.py`, following the structure for the existing regions. Hyperparameters for parameter estimation can be set using the command line arguments specified in `sim/lib/calibrationParser.py`, for example as listed in `sim/lib/settings/estimation.md`. The estimated parameters are saved and logged inside `sim/logs`. Depending on the size of the model, this step represents the major computational (yet one-time) cost. 

Thus, the region metadata file in `sim/lib/mobility` and the parameter estimation log in `sim/logs` represents the fixed state of a corresponding region, starting point for simulating scenarios and running counterfactual analyses. 

The results of a set of simulations such as `sim-*.py` are stored inside `sim/summaries/` and can be visualized using `sim/sim-plot.ipynb`. 


## Citation

If you use parts of the code in this repository for your own research purposes, please consider citing:

    @article{lorch2020quantifying,
        title={Quantifying the Effects of Contact Tracing, Testing, and Containment Measures in the Presence of Infection Hotspots},
        author={Lars Lorch and Heiner Kremer and William Trouleau and Stratis Tsirtsis and Aron Szanto and Bernhard Sch\"olkopf and Manuel Gomez-Rodriguez},
        journal={arXiv preprint arXiv:2004.07641},
        year={2020}
    }
    
or for PanCast specific aspects:

    @article{barthe2020pancast,
      title={PanCast: Listening to Bluetooth Beacons for Epidemic Risk Mitigation}, 
      author={Gilles Barthe and Roberta De Viti and Peter Druschel and Deepak Garg and Manuel Gomez-Rodriguez and Pierfrancesco Ingo and Matthew Lentz and Aastha Mehta and Bernhard Schölkopf},
      journal={arXiv preprint arXiv:2011.08069}
      year={2020},
    }
