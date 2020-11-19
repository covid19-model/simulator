***************************
README Calibration
***************************

Run

```python calibrate.py --help```

for help regarding arguments required for model calibration.

Inference runs:

## Germany
* Tubingen 
```
python calibrate.py --config_file "lib/settings/tuebingen_config.py"  --rollouts 96 --ninit 20 --niters 80
```

* Kaiserslautern 
```
python calibrate.py --config_file "lib/settings/kaiserslautern_config.py" --rollouts 96 --ninit 20 --niters 80
```

* Ruedesheim
```
python calibrate.py --config_file "lib/settings/ruedesheim_config.py" --rollouts 96 --ninit 20 --niters 80
```


## Switzerland

* Bern
```
python calibrate.py --config_file "lib/settings/bern_config.py" --rollouts 96 --ninit 20 --niters 80
```

* Locarno (Ticino)
```
python calibrate.py --config_file "lib/settings/locarno_config.py" --rollouts 96 --ninit 20 --niters 80
```

* Canton Jura 
```
python calibrate.py --config_file "lib/settings/jura_config.py" --rollouts 96 --ninit 20 --niters 80
```