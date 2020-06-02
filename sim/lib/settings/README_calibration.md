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
python calibrate.py --seed "tu0" --country "GER" --area "TU" --rollouts 50 --ninit 30 --niters 200
```

* Kaiserslautern 
```
python calibrate.py --seed "kl0" --country "GER" --area "KL" --rollouts 50 --ninit 30 --niters 200
```

* Ruedesheim
```
python calibrate.py --seed "rh0" --country "GER" --area "RH" --rollouts 50 --ninit 30 --niters 200
```

* Tirschenreuth 
```
python calibrate.py --seed "tr0" --country "GER" --area "TR" --rollouts 50 --ninit 30 --niters 200
```

## Switzerland

* Lausanne (Vaud)
```
python calibrate.py --seed "vd0" --country "CH" --area "VD" --rollouts 50 --ninit 30 --niters 200
```

* Bern
```
python calibrate.py --seed "be0" --country "CH" --area "BE" --rollouts 50 --ninit 30 --niters 200
```

* Locarno (Ticino)
```
python calibrate.py --seed "ti0" --country "CH" --area "TI" --rollouts 50 --ninit 30 --niters 200
```

* Canton Jura 
```
python calibrate.py --seed "ju0" --country "CH" --area "JU" --rollouts 50 --ninit 30 --niters 200
```