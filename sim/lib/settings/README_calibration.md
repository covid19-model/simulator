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
python calibrate.py --seed "tu0" --country "GER" --area "TU" --rollouts 48 --ninit 10 --niters 100
```

* Kaiserslautern 
```
python calibrate.py --seed "kl0" --country "GER" --area "KL" --rollouts 48 --ninit 10 --niters 100
```

* Ruedesheim
```
python calibrate.py --seed "rh0" --country "GER" --area "RH" --rollouts 48 --ninit 10 --niters 100
```

* Tirschenreuth 
```
python calibrate.py --seed "tr0" --country "GER" --area "TR" --rollouts 48 --ninit 10 --niters 100
```

## Switzerland

* Lausanne (Vaud)
```
python calibrate.py --seed "vd0" --country "CH" --area "VD" --rollouts 48 --ninit 10 --niters 100
```

* Bern
```
python calibrate.py --seed "be0" --country "CH" --area "BE" --rollouts 48 --ninit 10 --niters 100
```

* Locarno (Ticino)
```
python calibrate.py --seed "ti0" --country "CH" --area "TI" --rollouts 48 --ninit 10 --niters 100
```

* Canton Jura 
```
python calibrate.py --seed "ju0" --country "CH" --area "JU" --rollouts 48 --ninit 10 --niters 100
```