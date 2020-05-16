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
python calibrate.py --seed "tu0" --country "GER" --area "TU" --start "2020-03-11" --end "2020-03-26" --mob "lib/mobility/Tubingen_settings_10.pk" --rollouts 80 --ninit 30 --niters 300
```

* Kaiserslautern 
```
python calibrate.py --seed "kl0" --country "GER" --area "KL" --start "2020-03-11" --end "2020-03-26" --mob "lib/mobility/Kaiserslautern_settings_10.pk" --rollouts 80 --ninit 30 --niters 300
```

* Tirschenreuth 
```
python calibrate.py --seed "tr0" --country "GER" --area "TR" --start "2020-03-11" --end "2020-03-26" --mob "lib/mobility/Tirschenreuth_settings_10.pk" --rollouts 80 --ninit 30 --niters 300
```

* Ruedesheim
```
python calibrate.py --seed "rh0" --country "GER" --area "RH" --start "2020-03-11" --end "2020-03-26" --mob "lib/mobility/Ruedesheim_settings_10.pk" --rollouts 80 --ninit 30 --niters 300
```

## Switzerland
* Lausanne 
```
python calibrate.py --seed "vd0" --country "CH" --area "VD" --start "2020-03-04" --end "2020-03-18" --mob "lib/mobility/Lausanne_settings_10.pk" --rollouts 80 --ninit 30 --niters 300
```

* Lucerne
```
python calibrate.py --seed "lu0" --country "CH" --area "LU" --start "2020-03-04" --end "2020-03-18" --mob "lib/mobility/Lucerne_settings_5.pk" --rollouts 80 --ninit 30 --niters 300
```

* Locarno 
```
python calibrate.py --seed "ti0" --country "CH" --area "TI" --start "2020-03-04" --end "2020-03-18" --mob "lib/mobility/Locarno_settings_3.pk" --rollouts 80 --ninit 30 --niters 300
```

* Canton Schwyz 
```
python calibrate.py --seed "sz0" --country "CH" --area "SZ" --start "2020-03-04" --end "2020-03-18" --mob "lib/mobility/Schwyz_settings_20.pk" --rollouts 80 --ninit 30 --niters 300
```