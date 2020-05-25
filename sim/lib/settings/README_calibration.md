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
python calibrate.py --seed "tu0" --country "GER" --area "TU" --start "2020-03-12" --end "2020-03-28" --mob "lib/mobility/Tubingen_settings_10.pk" --rollouts 80 --ninit 30 --niters 200
```

* Kaiserslautern 
```
python calibrate.py --seed "kl0" --country "GER" --area "KL" --start "2020-03-15" --end "2020-03-28" --mob "lib/mobility/Kaiserslautern_settings_10.pk" --rollouts 80 --ninit 30 --niters 200
```

* Ruedesheim
```
python calibrate.py --seed "rh0" --country "GER" --area "RH" --start "2020-03-10" --end "2020-03-28" --mob "lib/mobility/Ruedesheim_settings_10.pk" --rollouts 80 --ninit 30 --niters 200
```

* Tirschenreuth 
```
python calibrate.py --seed "tr0" --country "GER" --area "TR" --start "2020-03-13" --end "2020-03-28" --mob "lib/mobility/Tirschenreuth_settings_10.pk" --rollouts 80 --ninit 30 --niters 200
```

## Switzerland

* Lausanne (Vaud)
```
python calibrate.py --seed "vd0" --country "CH" --area "VD" --start "2020-03-07" --end "2020-03-21" --mob "lib/mobility/Lausanne_settings_10.pk" --rollouts 80 --ninit 30 --niters 200
```

* Lucerne
```
python calibrate.py --seed "lu0" --country "CH" --area "LU" --start "2020-03-14" --end "2020-03-21" --mob "lib/mobility/Lucerne_settings_5.pk" --rollouts 80 --ninit 30 --niters 200
```

* Locarno (Ticino)
```
python calibrate.py --seed "ti0" --country "CH" --area "TI" --start "2020-03-09" --end "2020-03-21" --mob "lib/mobility/Locarno_settings_2.pk" --rollouts 80 --ninit 30 --niters 200
```

* Canton Jura 
```
python calibrate.py --seed "ju0" --country "CH" --area "JU" --start "2020-03-09" --end "2020-03-21" --mob "lib/mobility/Jura_settings_10.pk" --rollouts 80 --ninit 30 --niters 200
```