***************************
README Calibration
***************************

Run

```python calibrate.py --help```

for help regarding arguments required for model calibration.

Inference runs:

Germany
* Tubingen (beta in [0, 2])
```
python calibrate.py --seed "tu0" --country "GER" --area "TU" --start "2020-03-10" --end "2020-03-26" --mob "lib/mobility/Tubingen_settings_10_10_hh.pk" --downsample 10 --rollouts 80 --ninit 20 --niters 300
```

* Kaiserslautern (beta in [0, 2])
```
python calibrate.py --seed "kl0" --country "GER" --area "KL" --start "2020-03-18" --end "2020-03-26" --mob "lib/mobility/Kaiserslautern_settings_10_10_hh.pk" --downsample 10 --rollouts 80 --ninit 20 --niters 300
```

* Tirschenreuth (beta in [0, 2])
```
python calibrate.py --seed "tr0" --country "GER" --area "TR" --start "2020-03-10" --end "2020-03-26" --mob "lib/mobility/Tirschenreuth_settings_10_10_hh.pk" --downsample 10 --rollouts 80 --ninit 20 --niters 300
```

* Ruedesheim (beta in [0, 2])
```
python calibrate.py --seed "rh0" --country "GER" --area "RH" --start "2020-03-10" --end "2020-03-26" --mob "lib/mobility/Ruedesheim_settings_10_10_hh.pk" --downsample 10 --rollouts 80 --ninit 20 --niters 300
```

Switzerland
* Lausanne (beta in [0, 3])
```
python calibrate.py --seed "vd0" --country "CH" --area "VD" --start "2020-03-04" --end "2020-03-18" --mob "lib/mobility/Lausanne_settings_10_10_hh.pk" --downsample 10 --rollouts 80 --ninit 20 --niters 300
```

* Lucern (beta in [0, 2])
```
python calibrate.py --seed "lu0" --country "CH" --area "LU" --start "2020-03-11" --end "2020-03-18" --mob "lib/mobility/Lucerne_settings_5_5_hh.pk" --downsample 5 --rollouts 80 --ninit 20 --niters 300

```

* Locarno (beta in [0, 3])
```
python calibrate.py --seed "lo0" --country "CH" --area "TI" --start "2020-03-03" --end "2020-03-18" --mob "lib/mobility/Locarno_settings_5_5_hh.pk" --downsample 5 --rollouts 80 --ninit 20 --niters 300
```

* Canton Schwyz (beta in [0, 2])
```
python calibrate.py --seed "sz0" --country "CH" --area "SZ" --start "2020-03-09" --end "2020-03-18" --mob "lib/mobility/Schwyz_settings_10_10_hh.pk" --downsample 10 --rollouts 80 --ninit 20 --niters 300
```