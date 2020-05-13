***************************
README Calibration
***************************

Run

```python calibrate.py --help```

for help regarding arguments required for model calibration.

Inference runs:

Germany
1) Tubingen
```
python calibrate.py --seed "tu0" --country "GER" --area "TU" --start "2020-03-10" --end "2020-03-26" --mob "lib/mobility/Tubingen_settings_10_10_hh.pk" --downsample 10 --rollouts 80 --ninit 20 --niters 300
```

Switzerland
1) Lausanne
```
python calibrate.py --seed "lu0" --country "CH" --area "LU" --start "2020-03-10" --end "2020-03-18" --mob "lib/mobility/Lausanne_settings_10_10_hh.pk" --downsample 10 --rollouts 80 --ninit 20 --niters 300
```

