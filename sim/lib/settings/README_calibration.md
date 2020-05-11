***************************
README Calibration
***************************

Run

```python calibrate.py --help```

for help regarding arguments required for model calibration.
Example runs are for example:

```
python calibrate.py --seed "hb0" --country "GER" --area "HB" --start "2020-03-10" --days 16 --mob "lib/mobility/Heinsberg_settings_5_5_hh.pk" --downsample 5
python calibrate.py --seed "tu0" --country "GER" --area "TU" --start "2020-03-10" --days 16 --mob "lib/mobility/Tubingen_settings_10_10_hh.pk" --downsample 10
python calibrate.py --seed "rh0" --country "GER" --area "RH" --start "2020-03-10" --days 16 --mob "lib/mobility/Ruedesheim_settings_10_10_hh.pk" --downsample 10 
python calibrate.py --seed "kl0" --country "GER" --area "KL" --start "2020-03-18" --days 8 --mob "lib/mobility/Kaiserslautern_settings_10_10_hh.pk" --downsample 10 
python calibrate.py --seed "sz0" --country "CH" --area "SZ" --start 2020-03-09 --days 7 --mob "lib/mobility/Schwyz_settings_10_10_hh.pk" --downsample 10
python calibrate.py --seed "lo0" --country "CH" --area "TI" --start "2020-03-02" --days 14 --mob "lib/mobility/Locarno_settings_5_5_hh.pk" --downsample 5
python calibrate.py --seed "lu0" --country "CH" --area "LU" --start "2020-03-09" --days 7 --mob "lib/mobility/Lucerne_settings_10_10_hh.pk" --downsample 10
python calibrate.py --seed "la0" --country "CH" --area "VD" --start "2020-03-02" --days 14 --mob "lib/mobility/Lausanne_settings_10_10_hh.pk" --downsample 10






```