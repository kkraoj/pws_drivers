# pws_drivers

1. `rf_regression.py` is used for pws regression
1. `dirs.py` is a one stop shop to change directory locations of files
1. `data` directory contains h5 file required for regression
1. Not needed:
	1. `make_data.py` creates dataframe with input features and output and saves it as h5
	1. `reproject_clip.py` has fast implementation of clip and reproject. This was needed to create tifs at resolution of pws.