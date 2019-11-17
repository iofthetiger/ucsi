# UCSI
The git for the could detection

### Competition
* [Understanding Clouds from Satellite Images](https://www.kaggle.com/c/understanding_cloud_organization)


### Installations

* Installation [documentation](doc/INSTALL.md)

* Python [requirements for training](requirements.txt)

### Notebooks

* Core [training notebook](ref_b5.ipynb)

* Training [notebook](catalyst_train_newinf.ipynb) with threshold finding on training dataset


### Leads
* [x] FP16 will easily lead to gradient overflow in this case
* [x] Ensemble
* [x] Polygon
* [x] More experiments on min size/threshold
* [ ] sigmoid 1st then to float64

### Ploygon Convex Post Processing
 This is a CPU only operation
 ```
 python polygon_cpu.py --csv=subxxx.csv --minsize=5000
 ```
 
### Google Storage Access

* Check [this test notebook](google_storage_test.ipynb) for storage access api

* The list, download, upload functions are in [```./utils_google.py```](utils_google.py)