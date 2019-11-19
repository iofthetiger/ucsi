# UCSI
The git for the could detection

### Competition
* [Understanding Clouds from Satellite Images](https://www.kaggle.com/c/understanding_cloud_organization)
* We are team ```thunderstroke```, we got a bronze medal after the shakeup.

<img src="https://camo.githubusercontent.com/e9a8db6313cab9fb275ee5f7d8387ab81508a9fc/68747470733a2f2f692e696d6775722e636f6d2f454f767a356b642e706e67" alt="" data-canonical-src="https://i.imgur.com/EOvz5kd.png">

### Installations
* Installation [documentation](doc/INSTALL.md)
* Python [requirements for training](requirements.txt)

### Experiments

#### **FASTAI** API
* First, we tried the fastai [training notebook](train_fastai_nofunnel.ipynb)
* Unfortunately, for all experiments in this competition, we can't achieve any decent result from fastai API, no even a stable score above 0.5, there must be some detailed that we used it wrong, after hours of testing and debug, we decided switching to catalyst.

#### **Catalyst** API
* Our [baseline catalyst training notebook](catalyst_train_newinf.ipynb)
* Our ensemble [baseline notebook](https://github.com/iofthetiger/ucsi/blob/master/catalyst_ensemble.ipynb)

#### Tactics & Maneuvers
* Half precision: using float16 to replace float32 as basic float point storage and computation to save CUDA computation power. This method brings huge advantage to our previous competitions: [Recursion Cell Image Classification](https://github.com/raynardj/python4ml/tree/master/experiments/rcic), [Severstal Steel Defects Detection](https://github.com/raynardj/ssdd)
    * Our [fastai fp16 experiment](unet_fpn_train_fp16_fastai.ipynb)
    * Then we move on to [catalyst fp16 experiments](ref_b5_fp18.ipynb)
    * The loss of accuracy proved significant and detrimental to public LB score. We ditched the fp16 idea.

* **Convex** hull post processing, the [notebook version](polygon_cpu.ipynb) are from [this open kernel](https://www.kaggle.com/ratthachat/cloud-convexhull-polygon-postprocessing-no-gpu), please upvote this kernel
    * We managed to simplify the process to [this script](https://github.com/iofthetiger/ucsi/blob/master/polygon_cpu.py), so we can use it in this way:```python polygon_cpu.py --csv=subxxx.csv --minsize=5000```
    * Convex hull can bring **public LB score up around 0.002**. Though upon reviewing our submissions after the private LB released. Many submissions without convex process are much higher. The true effect on private LB is then, unknown.
    * Within this post process, total black strips are removed, aka ```img[img<(2/255)]=0.```. It's probably this technique that improve the public LB actually.

* Ensemble the probability, our ensemble [baseline](catalyst_ensemble.ipynb)
    * As usual, ensemble produce magic this time, improvement on public BL is around 0.003~0.012
    * Ensemble between similar score models are better
    * Ensemble between difference sizes of models are better (All our final submission tries are on different sizes)

* Ensemble using JIT.
    * Save the model to [jit weights](le_jit.ipynb)
    * In this contest, jit savings aren't entirely truthful

* Funnel structure. We fancied a funnel structure, which allows us digest larger pixel size and predict smaller ones. It didn't do well under experiment.

* **K-fold** Stack
    * Our K-fold [training notebook](catalyst_train_kfold.ipynb), all our final tries are 3~5 folds stacked. With our own ```KFoldManager``` class
    * All best weights (decided by **validation dice loss**) are saved in different log directory, uploaded to GCP bucket.
    * The improvement is not significant, but at least keep producing stable/healthy good result.

* **CSV ensemble**. Ensemble using models is time consuming and takes more effort to control. Instead we use submission csv to ensemble.
    * By using the discrete prediction (reconstructed run length encoding) instead of continuous prediction, and not opening image and not running through model. A huge ensemble can be performed within 5 minutes.
    * Our [csv ensemble notebook](ensemble-from-csv).

* Removing prediction using **classification model**.
    * As [this open kernel (a nice kernel, please upvote)](https://www.kaggle.com/mobassir/keras-efficientnetb2-for-classifying-cloud) suggested, even this is a segmentation competition. we can use a classification model to enhance our accuracy.
    * For the convenience of engineering, we didn't run classification model every time, we used the trained classified model to produce an [empty prediction list](empty_list.csv). It's a list of classification prediction on test dataset, all the image+class that has probability below 0.2.
    * Combine with the rumor about data leak, (at least 1 class, we have dedicated effort for this leak). We remove the prediction (from rle to '') if:
        * The image+class is in the [empty list](empty_list.csv).
        * It's not the single prediction of the picture.
        * We boil down the operation to ```python empty_list_filter.py --csv=xxxxx.csv```, we use this each time before we submit the csv since T - 30 hours.
        * The improvement on public LB is around 0.0010

* Test Time Augmentation (**TTA**), we start to use TTA since this [ensemble inference notebook](https://github.com/iofthetiger/ucsi/blob/master/catalyst_ensemble_v3_tta.ipynb).
    * We flip the tensor upside down and horizontally during inference and flip the y hat back.
    * So the inference time x3.
    * The improvement isn't detectable actually, but we stick through this method anyway.

* Data leak, as suggested by user Heng CherKeng, **each picture has, at least 1 class prediction**, hence no class predicted would be wrong
    * In the [ensemble process here](https://github.com/iofthetiger/ucsi/blob/master/catalyst_ensemble_v4_dataleak.ipynb), we created the idea of **discount factor**.
        * If no prediction is made under this picture, we will decrease the threshold by 0.05, and minimum section size by 800. ```threshold = THRESHOLD-discount*0.8```, ```minsize = MINSIZE-discount*800```
        * We can discount as much as 6 times to keep reasonable threshold and minsize.
    * Our [operation private Ryan](csv_safenet.ipynb) is targeting these missing predictions, to use predictions from other csv submissions. To guarantee at least 1 class prediction for each picture, as much as possible.
    * The public LB from 0.66560 to 0.66566. The private LB score, on retrospect, got worse. Our final selected submissions don't use operation Ryan at all.

### Google Storage Access

* Check [this test notebook](google_storage_test.ipynb) for storage access api

* The list, download, upload functions are in [```./utils_google.py```](utils_google.py)
