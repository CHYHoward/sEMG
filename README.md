## EMG-based hand gesture recognition (HGR) undergraduate project

**Step 1**: Download the NinaPro DB2 from the website
```
https://drive.google.com/drive/folders/1fRAQtZqMaVGaBJxBUiCQyEI7Vj_OSdXX?usp=sharing
download whole DataSet DB2(most popular dataset in the field)
```
**Step 2**: Start experiments! 
```
**CNN**
run_CNN.sh
type_filter=(several filter for the input signal) etc none, BPF(band pass filter) see data_preprocess.py
type_norm=(several normalization method) etc mvc, min_max, standization see feature_extractor.py
run_CNN_dropout.sh
change dropout rate

**DNN**
run_DNN_feature_depth.sh (different depth model) see dnn.py
**ViT = Tnet + Fnet**
inter = train by several subjects and test on several subjects.
intra = train by one single subject and test on one single subjects
p = windows pieces
s1-s10 = subject in the DB2 dataset
```

**Step 3**: Run the `main_develop_DB2.py` file to train and test the model.
```
python3 main_develop_DB2.py
```
or you can use the `run.sh` or `run_{model_type}.sh` script for detail argument setup as
```
./run.sh
```
**Step 4**: Start experiments! \
You can add new functions (e.g., filter, normalization, feature extraction, ...) and new model (e.g., MV-CNN, TraHGR, LST-EMG-NET, ...), and also change the argument setup to find the best hyperparameter.

**Setp 5**: Some sample result
  
