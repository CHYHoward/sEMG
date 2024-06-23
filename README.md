## EMG-based hand gesture recognition (HGR) undergraduate project

**Step 1**: Download the NinaPro DB2 from the website
```
cd Dataset/DB2
./DB2_download.sh
```
**Step 2**: Back to `EMG_HGR_UG` dictionary, convert the mat file to numpy file with segmentation
```
cd ../../
python3 mat2np_segment_all_subject.py
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

