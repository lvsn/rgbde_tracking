# RGB-D-E_tracking
**Project page available [here](https://lvsn.github.io/rgbde_tracking).**

Evaluation code and dataset from "RGB-D-E: Event Camera Calibration for Fast 6-DOF Object Tracking
" [\[arxiv paper\]](http://arxiv.org/abs/2006.05011)


## Evaluation Dataset

Download the evaluation dataset [here (12 GB)](https://hdrdb-public.s3.valeria.science/rgbde/rgbde_dataset.zip).

The dataset contains multiple sequences each in different folder.
Each sequence contains the following files:

 * `camera.json`: RGB-D sensor (Microsoft Kinect Azure) intrinsic calibration
 * `dvs.json`: Event based sensor (DAVIS346) intrinsic calibration
 * `transfo_mat.npy`: Extrinsic calibration
 * `fevents.npz`: Events data of shape 4xN (Timestamps, x, y, Polarity) 
 * `davis_frame.npz`: Grayscale frame record from the event sensor    
 * `davis_frame_ts.npz`: Timestamps associated to each grayscale frame
 * `frames.npz`: RGB-D frames from the Microsoft Kinect Azure
 * `ts_frames.npz`: Timestamps associated to each RGB-D frame
 * `poses.npy`: Ground truth 6DoF poses

## Tracker

### Download

Download and extract:
 * [3D model](https://hdrdb-public.s3.valeria.science/6dofobjecttracking/dragon_model.tar.gz)
 * [pre-trained networks](https://hdrdb-public.s3.valeria.science/rgbde/rgbde_model.zip)
 * [dataset](https://hdrdb-public.s3.valeria.science/rgbde/rgbde_dataset.zip)

### Running tracker

This repository used submodule and it should be initiated:
```
git submodule update --init --recursive
```

Update your `PYTHONPATH`:
```
export PYTHONPATH=$PYTHONPATH:./6DOF_tracking_evaluation
```

To run the tracker on the whole dataset and compute the tracking failures for both networks:
```
python tracking_event_6dof/inference/tracker_failure.py \
    -e ./model/event
    -f ./model/frame
    -d ./dataset
    -m ./dragon
```

To generate video result of each sequence:
```
python tracking_event_6dof/inference/tracker_failure.py \
    -e ./model/event
    -f ./model/frame
    -d ./dataset
    -m ./dragon
    -a /path/to/folder/to/save/videos
```


**Note**: Those examples suppose that the assets are extracted in the root folder

## Citation

```
@misc{dubeau2020rgbde,
    title={RGB-D-E: Event Camera Calibration for Fast 6-DOF Object Tracking},
    author={Etienne Dubeau and Mathieu Garon and Benoit Debaque and Raoul de Charette and Jean-François Lalonde},
    year={2020},
    eprint={2006.05011},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

