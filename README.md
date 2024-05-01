# Project Aria Eye Tracking

The primary function of this code is to provide tools to estimate gaze direction given Aria eye tracking camera images. The estimates are generated from a pytorch model trained on Aria device data.\
Additonally, this code contains tools to visualize the raw and derived gaze data.

# Installation & Usage


## Running from the GitHub repository
```
# Install requirements
python3 -m pip install easydict pyyaml torch torchvision projectaria_tools
```

### Run the demo
```
cd projectaria_eyetracking
python3 model_inference_demo.py --vrs sample.vrs
```
Enjoy!\
A file `general_eye_gaze.csv` contains the EyeGaze estimates that are compatible with projectaria_tools API.

## Install and run on your system
```
python3 -m ensurepip
python3 -m pip install .
eyetracking_demo --vrs sample.vrs
```

## License

The model is licensed under the [Apache 2.0 license](LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).
