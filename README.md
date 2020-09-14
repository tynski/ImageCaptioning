# Image Captioning
Python implementation of the system for generating a textual description of a given photo. For the demonstration of how the API is working check `ImageCaptioning.ipynb` it shows the whole pipeline:
* repository preparation
* prepare input data
* model preparation
* model training
* model validation

It also shows the use case of single image evaluation: predicted caption compared with true caption + attention weights describing the thought process of the model.  

Required dependencies are given in **requirements.txt** file. You can simply run Makefile to install them:

`pip install -r requirements.txt`

## Dataset
Model is trained on **COCO dataset**. I prepared bash script to acquire it, though it is a big amount of data so it requires some dependencies:
* gsutlis
* unzip
* curl

If you have them, you will get dataset by runing **getdataset.sh**: 

`./coco/getdataset.sh`

The script will acquire data and store it in the right location. **WARNING** it is a big amount of data, so be prepared. 

## Script proceeding 
1. Load images
1. Extract features from each image
1. Store extracted features
1. Train/validation split
1. Prepare model 
1. Train model:
	1. save results (loss plot, training history)
	1. checkpoint
	1. optimize trainable variables
1. Validate the trained model:
	1. compare prepared caption with ground truth
	1. prepare attention validation plot

## Model Architecture
* encoder - encode each feature by CNN dense layer
* decoder - decode encoded features by GRU reinforced with attention
* optimizer - ADAM
* loss object - sparse categorical cross-entropy 

## Results
The results of image captioning are stored under `results/` location. They are as follows:
* training loss list, `results/training/loss_plot.pkl`
* training history text file, `results/taining/loss_history.txt`
* validation results list of dictionaries with keys: `image_id` and `caption`,  `results/validation/captions_val2014.json`

The 'results' folder grows quite big. Please create our own results directory or get [mine](https://drive.google.com/file/d/18tvO624PcunwY_cXcLv9pvFVfqFWnr6y/view?usp=sharing).

## Repository
To avoid redundant computations the script is storing computed values necessary in further steps in `image_captioning/repository.pkl` file, repository.pkl contents:
* train feature path list
* train caption list
* tokenizer
* caption maximum length
* starting epoch
* validation image id list
* validation feature path list
* test image id list
* test feature path list

To restore my trained model please download my [repository.pkl](https://drive.google.com/file/d/1HqTEBzQpdNxd9SaFND1I9PWDyI0FT0fg/view?usp=sharing) and place it in directory given above.

## Checkpoints
Image captioning is a very computational demanding task, so Tensorflow checkpoints are utilized. Checkpoints are storing information about given TensorFlow objects:
* encoder
* optimizer
* decoder

This allows to break training at any moment, then return to trained models. The checkpoint system is also used in case of validation and evaluation images, it allows to use previously pre-trained models. Checkpoint are saved in `image_captioning/training_checkpoints/` directory.

To restore my trained model please download my [training checkpoints](https://drive.google.com/file/d/1JpliPfNLet-FjHtAz8JNxPd6ELK7Sz-Y/view?usp=sharing) or train model by yourself.

## TODO
* .git folder is too big
* name standardization/refactoring
* train models on more data
* replace GRU with LSTM
