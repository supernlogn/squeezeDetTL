# Usage for your projects
This text explains how to use the ms coco dataset with or without your own model. It basically explains the hyperparameters needed for this purpose.

All the hyperparameters can be found in [this](scripts/coco_tests/coco_squeezeDet_config.json) file.
For detailed explanation of each hyperparameter please refer [here](assets/explain_hyperparameters.md).

### Creating the tf record files
First of all, we need to create the dataset in a tf record format (if already then avoid these steps and continue to create your json configuration). Examples are in [this](scripts/dataset_creation) folder. Here we use the MS COCO dataset. So we type to a terminal:

    cd to_your_project_folder
    # personalize the 2 lines below
    BASE_DIR=/media/terabyte/projects/datasets/COCO
    ANN_DIR=$BASE_DIR/annotations
    python  ../../src/datasets/creation/create_coco_tf_record.py --data_dir=$BASE_DIR --set=train \
    --output_filepath=$BASE_DIR/coco_train.record   --shuffle_imgs=False
    python  ../../src/datasets/creation/create_coco_tf_record.py --data_dir=$BASE_DIR --set=val \
    --output_filepath=$BASE_DIR/coco_val.record     --shuffle_imgs=False


### Creating the json configuration
In order to specify the dataset you want to use, you have to just type:

    "DATASET_NAME"           : "COCO"

then you have to select the classes you want, below we select the classes with labels:
"/m/01g317", "/m/0199g", "/m/0k4j"

    "CLASS_NAMES"            : ["/m/01g317", "/m/0199g", "/m/0k4j"]

and we put their corresponding label indices starting always from 1 as found in the label map files (extension: .pbtxt).

We put the number of classes we want to train:
    "CLASSES"                : 3,

The data path, where the initial tensorflow record files exist. If you just created them use the path you used as $BASE_DIR:
    "DATA_PATH"              : "/media/terabyte/projects/datasets/COCO/"
Some folder to use for temporary pre-training processes:
    "PREPROCESSED_DATA_DIR"  : "/media/terabyte/projects/Thesis/trainings/coco_TRAIN_DIR0"

The time of choosing our model has arrived, 'SqueezeDet i pick you!':
    "NET"                    : "squeezeDet",

Some hyperparameteres which relate to your pretrained model are width and height. In order to change them we would have to perform the initial training again:

    "IMAGE_WIDTH"            : 1248,
    "IMAGE_HEIGHT"           : 384,
There other hyperparameters which are solely based on our new training:

    "BATCH_SIZE"             : 20,
    "IS_TRAINING"            : true,
    "REDUCE_DATASET"         : true,
    "ALREADY_PREPROCESSED"   : false,
    "EVAL_WITH_TRAIN"        : true,
    "TRAIN_EPOCHS"           : 200, 

    "OPTIMIZER"              : { "TYPE": "MOMENTUM",
                                "MOMENTUM"              : 0.9,
                                "BETA1"                 : 0.9,
                                "BETA2"                 : 0.999
                               },

    "WEIGHT_DECAY"           : 0.0001,
    "LEARNING_RATE"          : 0.01,
    "DECAY_STEPS"            : 10000,
    "MAX_GRAD_NORM"          : 1.0,
    "MOMENTUM"               : 0.9,
    "LR_DECAY_FACTOR"        : 0.5,

    "LOSS_COEF_BBOX"         : 5.0,
    "LOSS_COEF_CONF_POS"     : 75.0,
    "LOSS_COEF_CONF_NEG"     : 100.0,
    "LOSS_COEF_CLASS"        : 1.0,
  
    "NMS_THRESH"             : 0.4,
    "PROB_THRESH"            : 0.005,
    "TOP_N_DETECTION"        : 64,
  
    "DATA_AUGMENTATION"      : true,
    
    "DRIFT_X"                : 150,
    "DRIFT_Y"                : 100,
    "EXCLUDE_HARD_EXAMPLES"  : false,
    "FREEZE_LAYERS"          : {
        "conv1": true,
        "fire2": false,
        "fire3": false,
        "fire4": false,
        "fire5": false,
        "fire6": false,
        "fire7": false,
        "fire8": false,
        "fire9": false,
        "fire10": false,
        "fire11": false,
        "conv12": false
    },

Then we need to choose how anchor matching will happen and will be the default anchor shapes, this can be done automatically or maually. In the example below we do it manually:

    "INIT_ANCHOR_SHAPES"     : {"VALUE":
                               [[ 0.02884615,  0.09635417],
                               [ 0.29326923,  0.453125  ],
                               [ 0.09214744,  0.15364583],
                               [ 0.12980769,  0.2265625 ],
                               [ 0.03044872,  0.234375  ],
                               [ 0.20673077,  0.45052083],
                               [ 0.17948718,  0.28125   ],
                               [ 0.0625    ,  0.44270833],
                               [ 0.05769231,  0.11197917]],
                                "METHOD": "CONST"},

For doing it automatically, we could write:

    "ANCHOR_PER_GRID"        : 9,
    "INIT_ANCHOR_SHAPES"     : {"METHOD": "KNN"},

After all this we have to specify where our training will take place. E.g.:

    "TRAIN_DIR"               : "/media/terabyte/projects/Thesis/trainings/coco_TRAIN_DIR0"

It should be an empty folder for a new retraining. If it is not then the training will use the checkpoint files inside it as an initial node for resuming the retraining. So if you have a checkpoint file from a previous training then you can copy-paste it inside the folder and edit the "checkpoint" file inside it to point to the last checkpoint to your copy-pasted checkpoint file.

However, for using a pretrained model there is another way:
Inform the framework that pretrained model weights are going to be used:

    "LOAD_PRETRAINED_MODEL"  : true,

Then provide the framework with the pretrained model weights in a binary pkl file such as the ones found [here](https:www.dropbox.com/s/a6t3er8f03gd14z/model_checkpoints.tgz) :

    "PRETRAINED_MODEL_PATH"  : "/media/terabyte/projects/Thesis/SqueezeNet_imageNet_trained/squeezenet_v1.1.pkl"


After all this you are good to go!!!
