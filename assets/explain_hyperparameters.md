# Hyperaparameters
Here all hyperparameters for training/retraining a model with this framework are explained. Each explanation reports and possible values. A typical example can be found of hyperparameter configuration use can be found in [this example](../scripts/kitti_tests/kitti_squeezeDet_config.json).

### CLASS_NAMES 
Object categories to classify refered by name.

possible values: Array of strings

example: `["car", "pedestrian", "cyclist"]`

### LABEL INDICES
If only a subset of a dataset is used then this array contains the indices of the class names used in the initial full array. On the reduction of the initial dataset to a subset [here](../src/datasets/imdb.py#264) it is used.

possible values: Array of integers >= 1

example: `[1,2,3]`

### CLASSES 
Number of categories to classify = length of the CLASS_NAMES array.

possible values: Integer

example: `3`

### LEAKY_RELU
This hyperparameter indicates if leaky ReLUs are going to be used, rather than simple old (original) ReLUs. 
By default its value is false, as in the original work.

possible values: Boolean

example: `true`

### LEAKY_COEF 
Parameter used in leaky ReLU modules.

possible values: 0 <= LEAKY_COEF <= 1.0 

example: `0.1`

### KEEP_PROB 
Probability to keep a node in dropout modules.

possible values: 0 < KEEP_PROB <= 1.0 

example: `0.5`

### IMAGE_WIDTH 
Image width that the feature extractor receives.

possible values: Integer > 0

example: ``for SqueezeNet used by SqueezeDet: 1248``

### IMAGE_HEIGHT 
Image height that the feature extractor receives.

possible values: Integer > 0

example:
    ``for SqueezeNet used by SqueezeDet: 384``

### ANCHOR_BOX 
Anchor box, array of [cx, cy, w, h]. It is the ground trouth of responsible anchor centers and shapes.

possible values: Array of floats >= 0. Note: _It is automatically-created_.

### ANCHORS 
Number of anchor boxes.

possible values: len(ANCHOR_BOX), integer > 0

### ANCHOR_PER_GRID 
Number of anchor boxes per grid. As grid is ment the grid of the pre-final layer. This indicates also how many different type of centers will be created by the [get_initial_anchor_shapes](../src/datasets/imdb.py#335).

possible values: integer > 0

example: `9`

### INIT_ANCHOR_SHAPES 
Dictionary describing the method which will initialize the default values for biased width and height of the anchor boxes. The supported methods are:
+ "CONST"
+ "KNN"
possible values: 
example:
1. ``{"METHOD" : "KNN"}``
2. ``{"METHOD" : "CONST", "VALUE": [[0.02898551, 0.09866667], [0.29146538, 0.464], [0.09259259, 0.15733333], [0.13043478, 0.232], ...,[0.05797101, 0.11466667]]}``

### BATCH_SIZE 
Batch size to be used for training/evaluation.

possible values: Integer >= 1

example: `20 in the KITTI dataset`

### NMS_THRESH 
Bounding boxes pairs with IOU larger than this are going to have one of the boxes in the pair removed by the NMS algorithm.

possible values: 0.0 < NMS_THRESH <= 1.0

example: `0.4`

### TOP_N_DETECTION 
Maximum number of bounding boxes with highest score to accept in convDet filtering.

possible values: Integer > 0

example: `64`,  `128`

### BGR_MEANS 
Pixel mean values (BGR order) as a (1, 1, 3) array. It is the same used in pre-training, if there was one, the feature extractor. For example in SqueezeDet we use the `BGR_MEANS` used for training SqueezeNet in ImageNet. 

possible values: a float array with shape (1, 1, 3)

example: `[[[103.939, 116.779, 123.68]]]`

### LOSS_COEF_CLASS 
Loss coefficient for classification regression.

possible values: Float

example: `1.0`

### LOSS_COEF_BBOX
Loss coefficient for bounding box regression

possible values: Float

example: `5.0`

### LOSS_COEF_CONF_POS 
Possitive loss coefficient for confidence score regression

possible values: Float

example: `75.0`

### LOSS_COEF_CONF_NEG 
Negative loss coeefficient for confidence score regression

possible values: Float

example: `100.0`

### DECAY_STEPS 
Reduce learning rate after this many steps

possible values: Integer > 0

example: `10000`

### LR_DECAY_FACTOR 
Update the learning rate by multiplication with this factor each `DECAY_STEPS`.

possible values: 0 < `DECAY_STEPS` < 1.0

example: 0.5

### LEARNING_RATE 
Initial learning rate. As small the datasets gets, so larger it has to be, bacause the agent should learn as much as possible from the dataset.

possible values: 0.0 < `LEARNING_RATE` < 1.0 to be stable.

example: `0.01`

### OPTIMIZER
Dictionary describing the optimizer's algorithm. The fields it can contain are depend on the field "TYPE".
"TYPE" can be either:
+ "MOMENTUM"
+ "ADAM"

if "TYPE" is "MOMENTUM" then an extra field "MOMENTUM" is needed:
#### MOMENTUM
    Momentum, used for momentum optimizer.
    possible values: 0 <= MOMENTUM < 1.0
    example: 0.9

if "TYPE" is "ADAM" then two extra fields "BETA1" and "BETA2" are needed:
#### BETA1 
    Adam's beta1
    possible values: Float < 1.0 
    example: 0.9
#### BETA2 
    Adam's beta2
    possible values: Float < 1.0, BETA2 > BETA1
    example: 0.999
For more information about these fields, see [this article](https://arxiv.org/abs/1609.04747).

### WEIGHT_DECAY
Weight decay. For more see [this](https://stats.stackexchange.com/a/31334).

possible values: Float < 1.0

example: `0.0001`

### LOAD_PRETRAINED_MODEL 
Whether to load pre-trained model of feature extractor. By default the feature extractors supported are SqueezeNet, VGG16, ResNet50.

possible values: Boolean

example: `true`

### PRETRAINED_MODEL_PATH 
Path to load the pre-trained model.

possible values: Existing path string.

example: `"/media/terabyte/projects/Thesis/SqueezeNet_imageNet_trained/squeezenet_v1.1.pkl"`

### DEBUG_MODE 
Print log to console in debug mode.

possible values: Boolean

example: `false`

### EPSILON
A small value used to prevent numerical instability of divisions by zero.

possible values: Float

example: `Numpy's np.eps` , `1e-16`

### EXP_THRESH 
Threshold for safe exponential operation.

possible values: Float

example: `1.0`

### MAX_GRAD_NORM 
Gradients with norm larger than this are going to be clipped during back-propagation.

possible values: Float

example: `1.0`

### DATA_AUGMENTATION
Whether to do data augmentation.

possible values: Boolean

example: `true`

### DRIFT_X 
The maximum random shift of the image left or right during data augmentation. This ensures that the bounding box will not move wherever in the horizontal dimension.

possible values: Integer >= 0 and smaller than IMAGE_WIDTH

example: `150`

### DRIFT_Y 
The maximum random shift of the image up or down during data augmentation. This ensures that the bounding box will not move wherever in the vertical dimension. 

possible values: Integer >= 0 and smaller than IMAGE_HEIGHT

example: `100`

### EXCLUDE_HARD_EXAMPLES 
Whether to exclude images harder than hard-category. Only useful for KITTI dataset. __Note: Not currently in use__.

possible values: 

example: 


### FREEZE_LAYERS 
Dictionary defining which layers of the net should freeze during training.

possible values: Dictionary, should specify all layers in model.

example: for SqueezeDet: 
    
    {
        "conv1": false, <-- indicates that this layer 
                            will not be frozen in training.
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
    }

### HOT_LAYERS
The reverse of the __FREEZE_LAYERS__. It is automatically constructed.

### IS_TRAINING 
Indicate if the model is going to be trained.

possible values: Boolean

example: `true`

### NUM_THREADS
Number of threads to be used for reading and parsing data.

possible values: Integer > 0

example: `16 for an 8-cores system`

### ALREADY_PREPROCESSED
If dataset has already been reduced/preprocessed before this training.

possible values: Boolean

example: `true`

### PREPROCESSED_DATA_DIR
The path where the already reduced/preprocessed dataset is. This path will be used for fetching the data, rather than the __DATA_PATH__ variable if the __ALREADY_PREPROCESSED__ flag is true.

possible values: Existing path string

example: `/media/terabyte/projects/Thesis/trainings/Pascal_TRAIN_DIR3`

### REDUCE_DATASET
If the dataset will be reduced to one with less classes, the reduced one will be copied to the __BASE_DIR__ folder. The operation will take place only if the __ALREADY_PREPROCESSED__ flag is false. This is a good choice for small datasets such as PASCAL VOC which do not require much disk memory and can improove the training speed. 

possible values: Boolean

example: `true`

### DATASET_NAME
A string declaring the dataset annotation type.

possible values: one of the three strings "PASCAL_VOC", "KITTI", "COCO". Any other value defaults to "PASCAL_VOC".

### NET
The name of the neural network model to be used.

possible values: one of "squeezeDet", "squeezeDet+", "vgg16+convDet", "resnet50+convDet". Any other value defaults to "squeezeDet+".

### DATA_PATH
The path where the dataset's record files are. If I list the files to the __DATA_PATH__, I see:

```
ls /media/terabyte/projects/datasets/pascal-voc
annotations_cache  pascal_voc_train.record  pascal_voc_val.record  VOC2012
``` 
The files needed for training/evaluation have names `*.record`.

possible values: Existing path string.

example: `"/media/terabyte/projects/datasets/pascal-voc"`

### BASE_DIR
Where all training and evaluation files will be stored. It is also the default folder where the reduced dataset is saved.

possible values: Existing path string.

example: `"/media/terabyte/projects/Thesis/trainings/Pascal_full_train_dir0"`

### TRAIN_DIR
Where all the training files will take place. It is better to specify the __BASE_DIR__ variable and let the __TRAIN_DIR__ to be created automatically.

possible values: Existing path string.

example: `"/media/terabyte/projects/Thesis/trainings/Pascal_full_train_dir0/train"`

### EVAL_DIR
Where all files after evaluation will take place. It is better to specify the __BASE_DIR__ variable and let the __EVAL_DIR__ to be created automatically.

possible values: Existing path string.

example: `"/media/terabyte/projects/Thesis/trainings/Pascal_full_train_dir0/evals"`

### EVAL_WITH_TRAIN
If during training after every __EVAL_PER_STEPS__ number of steps evaluation will take place.

possible values: Boolean

example: `true`

### max_steps
Maximum number of training steps. At each step a batch of input is used.

possible values: Integer >= 0

example: `35000`

### SUMMARY_STEP
At each __SUMMARY_STEP__ number of steps a summary is saved. The summary can be viewed using tensorboard.

possible values: Integer > 0

example: `2000`

### checkpoint_step
At each __checkpoint_step__ number of steps a checkpoint is saved. The checkpoint can be loaded later for retraining/evaluation or any other use.

possible values: Integer > 0

example: `2000`

### GPU               
GPU id to be used.

possible values: Integer

example: `0`

### EVAL_PER_STEPS    
At each __EVAL_PER_STEPS__ number of steps evaluation takes place and the results are saved in the __EVAL_DIR__.

possible values: Integer > 0

example: `2000`

### SAVE_XLA_TIMELINE 
Whether to activate the XLA timeline saving procedure. The timelines are saved and can be viewed using google chrome.

possible values: Boolean

example: `false`

### VISUALIZE_ON
Whether to save visualization data in summaries. Defaults to `true`.

possible values: Boolean

example: `true`

### HOPT
Dictionary describing how to use hyperoptimization of hyperparameters (variables in a json file).
To declare a hyperparameter in a json file as hyperoptimizable you have to folow the guide in the [README.md](../README.md#Hyperoptimization) file.
For the variables declared as hyperoptimizable, a hyperoptimization algorithm can be used for a certain number of steps.
The algorihm can be stopped any time, even before all the iterations have been executed. At each iteration the results are saved to
a log file. So before maximum number of iterations has been executed, the user can receive some results and it can continue afterwards,
because the hyperoptimizer's state is saved also after each iteration. To do the later, the same folder should be used as before.
The HOP dictionary contains two fields:
#### MAX_ITERATIONS
    This variable describes the maximum number of iterations the hyperoptimizer will perform.
    possible values: Integer > 0
    example: 20
#### ALGORITHM
    This variable is a string with the name of the method to be used for hyperoptimization.
    possible values: "adalipo". Only this value is supported and it uses dlib.global_function_search.
    example: "adalipo"

