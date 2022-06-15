Sets of classes and interfaces for the definition of deep learning models.
The framework is conceived to be extensible and serializable, in order to provide models that can be stored and/or sent through the network.
## Interfaces.py
Set of abstract classes that describe models and model providers. 
### DeepLearningClass
The rationale behind this is that an algorithm not only depends on the model, but might require some preprocessing steps (reformatting, normalization) before the application of the model itself. The interface is also "incremental-learning-oriented" as it provides abstract methods for the calculation and integration of deltas. The class provides operator override for the calculation of deltas (D = A-B) and call override for the apply method.
An instance of DeepLearningClass should be able to simply provide the expected result when called with the appropriate input data (in form of dict). The most common data for our case will be:

 - Input: `{'image': 2D image, 'resolution': Sequence with pixel sizes}`
 - Output: for Classifiers: `str`, for Segmenters: `dict[str: image]` where str is the label and the image is a 2D numpy array containing the mask corresponding to the string.

### ModelProvider
This is an abstract class that is intended to encapsulate the logic of model load/transfer. It will have two subclasses, LocalModelProvider and RemoteModelProvider. In both cases, the method load_model of this class accepts a description of the requested model and will return an instance of DeepLearningClass. In principle, LocalModelProvider will be used on the server, and RemoteModelProvider on the client.

## DynamicDLModel
Concrete implementation of a DeepLearningClass which provides flexibility and serialization. This class delegates the important methods of the model implementation to top-level functions that are passed at the construction time. These functions can be easily serialized with the dill library.
Constructor:

        def __init__(self, model_id, # a unique ID to avoid mixing different models
                 init_model_function, # inits the model. Accepts no parameters and returns the model
                 apply_model_function, # function that applies the model. Has the object, and image, and a sequence containing resolutions as parameters
                 weights_to_model_function = default_keras_weights_to_model_function, # put model weights inside the model.
                 model_to_weights_function = default_keras_model_to_weights_function, # get the weights from the model in a pickable format
                 calc_delta_function = default_keras_delta_function, # calculate the weight delta
                 apply_delta_function = default_keras_apply_delta_function, # apply a weight delta
                 incremental_learn_function = None, # function to perform an incremental learning step
                 weights = None): # initial weights
This allows the implementation of a very generic deep learning algorithm which includes the preprocessing steps in a way that can be serialized and defined at runtime, so if we want to change the model, we don't need to change the code of the client or server, as the implementation is self-contained within the model.
The class provides the methods `dump(file_descriptor)` and `str = dumps()` to serialize and the static methods `Load(file_descriptor)` and `Loads(str)` to deserialize.
Default functions for loading/setting keras weights and calculating deltas from keras models (which provide a get_weights(), set_weights() interface with lists of numpy arrays) are currently provided.
**Important note when defining the functions**: in order for them to be serializable, they must be completely self-contained. That is, all imports should happen inside the functions and all the external function call should be implemented as nested functions. Common algorithms (such as padorcut.py which pads or cuts an image to fit it to a specific matrix size) should be placed in the repository.
