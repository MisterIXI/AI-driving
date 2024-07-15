# Model.py explanation
`Model.py` is the main file for the AI and here is the explanation for the one used in the Tensorflow version.  
## General architecture
This file contains one class that you initialize with the right settings and then use all AI operations you want to do.  
## Initialization
For initialization you should provide the following parameters: (all are optional)  
- debug_level: (int) 0-3, 0 is no debug output, 3 is all debug output (There are DEBUG level constants provided in the file)  
- IMG_WIDTH/IMG_HEIGHT: (int) the pixel width and height of the input of the model.  
- path_name: (str) the path to the folder where the model will be saved and loaded from.  
Once the model class is initialized, you can freely access internal parameters and adjust them for your use case (like learning rate, epsilon decay, etc.).
## Saving and Loading
The model saves itself after every training step, and can be loaded from the files. In the pytorch version, more parameters are saved to file, so for the tensorflow version, all parameters set for the initial training should be manually adjusted each time.  
## Creation from scratch
If you want to create a new model, you can use the `create_model()` function. This requires information about the shape of possible actions, which it will use to return information for, and the layernames are just saved as internal parameters for later use.  
Here the actual model is create via the tensorflow Keras API.
It starts with the convolutional layers and then flattens the output to feed it into the dense layers.  
## Running the model
The model will record all information during "play" and save them in the end. For that you call the `step()` function with the current state and possibly bonus reward. The model will add neutral rewards on its own, and a negative reward for dying. You can configure those values after creation or loading of the model.  
To finish off a run, you have to call the `finish_run()` function, which will save the final reward to the last state and then save the state data to disk.  
## Training the model
At any time there exists data in the respective folder, you can call the `train()` function to train the model. It will automatically randomly (weighted towards newer runs) pull from past experiences to train on.  
The function will always save and overwrite the model on disk after the training call.  
