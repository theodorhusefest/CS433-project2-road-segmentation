# Road Segmentation Project - EPFL Autumn 2019

Project Members:
- Anna Vera Linnea Fristedt Andersson
- Erik Agaton Sjöberg
- Theodor Tveit Husefest

Best submission 0.902 - #29400 on AICrowd.

## How to run

Install required packages with either ```pip install -r requirements.txt``` or ```conda install requirements.txt```.

Before running the program make sure to unzip the files in the data folder.  

To run the program call ```python3 run.py ``` from root directory to create submission with best pre-trained model.  
To retrain the model use: ```python3 run.py --train```.

The best model is trained with 4 GPU's and need 4 GPU's to create a new submission. This is an error in Keras, but there is no problem when training a new model.