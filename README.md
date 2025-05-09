# flower-dl-model
This project is for the midterm assignment which is to deploy a deep learning image classification for flower dataset

I used google colab to train the model. 
First, the user needs to download and store the dataset from kaggle and so he should use the code in kaggle_lib.py.

Then, The user will copy the colab_model.py file and run it in google colab. This code will train the models one for each needed feature (custom CNN - VGG16 feature extraction - VGG16 fine tuning).

Then, after finishing the training, the user can download the models files and store them and use them.

I have already trained the models and so the user can just download them from the v1.0.1 tag from Github.

The project_file.py contains the main code of the training and the models.

once the user downloads everything and then run the main file, he should get the result of testing the 3 models on 5 images and all of them should give the right results in all cases.


The training results:

1. scratch - validation accuracy of 79.5% - 62 epoches - Loss: 11.3426
2. feature extractor - validation accuracy of 92% - 55 epoches - Loss: 22.0508
3. Fine Tuning - first: validation accuracy of 34% - then I unfrooze 16 layers and got validation accuracy of 40% at heights - now, 24 layers and got 71.5% validation accuracy in 47 epochs and test accuracy of 60%.

Before, I make the file, I have trained my self on using VGG16 on fine tuning which can be seen in VGG16_model.py file. I also did the same using another model ResNet50 on the resNet50_model.py file.

Then, I started with learning how to make a custom CNN.

I learnt about the difference between feature extraction and fine tuning. 

Then, I starting writing the code. There were some problems that happened specially at the training in fine-tuning as I had to unfreeze 24 layers to get these results.

There were some stuff that I added, such as early-stopping to stop the model when it no longer learns.
I also used the save model to save the best model made at all cases.
Added predict method to use the models.
lastly, I added a method to show the confusion matrix of the model after training. you can check them inside the conf_matrix folder.

You should download the flower dataset and copy the path and paste it in data_path variable, also change the dash lines into this "/".

You can download the entire thing from github. Then, you download the models from the v1.0.v tag and store them in a folder called models then, use the main.py file.
Or, you can use the kaggle_lib and colab_model in google colab to create new models.

lastly, to run the model, you can just run the run_model.bat file which will then run the main file without the need of opening the files in vscode.

This project was completed as a part of the SE3508 Introduction to Artifitial Intelligence course, instructed by Dr. Selim Yilmaz, Department of Software Engineering at Mugla Sitki Kocman University, 2025.

Note: This repository must not be used by students in the same faculty in future years - whether partially of fully - as their own submission. Any form of code reuse without proper modification and original contribution will be considered by the instructor a violation of academic integrity policies.