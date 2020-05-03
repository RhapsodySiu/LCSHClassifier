# LCSH Classifier online DEMO

2020-02-01 - First version. Can perform single subject heading prediction among 60 classes.
2020-05-03 - Final version. Provide result of the multiclass classifiers (portrait, district and keyword) as well as those in multilabel classifiers (HKLCSH60/275/570) 

This flask web app allows user to upload an image each time and preview the prediction result from the project classifiers.

## dependency
The app is tested on Python3.6 only. `requirements.txt` lists the modules required to run the app.
The models can be download from [Google Drive](https://drive.google.com/drive/folders/1mCvi9DSeUi9ya-7Osia6hAohE8qyXXLp?usp=sharing). Put them in the `model` folder of this directory.

## start
In the console,
```
flask run 
```


