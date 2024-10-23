
#  ML.NET Intro and Sample project

## ML.NET​

-   Machine  learning framework for .NET Developers​ 
-   Open source, cross platform​
-   Released in 2020
-   Current version 3.0.1 (4.0 preview,  [final version due by November 12, 2024](https://github.com/dotnet/machinelearning/milestone/16))

## What can ML.NET do?​

### Consume pre-trained models
-   TensorFlow​
-   Open Neutral Network Exchange
### Train custom models
-   Classification​
-   Vision​
-   Decision​
-   Analytics & Insights
## Demo project
### NOTE: 
**Following information is just addition to code itself you can find in this repo and should read through**
### Summary
- Learn to recognize sentiment from ​ČSFD movie reviews
- Get rating based on the provided review

![CSDF](https://i.ibb.co/z6LXR9t/csfd.png)

- Reviews contains review text and rating with values **odpad** (trash) as lowest and then **1** through **5 stars**, so 6 possible values
- Our goal is to create model that will learn to match review text to final rating value
### Prerequisites
- Enough scrapped reviews (with omitted diacritical marks) from csfd.cz website in tsv format where in first column contains review text and second one review value in numeric format 0-5 (tested with 50k reviews) **[Dataset is not included in this repository]**
### Machine learning training workflow
![Machine learning training workflow](https://i.ibb.co/mN8Vg1w/mlworkflow.png)
#### Demo project:
- with simplified version of our dataset by modifying rating values so we have only two possible values 0 for negative, 1 for positive (by taking 0-2 values as negative, 3-5 as positive)
- project in provided solution **Nalejvarna.MLdotnetIntro.TextClass.Binary**

![Binary output](https://i.ibb.co/94FYgf1/binary.png)

- with original dataset
- project in provided solution **Nalejvarna.MLdotnetIntro.TextClass.Multi**

![Multi output](https://i.ibb.co/SKc1GHQ/multi.png)
## AutoML​
- Automates the process of applying machine learning to data
- Given a task and a dataset, you can run AutoML to iternate over different data transformations, machine learning algorithms, and hyperparameters to select  the  best model
### Machine learning training workflow (AutoML)​
![Machine learning training workflow (AutoML)](https://i.ibb.co/58tXPCK/automlworkflow.png)
#### Demo project:
- with simplified version of our dataset by modifying rating values so we have only two possible values 0 for negative, 1 for positive (by taking 0-2 values as negative, 3-5 as positive)
- project in provided solution **Nalejvarna.MLdotnetIntro.AutoML.TextClass.Binary**

![AutoML Binary output](https://i.ibb.co/h7pcj3Q/automlbinary.png)

- with original dataset
- project in provided solution **Nalejvarna.MLdotnetIntro.AutoML.TextClass.Multi**

![AutoML Multi output](https://i.ibb.co/8b2nYkW/automlmulti.png)

#### TIP: you can increase time of AutoML experiment to get better suited algorithm hence better result from the generated model
## ML.NET Ecosystem
![ML.NET Ecosystem](https://i.ibb.co/WfKhhcF/mlnetecosystem.png)
### Model Builder
Model Builder provides GUI in VisualStudio to do what you can do with AutoML in the code, you can access it by opening .mbconfig file. You can find three projects in the repo for using ModelBuilder:
- **Nalejvarna.MLdotnetIntro.ModelBuilder.TextClass.Binary**
- **Nalejvarna.MLdotnetIntro.ModelBuilder.TextClass.Multi**
- **Nalejvarna.MLdotnetIntro.ModelBuilder.ImageClass**
#### Prerequisite:
- [ML.NET Model Builder](https://dotnet.microsoft.com/en-us/apps/machinelearning-ai/ml-dotnet/model-builder)
#### Model Builder demo:

![Model Builder 1](https://i.ibb.co/18qr1RS/mb1.png)
![Model Builder 2](https://i.ibb.co/GC2Y1Pf/mb2.png)
![Model Builder 3](https://i.ibb.co/kgfvSzJ/mb3.png)
![Model Builder 4](https://i.ibb.co/3zM5cpC/mb4.png)
![Model Builder 5](https://i.ibb.co/n3rwX4k/mb5.png)

Last project is actually for Image classification which you can try with following data set: [The Comprehensive Cars](https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/)