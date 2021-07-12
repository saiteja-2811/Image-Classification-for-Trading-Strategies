# Trading Strategies - Image Classification

This repository contains all the files of the stock prediction project, related to my Machine Learning class.

![Trading](images/trading.png)
# Table of Contents
- [Overview](#overview)
- [Introduction](#introduction)
- [Trading Strategies](#trading-strategies)
  - [1. Bollinger Bands](#1-bollinger-bands)
    - [Logical Background](#logical-background---bb)
    - [Application](#application---bb)
  - [2. Relative Strength Index](#2-relative-strength-index)
    - [Logical Background](#logical-background---rsi)
    - [Application](#application---rsi)
- [Methodology](#methodology)
  - [1. Image Generation](#1image-generation)
    - [Data Collection](#data-collection)
    - [Data Preparation](#data-preparation)
    - [Signal Generation](#signal-generation)
      - [Buy vs Sell](#buy-vs-sell)
      - [Buy vs Sell/Hold](#buy-vs-sellhold)
  - [2. Feature Engineering](#2feature-engineering)
    - [Convolution](#convolution)
    - [Max Pooling](#max-pooling)
  - [3. Modeling](#3modeling)
    - [Train & Validation Split](#train--validation-split)
    - [Model Architecture](#model-architecture)    
    - [Model Selection](#performance-metrics)
- [Conclusion](#conclusion)
  
## Overview
Trading decisions have always been subjective i.e., there is `no definitive answer` for a decision when a `trader buys/sells/holds` a stock. This means that the subject-matter experts are in high demand, and we need to `invest in automation` to minimize human intervention. So let us explore the feasibility of building a new system that can `replicate the way humans trade`.

## Introduction
We will be analyzing the S&P 500 Global index data from `03-Jan-1983` till `18-Jun-2021`.We will use this data and `generate images and labels (buy/sell/hold)` using some popular trading strategies. Then we use these `images and train the model` to classify the images and `compare that with the labels` that we have generated for performance evaluation.

## Trading Strategies
### 1. Bollinger Bands
Let us check for the ```missing values``` for each variable first, and then we will impute them with the appropriate methods. 
![Missing](stockpred/images/Missing_Train.PNG)

#### Logical Background - BB
- Drop ```cabin``` variable due to high missing percentage
- Drop ```fare``` because of **high correlation** with ```PClass```
- Drop ```Ticket``` variable due to low value addition
#### Application - BB
We have already dropped the ```cabin``` variable, so we have to impute the ```Age``` and ```Embarked``` variables. Age is a continuous variable and its distribution is below. From the figure we can say that ```Age``` is skewed.
![Age](images/Age_Dist.PNG)

I have used the median value of ```Age``` for a `passenger class` and `gender` to impute the missing values.

### 2. Relative strength index
This is an import aspect of the methodology, because this is where the business intuition and domain expertise come in. And we all know how crucial these two are to make better predictions and to interpret the results of the model.  

#### Logical Background - RSI
Though this variable `might not look important` at first, but we can extract some `hidden information` from this i.e., we can get the `Title` of each passenger and analyze if some `titles` have high survival probability. We have `Capt`, `Col`, `Don`, `Dr`, `Jonkheer`, `Lady`, `Major`, `Rev`, `Sir`, `the Countess`, `Miss`, and `Mrs`
#### Application - RSI
I have now `grouped some categories` together because they have the `same event rates` i.e., `same probability of survival`.

## Methodology
We have `cleaned` the data and `derived` some variables so that we can make better predictions. So let us `predict` now. But we need to follow some steps to make a robust model and `avoid over-fitting` the data.

### 1.Image Generation
The training data will be `randomly` split into `70:30` ratio into `training` and `validation` datasets. We now use the first one to train our model, and the validation data to validate our model's accuracy.

#### Data Collection
I have explored `six` different techniques to train the model. Click on the links for literature review.

#### Data Preparation
The performance of the above models can be judged based on the validation dataset. The results are below, so my best model is Light GBM.

#### Signal Generation
We now have a model, trained and validated. Recollect that we have been provided a `test` dataset to make predictions for the `future`. So we perform the same `data-preprocessing` steps on this as well and predict the `Survived` column. But, for this we can `train` our model on the `whole training` dataset and again and use that model so that we have more data to train our model.

##### Buy vs Sell

##### Buy vs Sell/Hold

### 2.Feature Engineering

#### Convolution

#### Max Pooling

#### Model Architecture

### 3.Modeling

#### Train & Validation Split

#### Performance Metrics 

## Conclusion



