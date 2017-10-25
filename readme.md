# Two Sigma & Renthop Rental Interest Classification

https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries

My first real Kaggle competiton - I ranked 254th out of 2488 competitors (kaggle lyang24)

small data set with 3 classes evaluated on logloss.


![alt text](https://github.com/lyang24/KaggleRentalInquries/blob/master/ranking.PNG?raw=true "Description goes here")


### My main features are:
* basic numerics features
* date time features
* high cardinal features encoded
* geo clustering features
* features from description

I stacked XGBOOST and LIGHTGBM based on parameter tuning scores.

see my notebook for more details.

### Improvements:
* stack with linear models (Neural Networks, Logistic Regression ...)
* Study and generate features from renthop's hopscore
* Generate more features based on manager id

