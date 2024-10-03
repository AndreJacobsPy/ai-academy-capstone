# Student Success Analysis Using Machine Learning

## Introduction

In today's world, Higher Education can be the key to an impactful and
fulfilling career. Whether it is in a typical University setting or
other modern mediums.

Higher education institutions collect tons of data about their students
and their backgrounds. Our team believes in the potential of this data
and how it can be used to improve the student experience. Academic institutions
have a large impact on the lives of young people and deserve the chance 
to understand their students better. Through this better understanding
Universities will have guidance to help student who are in need. In many 
cases students who dropout are experiencing difficult circumstances and 
with the correct guidance and support they can be successful.

## Challenge

With around 40% of undergraduate student leaving universities and colleges
every year[^1], our team believes it is crucial that Higher Education institutions
do what they can to reverse this trend. On average students who do not graduate are
expected to earn $21,000 less than college graduates.

![image](https://github.com/user-attachments/assets/c814dd34-a4c6-4116-be24-5ca10fa17e54)

The data we will be using collects information about prior student experience, background
of students immediate family, demographic information, and basic economic indicators.
Our deliverable will help institutions to know their students better in regard to their
likely outcome and be able to provide resources to those in need of assistance, to counter 
high dropout rates.

[^1]: Article about dropout rates [support link](https://research.com/universities-colleges/college-dropout-rates)

## Prior Research

There has been a paper published[^2] using this same dataset. This paper was more 
focused on feature importance than predictions. They used Random Forests and XGBoost to find
feature importance. They found that *Curricular units 2nd sem (approved)* was the most
impactful feature. Their research also showed that Course a student was taking and Tuition
Fees up to date were relevant to a student's result.

![image](https://github.com/user-attachments/assets/34ebde19-9d00-4cc3-8a61-a1bc5bd29a79)

In a second paper[^3], there was a more predictive focus on Dropout prediction. They used
Linear Discriminant Analysis for dimensionality reduction, as well as, Random Forest and 
Support Vector Machines for prediction. Their data included Gender, Age Range, Additional
Learning Requirements, and Course Credits. This paper used a Binary outcome for dropout.


This second paper uses High School data, which could have a different population but, 
their methods remain relevant. A large challenge that both of these studies neglected was the
appearance of fields in the data that could bias a model used in production. Features such as
Nationality, Gender, Age, and background of parents could make model discriminant.

[^2]: Published paper using same data [support link](https://www.mdpi.com/journal/data)
[^3]: Published paper with other data for prediction [support link](https://link.springer.com/chapter/10.1007/978-3-030-52237-7_11)

## Methods & Results

### Model evaluation and Strategy
Our strategy to evaluating the problem and create a model was a tree pronged approach. The team wanted to see 
how using different models would improve accuracy of prediction as well as decrease our overall evaluation metric. 
The models we decided to use were decision tree, random forest and XG Boost

#### Decision Tree

When working with tree based models, decision trees are the simplest to understand. They are a good starting point
because it is very clear why they output certain results and are very interpretable by non-technical people.
They do have a tendency to overfit the data, so we will need to be careful with the hyperparameters we use.
In my own experience, models such as Random Forest and XGBoost are more accurate because they are able to model 
more complex data.

#### XGBoost

XGBoost is a very powerful model that is able to model very complex data. It is a very popular model in the 
data science world. It is an ensemble model that uses a combination of weak learners to create a strong learner, 
similarly to a Random Forest. This model uses boosting rather than bagging to create the ensemble. Boosting is
a method that uses the errors of the previous model to improve the next model. This is done by increasing the
weights of the errors of the previous model. This model is very powerful and is able to model very complex data.

#### Random Forest

Random Forest is another ensemble model that is able to model very complex data. It is a very popular model in the
data science world. It is an ensemble model that uses a combination of weak learners to create a strong learner.
This model uses bagging rather than boosting to create the ensemble. Bagging is a method that samples data with 
replacement and usually only uses subsets of columns to create the model. This model is very powerful and is able
to model very complex data.

#### Final Model

Our final model was a Random Forest model. We chose this model because it was the most accurate model that we
created. This model had the following metrics:

- **Accuracy**: 0.86
- **Recall**: 0.73

To achieve these scores below is the hyperparameters we used:

```python
from sklearn.ensemble import RandomForestClassifier
forest_final = RandomForestClassifier(
    random_state=42, class_weight={0:7, 1:3}
)
```

We used random state to help make trees reproducible and class weight to help improve recall.
For this model, recall is our most important metric because, we want to focus on finding 
at risk students early. It is not as bad of a consequence to try to assist a student who is 
not at risk for dropping out. This model will help provide some focus to Universities on 
which students to help and when to step in.

## Impact

Our model will be able to predict the likelihood of a student dropping out of a course. This will allow Universities
to provide resources to students who are at risk of dropping out. This will help to increase the graduation rate of
students and help to improve the overall success of students.

Having the ability to help students early enough that, they can be successful is a huge benefit to the students and
the Universities. This will help to increase the graduation rate of students and help to improve the overall success
of their students. Large student debt is a serious concern in our country and students graduating from school is crucial
to the success of our economy. Universities being able to help students with topics such as: financial aid, tutoring,
and mental health will allow these students to be successful, in turn, giving back and becoming a valuable part of
an alumni network that can help future students.

Many universities have a large amount of data on their students but, fail to make use of that unorganized information.
It is time to use that power to help the future generations of students and to help end the cycle of student debt and
economic struggles.
