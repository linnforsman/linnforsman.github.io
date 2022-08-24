---
layout: post
title: Using Spark to Predict Churn with Insight Data Science
---

> [Read article on medium.com](https://linnforsman.medium.com/predicting-music-streaming-service-churn-with-pyspark-78825dfa491c)

# {{page.title}}


## Problem Introduction
Customer churn is a crucial business problem because it is a vital source of revenue for many business models. Predicting customer churn is a critical step in the business process to prioritize retention programs on customers who are likely to churn. In this project, I used PySpark to analyze and predict churn based on the 12GB customer activity dataset of the fictitious music streaming service Sparkify. Firstly, I used a small subset of the entire dataset to perform exploratory data analysis and prototype machine learning models: logistic regression, random forest, and gradient-boosted tree model for further tuning given its superior performance. In the final section, I wrote a conclusion and reflect on what could be further improved.
### Exploratory Data Analysis
__Overview of dataset__
The dataset logs user demographic information (e.g., user name, gender, location) and activity (e.g., song listened, event type, a device used) at individual timestamps. There are 286000 rows in the subset that's assumed representative of the entire dataset. Missing values in `userId` were dropped because they represent users in the middle of or before sign-in or registration.
``` 
 | - userId: ID of user
 | - gender: gender of user
 | - level: level of user (free vs. paid)
 | - location: location of user (e.g. Bakersfield, CA)
 | - registration: registration time of user
| - page: type of page visit event (e.g. add a friend, listen to a song)
 | - ts: timestamp of event
 | - song: name of song
 | - artist: artist of song
 | - length: length of song
 | - userAgent: device used (e.g. Mozilla/5.0 Macintosh…) | - sessionId: ID of current session
 | - itemInSession: order of event in current session
 | - …
 ```
#### __Define churn__
Churn was defined as users who have page = "Cancellation Confirmation".
```
flag_churn_event = udf(lambda x: 1 if x == "Cancellation Confirmation" else 0, IntegerType())
df.withColumn("churned", flag_churn_event("page"))\
    .groupBy('userId').agg(max('churned'))\
    .withColumnRenamed('max(churned)', 'churn')
```
There are 52 (23%) churn users and 173 (77%) non-churn users in this subset.
#### __Compare the behavior of churn vs. non-churn users.__
__Hour of the day__: The number of users is relatively consistent at different hours of the day.
__Day of the week__: There are slightly fewer users on the weekend than during weekdays.
__User level__: There are more free users than paid users, and the free users have a slightly higher churn rate than paid users. While free users have a larger population, they contribute much fewer page visits than paid users, suggesting that paid users are more engaged.
__Page events__: The page visits are dominated by listening to songs ("NextSong", note the logarithm scale in the y-axis). In general, churn users have less engagement than no-churn users. Churn users have generally added fewer friends, added fewer playlists, requested less help, listened to fewer songs, and even encountered fewer errors. Although churn users account for a small fraction of the total number of users, they contribute to an almost comparable amount of page visits relative to non-churn users, suggesting that the company should NOT forego this cohort.
When examining the proportion of page visits by churn vs. non-churn users, it is interesting that churn users have generally received more advertisements. Users also adjusted settings more frequently, requested more upgrades, and gave more thumbs-downs to non-churn users. This pictures churn users who have received too many commercials and were unsatisfied with the songs.
__Device used__: Not surprisingly, Windows and Mac are the most commonly used devices. Windows and Mac users have about the same proportion of churn. In comparison, almost half of X11 Linux users churn. Assuming the observed patterns are representative of the whole population, this high churn rate may suggest possible issues in the app's user interface on X11 Linux.
## Modeling
### Feature engineering
__Create features per user basis__
To prepare machine learning data, I first re-engineered several features on a per-user basis.
- Latest level of user
```
func_levels = udf(lambda x: 1 if x=="paid" else 0, IntegerType())
levels = df.select(['userId', 'level', 'ts'])\
    .orderBy(desc('ts'))\
    .dropDuplicates(['userId'])\
    .select(['userId', 'level'])\
    .withColumn('level', func_levels('level').cast(IntegerType()))
```
- Time since registration (imputing missing values with mean)
```
time_regi = df.groupBy('userId')\
    .agg(max('ts'), avg('registration'))\
    .withColumn('time_since_regi', (col('max(ts)')-col('avg(registration)'))/lit(1000))

avg_time = time_regi.select(avg('time_since_regi'))\
    .collect()[0]['avg(time_since_regi)']
time_regi = time_regi.fillna(avg_time, subset=['time_since_regi'])\
    .drop(['max(ts)', 'avg(registration)'])
    
```
- Gender of user
```
func_gender = udf(lambda x: 1 if x=="M" else (0 if x=="F" else -1), IntegerType())
gender = df.select(['userId', 'gender'])\
    .dropDuplicates()\
    .withColumn('gender', func_gender('gender'))
```
- The amount of time, number of artists, number of songs, and number of sessions that the user has engaged

```
engagement = df.groupBy('userId')\
    .agg(
         countDistinct('artist').alias('num_artists_dist'), 
         countDistinct('sessionId').alias('num_sessions'),
         countDistinct('song').alias('num_songs_dist'),
         count('song').alias('num_songs'),
         count('page').alias('num_events'),
         Fsum('length').alias('tot_length')
    )
```

- Mean and standard deviation of the number of songs listened to per artist

```
per_artist = df.filter(~df['artist'].isNull())\
    .groupBy(['userId', 'artist'])\
    .agg(count('song').alias('num_songs'))\
    .groupBy('userId')\
    .agg(avg(col('num_songs')).alias('avg_songs_per_artist'),
         stddev(col('num_songs')).alias('std_songs_per_artist')
    )\
    .fillna(0)
``` 
- Mean and standard deviation of the number of songs listened to per session and the time spent per session


```
per_session = df.groupBy(['userId', 'sessionId'])\
    .agg(
         max('ts'), 
         min('ts'), 
         count('song').alias('num_songs')
    )\
    .withColumn('time', (col('max(ts)')-col('min(ts)'))/lit(1000))\
    .groupBy('userId')\
    .agg(
         stddev(col('time')).alias('std_time_per_session'), 
         avg(col('time')).alias('avg_time_per_session'),
         stddev(col('num_songs')).alias('std_songs_per_session'),
         avg(col('num_songs')).alias('avg_songs_per_session')
    )\
    .fillna(0)
```
- Device used

The raw values of the user's device were in the format: `Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36`. I converted the bulky texts into just the device name, e.g. `Macintosh`.

```
window = Window.partitionBy("userId")\
    .rowsBetween(
        Window.unboundedPreceding,
        Window.unboundedFollowing
    )
func_agent_device = udf(
    lambda x: "user_agent_"+x.split('(')[1].replace(";", " ").split(" ")[0] if '(' in str(x) else 'user_agent_none', 
    StringType()
)

agents = df.withColumn(
    "userAgent", func_agent_device(col("userAgent"))
    )\
    .groupBy(["userId", 'userAgent'])\
    .agg(count("userAgent").alias("user_agent_usage_count"))\
    .withColumn(
        'total', Fsum(col('user_agent_usage_count')).over(window)
    )\
    .withColumn(
        'user_agent_usage', 
        col('user_agent_usage_count')/col('total')
    )\
    .groupBy("userId").pivot("userAgent").sum("user_agent_usage")\
    .drop('user_agent_none').fillna(0)
```
- Count of each event type
```
pages_to_exclude = ['Cancel', 'Downgrade', 'Cancellation Confirmation', 'Upgrade', 'Submit Registration', 'Login', 'Register']
func_pages = udf(lambda x: "page_"+x.replace(" ", "_").lower())pages = df.filter(~df['page'].isin(pages_to_exclude))\
    .withColumn("page", func_pages(df["page"]))\
    .groupBy(['userId']).pivot("page").agg(count('page'))\
    .fillna(0)\
    .withColumn(
        "page_up_down_ratio", 
        pages["page_thumbs_up"]/(pages['page_thumbs_down']+0.1)
)
```
- Proportion of each event type
```
pages = pages.withColumn(
    'total', sum(pages[coln] for coln in pages.columns if coln not in ['userId', 'page_up_down_ratio'])
)
for coln in pages.columns:
    if coln not in ['userId', 'total', 'page_up_down_ratio']:
        new_col_name = coln[0:5]+'frac_'+coln[5:]
        pages = pages.withColumn(
            new_col_name, pages[coln] / pages['total']
        )
pages = pages.drop('total')
```
To this end, I have 64 feature columns and 1 label column for all the users.
```
dataset = churn.join(levels, ['userId'])\
    .join(time_regi, ['userId'])\
    .join(gender, ['userId'])\
    .join(engagement, ['userId'])\
    .join(per_artist, ['userId'])\
    .join(per_session, ['userId'])\
    .join(agents, ['userId'])\
    .join(pages, ['userId'])\
    .join(locations, ['userId'])
```
### __Check feature correlations__
Second, I assessed the correlation between each pair of features. To get a succinct feature set, I removed one feature from each pair of strongly correlated features (correlation coefficient > 0.8). To this end, I have 43 feature columns and 1 label column for all the users.
Heatmap graph showing the correlation between features. Intense color represents a stronger correlation.

#### __Feature transformation__
Third, I applied log transformation on skewed features to change their distributions closer to normal.
```
for col_name in col_names:
    if col_name in columns_to_transform:
        dataset = dataset.withColumn(
            col_name, log(dataset[col_name]+1)
        )
```
### __Machine learning__
The goal of the machine learning model is to predict churn (label=1) vs. non-churn (label=0) based on the features I re-engineered in step 2.
#### __Evaluation metric__
Because churn users only represent 23% of all the users, I chose the F1 score as the metric to evaluate model performance instead of the accuracy score.
Briefly, accuracy is defined as:
*accuracy = (number of correct predictions) / (total number of predictions)*
If accuracy is used as the evaluation metric, a "naive model" that predicts "no-churn" will have reasonably good accuracy (77%) but abysmal performance because it is never able to identify a churn. So accuracy wouldn't be an appropriate metric to use here.
In comparison, F1 score is defined as:
*F1 = 2*precision*recall / (precision + recall)*
Where precision is the number of correctly identified churns out of total identified churns, and recall is the number of correctly identified churns out of total real churns. When predicting a churn, precision ensures it is a churn, whereas recall aims not to miss any real churn. F1, which averages between precision and recall, makes more sense with imbalanced classes.
### __Spark pipeline__
After the train-test split, I created a PySpark machine learning pipeline that consists of:
- VectorAssembler, which vectorizes input features
- MaxAbsScaler, which re-scales each feature to the range [-1, 1]
- A classifier of choice

#### __Initial model evaluation on data subset__
I compared the model performance of the following classifiers by using their default hyperparameters.
Naive predictor, which always predicts no-churn
Logistic regression
Random forest
Gradient-boosted Tree

Naive model:
```
+------+--------+
|    f1|accuracy|
+------+--------+
|0.6684|  0.7689|
+------+--------+
```
Logistic Regression:
```
+----------+--------+---------+-------+--------+
|train_time|f1_train|acc_train|f1_test|acc_test|
+----------+--------+---------+-------+--------+
| 1448.4415|  0.842 |   0.8534| 0.6804|  0.7059|
+----------+--------+---------+-------+--------+
``` 
Random Forest:
```
+----------+--------+---------+-------+--------+
|train_time|f1_train|acc_train|f1_test|acc_test|
+----------+--------+---------+-------+--------+
|  689.0333|  0.9339|   0.9372| 0.6479|  0.7353|
+----------+--------+---------+-------+--------+
```
Gradient-Boosted Tree:
```
+----------+--------+---------+-------+--------+
|train_time|f1_train|acc_train|f1_test|acc_test|
+----------+--------+---------+-------+--------+
| 2025.4227|     1.0|      1.0| 0.6868|  0.6765|
+----------+--------+---------+-------+--------+
```
The naive model sets a baseline of model performance, F1 = 0.67 and accuracy = 0.77. As expected, the three machine learning classifiers can perform better than the naive model on the training set. Among others, Random Forest takes the least time to train, achieves second-best performance on the training set (F1 = 0.93, accuracy = 0.94), and achieves the best performance on the testing set (F1 = 0.65 accuracy = 0.74). Gradient-Boosted Tree takes longest time to train, achieves best performance on the training set (F1 = 1.0, accuracy = 1.0), and achieves second-best performance on the testing set (F1 = 0.69, accuracy = 0.68).
These scores are pretty promising, given that the hyperparameters have not been tuned yet. Gradient-Boosted Tree seems to have the most predictive power (least bias), although it tends to overfit and does not generalize very well (high variance).
Since the analysis conducted here could scale and be trained on the entire dataset, provided the code be deployed on a cluster capable of handling the computations necessary. Given that the complete dataset will provide more training data to help resolve overfitting, the Gradient-Boosted Tree model's hyperparameters could be trained further.


### Hyperparameter tuning
I tuned the hyperparameters of the Gradient-Boosted Tree classifier.

```
+-------+--------+
|f1_test|acc_test|
+-------+--------+
| 0.8229|  0.8387|
+-------+--------+
```
The most important features in predicting churn include the time since the user registered, the number of advertisements that the users have encountered, the number of thumbs up and thumbs down by the user, and the amount of user engagement (e.g. listening to songs, adding friends).
Horizontal bar graph showing top 20 most important features.


### __Conclusion/Reflection/Possible Improvements__
Sparkify should reduce churn because churn users contribute significantly to total usage despite a small proportion of the total population.
The machine learning model can predict churn reasonably well, which will help "Sparkify" prioritize retaining users who have the highest probability of churn. The model performance can be further improved by tuning broader ranges of hyperparameters and additional engineering features such as distributions of user activity by weekday.
Churns relate to users who have received more advertisements, disliked songs more often than liked, and registered more lately.
This analysis would gain from leveraging the full dataset and being deployed on a Spark cluster in the cloud. Grid search is a particularly computation expensive operation, but with larger resources and more time, a more extensive search over a larger dataset and hyperparameter space could be conducted to tune the model further and likely improve overall accuracy.
These churn characteristics will also help the fictitious music streaming service Sparkify determine what actions to act in. e.g., 1. Reducing the number of commercials for identified users. 2. Improving recommendation algorithms to recommend songs and friends that engage the users more 3. Implement a concise tutorial immediately after user registration to make it easier for users to engage. Sparkify will need A/B tests to assess profits vs. costs by each action statistically.
