# Individual_Project_Airbnb






***
[[Project Description](#project_description)]
[[Project Planning](#planning)]
[[Key Findings](#findings)]
[[Data Dictionary](#dictionary)]
[[Data Acquire and Prep](#wrangle)]
[[Data Exploration](#explore)]
[[Statistical Analysis](#stats)]
[[Modeling](#model)]
[[Conclusion](#conclusion)]
___
## <a name="project_description"></a>Project Description:

****
# Executive Summary
##  Goals:

# The purpose of this project is utilize a database of Airbnb Properties to construct a model that can predict their price. 
The datasource includes the Listing.csv from Kagle. Can that model be beaten and give us a business advantage?

Goal : Build a model using only the features of the property to predict listing price. 

# Question asked
## Is Listing Price related to number of bedrooms
## Is Listing Price related to number of ammenities
## Is Listing Price related to location of listing




## Takeaway:

More rooms and More listed Ammenities does allow for increased price of Airbnb, if listing an Airbnb it would be more profitable to have a lot of rooms and a list everything in the property. 

## Recommendation:

Get more information such as time of year data, quality, property tax value, and possibly begin to jump in to looking at rating and how ratings correlates to all the factors give.  

***
---
[[Back to top](#top)]

***
## <a name="planning"></a>Project Planning: 
a)Create deliverables:
- README
- final_report.ipynb
- working_report.ipynb
- wrangler.py

b) Build functional wrangle.py, explore.py, and model.py files

c) Acquire the data from the Code Up database via the wrangle.acquire functions

d) Prepare and split the data via the wrangle.prepare functions

e) Explore the data and define hypothesis. Run the appropriate statistical tests in order to accept or reject each null hypothesis. Document findings and takeaways.

f) Create a baseline model in predicting listing price and document the RSME.

g) Fit and train models to predict price on the train dataset.

i) Evaluate the models by comparing the train and validation data.

j) Select the best model and evaluate it on the train data.

k) Develop and document all findings, takeaways, recommendations and next steps.


[[Back to top](#top)]

### Project Outline:
Acquire data
Prepare Data
Explore Data
Create Hypothesis
Test Model 
Conclusion

        
### Hypothesis
---
# Hypothesis 1

## $H_0$: Price is independent of the Number of Bedrooms of a listing has

## $H_a$ : Price is dependent of Number of Bedrooms of a listing has

---

# Hypothesis 2

## $H_0$: Price is independent of the Number of Ammenities of a listing has

## $H_a$ : Price is dependent of Number of Ammenities of a listing has

---
~# Hypothesis 3~
#Dropped because of issues with feature

~## $H_0$: Price is independent of the Location of a listing~

~## $H_a$ : Price is dependent of the location of a listing~

---

### Target variable
Price

### Need to haves (Deliverables):
acquire.py
prepare.py
Final_notebook.ipybn
this readme.md


### Nice to haves (With more time):
#### Functions to so no code is shown 



***

## <a name="findings"></a>Key Findings:


While location City of the property does impact price it really is a variable that had to be dropped in order to make a better model.


Bedroom count is related to the price of a Airbnb but there are other factors that were not included in this analysis such as location, time of year, and quality of bedroom. That is because there was not time of year data present with prices and the this created numerous outliers one example is a room in Brasil was going for $625,615
so there was likely a currency issue along a timing issue with this one bedroom shared which was listed for over $600k.

Number of Ammenities is related to the price of an Airbnb but there is no real quality control over this feature, so a lister could list everything in their Airbnb and make the total number of their Airbnb skew the data. 



[[Back to top](#top)]


***

## <a name="dictionary"></a>Data Dictionary  


### Data Used
---
***
| Target | Definition | Data Type |
| ----- | ----- | ----- |
|price|	Listing price (in each country's currency)|int64|
***
---
| Attribute | Definition | Data Type |
| ----- | ----- | ----- |
|listing_id|	Listing ID|int64|
|name|	Listing Name|object|
|host_id|	Host ID|int64|
|host_since|	Date the Host joined Airbnb|object|
|host_location|	Location where the Host is based|object|
|host_response_time|	Estimate of how long the Host takes to respond|object|
|host_response_rate|	Percentage of times the Host responds|float64|
|host_acceptance_rate|	Percentage of times the Host accepts a booking request|float64|
|host_is_superhost|	Binary field to determine if the Host is a Superhost|object|
|host_total_listings_count|	Total listings the Host has in Airbnb|object|
|host_has_profile_pic|	Binary field to determine if the Host has a profile picture|object|
|host_identity_verified|Binary field to determine if the Host has a verified identity|object|
|neighbourhood|	Neighborhood the Listing is in|object|
|district|	District the Listing is in|object|
|city|	City the Listing is in|object|
|latitude|	Listing's latitude|float64|
|longitude|	Listing's longitude|float64|
|property_type|	Type of property for the Listing|object|
|room_type|	Type of room type in Airbnb for the Listing|object|
|accommodates|	Guests the Listing accomodates|int64|
|bedrooms|	Bedrooms in the Listing|float64|
|amenities|	Amenities the Listing includes|object|
|price|	Listing price (in each country's currency)|int64|
|minimum_nights|	Minimum nights per booking|int64|
|maximum_nights|	Maximum nights per booking|int64|
|review_scores_rating|	Listing's overall rating (out of 100)|float64|
|review_scores_accuracy|	Listing's accuracy score based on what's promoted in Airbnb (out of 10)|float64|
|review_scores_cleanliness|	Listing's cleanliness score (out of 10)|float64|
|review_scores_checkin|	Listing's check-in experience score (out of 10)|float64|
|review_scores_communication|	Listing's communication with the Host score (out of 10)|float64|
|review_scores_location|	Listing's location score within the city (out of 10)|float64|
|review_scores_value|	Listing's value score relative to its price (out of 10)|float64|
|instant_bookable|	Binary field to determine if the Listing can be booked instantly|object|
|total_amenities|  Sum of Amenities after transformed to dummies|int64|
***
[[Back to top](#top)]
## <a name="wrangle"></a>Data Acquisition and Preparation
[[Back to top](#top)]

![]()


### Wrangle steps: 
Try to Make pretty pictures
Repeat until you get something you understand.

*********************

## <a name="explore"></a>Data Exploration:
[[Back to top](#top)]
- Python files used for exploration:

    - explore.py



### Takeaways from exploration:
Need to drop City because correlation between Price and Location is hard to really account for. 

***

## <a name="stats"></a>Statistical Analysis
[[Back to top](#top)]


# Hypothesis 1:


## alpha = .05

## Pearson R test

## $H_0$: Price is independent of the Number of Bedrooms of a listing has

## $H_a$ : Price is dependent of Number of Bedrooms of a listing has


- alpha = 1 - confidence, therefore alpha is 0.05


#### Results:
We reject the null hypothesis.

#### Summary:
While I still do not fully grasp this process it was completed
***


# Hypothesis 2:


## alpha = .05

## Pearson R test

## $H_0$: Price is independent of the Number of Ammenities of a listing has

## $H_a$ : Price is dependent of Number of Ammenities of a listing has
#### Results:
We reject the null hypothesis.


#### Summary:
While I still do not fully grasp this process it was completed

***

## <a name="model"></a>Modeling:
[[Back to top](#top)]

### Model Preparation:

### Baseline
    
- Baseline Results: 
Our baseline accuracy in all cases on the Dataset is :

RRMSE using Mean
Train/In-Sample:  235.4 
Validate/Out-of-Sample:  237.29

- Selected features to input into models:
    - features = Total_amenities, Bedrooms

***

### Models and R<sup>2</sup> Values:
- Will run the following regression models:

RMSE for Lasso + Lars
Training/In-Sample:  235.4 
Validation/Out-of-Sample:  237.29
R2 Value: 0.0
---
RMSE for OLS using LinearRegression
Training/In-Sample:  229.0 
Validation/Out-of-Sample:  230.77
R2 Value: 0.05
---
RMSE for Polynomial Model, degrees=2
Training/In-Sample:  228.95 
Validation/Out-of-Sample:  230.79
R2 Value: 0.05
---
- Other indicators of model performance with breif defiition and why it's important:

    


# Selecting the Best Model:
Lasso + Lars was selected but it is likely overfit

RMSE for Lasso + Lars
Test/In-Sample:  236.706649483469





***

## <a name="conclusion"></a>Conclusion:
While location City of the property does impact price it really is a variable that had to be dropped in order to make a better model.

Bedroom count is related to the price of a Airbnb but there are other factors that were not included in this analysis such as location, time of year, and quality of bedroom. That is because there was not time of year data present with prices and the this created numerous outliers one example is a room in Brasil was going for $625,615.

So there was likely a currency issue along a timing issue with this one bedroom shared which was listed for over $600k.

Number of Ammenities is related to the price of an Airbnb but there is no real quality control over this feature, so a lister could list everything in their Airbnb and make the total number of their Airbnb skew the data.

Takeaway:
More rooms and More listed Ammenities does allow for increased price of Airbnb, if listing an Airbnb it would be more profitable to have a lot of rooms and a list everything in the property.

Recommendation:
Get more information such as time of year data, quality, property tax value, and possibly begin to jump in to looking at rating and how ratings correlates to all the factors give.

Next steps are to refine functions clean up notebook and then run test on other features.

---
### Steps to Reproduce
Your readme should include useful and adequate instructions for reproducing your analysis and final report.


1)You will need download the Listing.csv from https://www.kaggle.com/datasets/mysarahmadbhat/airbnb-listings-reviews

2)clone my repo (including the wrangler.py) (confirm .gitignore is hiding your env.py file)

3)libraries used are pandas, matplotlib, seaborn, numpy, sklearn,scipy, math.

4)you should be able to create the same results.


[[Back to top](#top)]
