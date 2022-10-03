import pandas as pd
from datetime import timedelta, datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import f_regression 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from scipy import stats



from sklearn.metrics import mean_squared_error
from math import sqrt

from pandas.plotting import register_matplotlib_converters

from statsmodels.tsa.api import Holt





def get_airbnb_data():
    ''' This imports the CSV that was downloaded from 
    'https://www.kaggle.com/datasets/mysarahmadbhat/airbnb-listings-reviews' 
    it is not in standard coding so it had to be converted using the recommended method'''
    
    df = df = pd.read_csv('Listings.csv',encoding='ISO-8859-1')
    return df


def wrangle_data(df):
    ''' This drops columns from 32 to 4, keeps only properties with less than 6 rooms 
    and that are less than $1001 '''
    df = df.drop(['name',
              'host_id',
              'host_since',
              'host_location',
              'host_response_time',
              'host_response_rate',
              'host_acceptance_rate',
              'host_is_superhost',
              'host_total_listings_count',
              'host_has_profile_pic',
              'host_identity_verified',
              'neighbourhood',
              'latitude',
              'longitude',
              'property_type',
              'room_type',
              'accommodates',
              'district',
              'minimum_nights',
              'maximum_nights',
              'review_scores_rating',
              'review_scores_accuracy',
              'review_scores_cleanliness',
              'review_scores_checkin',
              'review_scores_communication',
              'review_scores_location',
              'review_scores_value',
              'instant_bookable' 
               ], axis=1)

    df=df.dropna()
    #this is a complete mess had to create dummies 
    dummies = df["amenities"].str.get_dummies(",")
    #jesus there was 2500 columns so had to get the sum
    dummies['total_amenities'] = dummies.sum(axis=1, numeric_only= True)
    #took the total made the column the df dropping the other 2500 columns
    dummies=dummies.total_amenities
    #joined dummies to df
    df = df.join(dummies)
    #dropped amenities column now that i created sum of total per property
    df = df.drop(['amenities'], axis=1)
    df = df[df['bedrooms'] < 6]
    df = df[df['price'] < 1001]
    return df

def split_data(df):
    '''
    Takes in a dataframe and target (as a string). Returns train, validate, and test subset 
    dataframes with the .2/.8 and .25/.75 splits to create a final .2/.2/.6 split between datasets
    '''
    # split the data into train and test. 
    train, test = train_test_split(df, test_size = .2, random_state=123)
    
    # split the train data into train and validate
    train, validate = train_test_split(train, test_size = .25, random_state=123)
    
    return train, validate, test

def vis_data(train):
    ''' makes visuals for people'''
    
    num_vars = ['bedrooms', 'price', 'total_amenities']
    cat_vars = ['city']

    for col in cat_vars:
    
   
        plt.figure(figsize=(24,12))
        sns.set(font_scale=2)
        sns.countplot(x=col, data=train)
   
        plt.show()
        plt.figure(figsize = (24, 12))
        ax = sns.barplot(x='city', y='price', data=train)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    
    for col in num_vars:
        plt.figure(figsize = (24, 12))
        train[col].hist()
        plt.title(col+' Distribution')
        plt.show()
        
    return

def vis_price_city(train):
    '''shows why we dont use city after a point'''
    plt.figure(figsize = (24, 12))
    ax = sns.barplot(x='city', y='price', data=train)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.show()

def model_(y_train):
    '''makes me some models, while not pretty I need at least 7'''
    
    plt.hist(y_train.price, color='blue', alpha=.5, label="Actual Price")
    plt.hist(y_train.price_pred_mean, bins=1, color='red', alpha=.5, 
         label="Predicted Price - Mean")
    plt.hist(y_train.price_pred_median, bins=1, color='orange', alpha=.5,
         label="Predicted Price - Median")
    plt.xlabel("Price ")
    plt.ylabel("Number of Properties")
    plt.legend()
    plt.show()
    return

def eval_models(y_train, y_validate, X_train, X_validate, X_test):
    ''' this creates the models for prediction it does fairly well in my opinion'''
    
    # create the model object
    lm = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(X_train, y_train.price)

    # predict train
    y_train['price_pred_lm'] = lm.predict(X_train)

    # predict validate
    y_validate['price_pred_lm'] = lm.predict(X_validate)

    # Getting rid of the negative predicted value
    replace_lm = y_validate['price_pred_lm'].min()
    replace_lm_avg = y_validate['price_pred_lm'].mean()
    y_validate['price_pred_lm'] = y_validate['price_pred_lm'].replace(replace_lm, replace_lm_avg)

    # create the model object
    lars = LassoLars(alpha=1.0)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lars.fit(X_train, y_train.price)

    # predict train
    y_train['price_pred_lars'] = lars.predict(X_train)

    # predict validate
    y_validate['price_pred_lars'] = lars.predict(X_validate)

    # Getting rid of the negative predicted value
    replace_lars = y_validate['price_pred_lars'].min()
    replace_lars_avg = y_validate['price_pred_lars'].mean()
    y_validate['price_pred_lars'] = y_validate['price_pred_lars'].replace(replace_lars, replace_lars_avg)

    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate)
    X_test_degree2 = pf.transform(X_test)

    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.price)

    # predict train
    y_train['price_pred_lm2'] = lm2.predict(X_train_degree2)

    # predict validate
    y_validate['price_pred_lm2'] = lm2.predict(X_validate_degree2)

    # Getting rid of the negative predicted value
    replace_lm2 = y_validate['price_pred_lm2'].min()
    replace_lm2_avg = y_validate['price_pred_lm2'].mode()
    y_validate['price_pred_lm2'] = y_validate['price_pred_lm2'].replace(replace_lm2, replace_lm2_avg[0])

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.price, y_train.price_pred_lars)**(1/2)
    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.price, y_validate.price_pred_lars)**(1/2)
    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", round(rmse_train, 2), 
        "\nValidation/Out-of-Sample: ", round(rmse_validate, 2))
    print("R2 Value:", round(r2_score(y_train.price, y_train.price_pred_lars), 2))
    print('-----------------------------------------------')
    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.price, y_train.price_pred_lm)**(1/2)
    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.price, y_validate.price_pred_lm)**(1/2)
    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", round(rmse_train, 2), 
        "\nValidation/Out-of-Sample: ", round(rmse_validate, 2))
    print("R2 Value:", round(r2_score(y_train.price, y_train.price_pred_lm), 2))
    print('-----------------------------------------------')
    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.price, y_train.price_pred_lm2)**(1/2)
    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.price, y_validate.price_pred_lm2)**(1/2)
    print("RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: ", round(rmse_train, 2), 
        "\nValidation/Out-of-Sample: ", round(rmse_validate, 2))
    print("R2 Value:", round(r2_score(y_train.price, y_train.price_pred_lm2), 2))
    
def test_laso(X_test,y_test):
    '''best working model'''
    
# create the model object
    lars = LassoLars(alpha=1.0)

# fit the model to our training data. We must specify the column in y_train, 
# since we have converted it to a dataframe from a series! 
    lars.fit(X_test, y_test.price)

# predict train
    y_test['price_pred_lars'] = lars.predict(X_test)

# evaluate: rmse
    rmse_test = mean_squared_error(y_test.price, y_test.price_pred_lars)**(1/2)


    print("RMSE for Lasso + Lars\nTest/In-Sample: ", rmse_test)

def baseline_RMSE(y_train, y_validate):
    '''creates baseline rmse'''
    # 1. Predict price_value_pred_mean
    price_pred_mean = y_train['price'].mean()
    y_train['price_pred_mean'] = price_pred_mean
    y_validate['price_pred_mean'] = price_pred_mean

    # 2. RMSE of price_value_pred_mean
    rmse_train = mean_squared_error(y_train.price, y_train.price_pred_mean)**(1/2)
    rmse_validate = mean_squared_error(y_validate.price, y_validate.price_pred_mean)**(1/2)

    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))
    
def pearsonr_corr_explore_plot(train,num1,num2):
    ''' 
    takes in a dataframe and two series and runs a pearsonr test to determine if there's a correlation between the features
    '''
    ## putting price value and pricees yearly into a pearsonr and then graphing it for a visual result as a result
    ## of the heat map above highlighting a good possibility of a relation


    H0 = f"That the distributions underlying the samples of {num1} and {num2} are unrelated"
    Ha = f"That the distributions underlying the samples of {num2} and {num2} are related"
    alpha = .05

    r, p = stats.pearsonr(train[num1],train[num2])

    plt.figure(figsize=(10,6))
    plt.scatter( train[num1], train[num2])
    m, b = np.polyfit(train[num1], train[num2], deg=1)
    plt.plot(train[num1], b + m * train[num1], color="k", lw=2.5,label=f"regression line - f(x)={round(m,5)}x+{round(b,0)}")
    plt.xlabel(num1)
    plt.ylabel(num2)
   

    plt.title(f'Is the correlation value indicative? (r={round(r,1)})', size=16)
    plt.legend()
    plt.show()

    print('r =', r)

    if p > alpha:
        print("\n We fail to reject the null hypothesis (",(H0) , ")",'p=%.5f' % (p))
    else:
        print("\n We reject the null Hypothesis (", '\u0336'.join(H0) + '\u0336' ,")", 'p=%.5f' % (p))

    
  
    return
