# Data-Science-Project

Third year Data science project of Restaurant Revenue Prediction

    import numpy as np
    import pandas as pd
    train_data = pd.read_csv('PATH OF TRAIN FILE IN YOUR MACHINE/train.csv',index_col=0)
    train_data.head()

Descibe data

    train_data.describe()

Data types of train data

    train_data.dtypes

lets check which city has maximum number of restaurants

    train_data["City"].value_counts()
    
check how 'city affects' our revenue feature

    import matplotlib.pyplot as plt

    plt.subplots(figsize=(30,10))
    city_revenue_group = train_data["revenue"].groupby(train_data["City"])
    agg_data = city_revenue_group.sum()

    x_axis = agg_data.index
    y_axis = agg_data

    plt.bar(x_axis,y_axis)
    plt.xlabel("CITY")
    plt.ylabel("REVENUE")
    plt.show()

Istanbul generating high revenue as compared to other cities and then ankara which is nearly 1/4th of istanbul

Now lets check how 'city groups' affects our revenue feature

    city_group_revenue_group = train_data["revenue"].groupby(train_data["City Group"])
    agg_data = city_group_revenue_group.sum()

    x_axis = agg_data.index
    y_axis = agg_data
    plt.bar(x_axis,y_axis)

    plt.xlabel("City Group")
    plt.ylabel("Revenue")
    plt.show()

visualizing remaining features by generating heat map , looking for correlation between them

    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme() 
    fig, ax = plt.subplots(figsize=(50,50)) 
    correlation_matrix = train_data.corr() 
    
    sns.heatmap(correlation_matrix,annot=True,linewidths=.5,ax=ax)

Now finding Variance inflation factor (VIF) of features
it is a measure of the amount of multicollinearity in a set of multiple regression variables.

    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # exluding revenue from VIF calculation because it's variable to be predicted

    features=train_data.loc[:,"P1":"P37"]

    vif_data = pd.DataFrame()

    vif_data["features"] = features.columns
    vif_data["vif"] = [variance_inflation_factor(features.values, i) for i in range(len(features.columns))]
    vif_data = vif_data.sort_values(by=["vif"])
    vif_data

We can see that there exits high multicollinearity in our data.
By Observing the Heat Map we can say that P2(0.19), P6(0.14) and P28(0.16) have a signicant correlation with revenue as compared to others.

#check correlation between P2-revenue,P6-revenue,P28-revenue 

    plt.figure(1)
    plt.xlabel("P2")
    plt.ylabel("revenue")
    plt.scatter(train_data["P2"],train_data["revenue"])

    plt.figure(2)
    plt.xlabel("P6")
    plt.ylabel("revenue")
    plt.scatter(train_data["P6"],train_data["revenue"])

    plt.figure(3)
    plt.xlabel("P28")
    plt.ylabel("revenue")
    plt.scatter(train_data["P28"],train_data["revenue"])
    
As we have seen above "Istanbul" is the only city that has maximum number of restaurants.
Most of the other restaurants have significantly less number of restaurants.
We can't have one hot encoding for each and every city, it will make so many features

So I will divide the city restaurants into different groups
    All cities having more than 3 restaurants will have individual group
    Rest all with be put under the group "other".
    
Now cities with No, of restaurants >3 will be encoded as individual column and rest will be put in other groups

    train_data["City"].value_counts() > 3
    
    city_encodings = pd.get_dummies(train_data[["City"]], prefix = ['City'])
    city_encodings["City_Other"] = 0

    for index, rows in city_encodings.iterrows():
        if (rows["City_İstanbul"] == 0 and rows["City_Ankara"] == 0 and rows["City_İzmir"] == 0 and rows["City_Bursa"] == 0 and rows["City_Samsun"] == 0 and rows["City_Antalya"] == 0 and rows["City_Sakarya"] == 0):
            city_encodings["City_Other"][index] = 1

    city_encodings = city_encodings[["City_İstanbul", "City_Ankara", "City_İzmir", "City_Bursa", "City_Samsun", "City_Antalya", "City_Sakarya", "City_Other"]]
    city_encodings
    
    train_data = pd.merge(train_data, city_encodings, left_index = True, right_index = True)
    train_data.drop(["City"],axis=1,inplace=True)
    train_data.head()
    
After grouping Cities, lets group City Group feature
    
    city_group_encodings = pd.get_dummies(train_data[["City Group"]], prefix = ['City Group'])
    city_group_encodings
    
mergind essentaial data
    
    train_data = pd.merge(train_data, city_group_encodings, left_index = True, right_index = True)
    train_data.drop(["City Group"], axis=1,inplace=True)
    train_data.head()

Encoding  Type_DT, Type_FC, Type_Other

    type_encodings = pd.get_dummies(train_data[["Type"]], prefix = ['Type'])
    type_encodings["Type_Other"] = 0
    for index, rows in type_encodings.iterrows():
        if (rows["Type_DT"] == 0 and rows["Type_FC"] == 0):
            type_encodings["Type_Other"][index] = 1
    type_encodings = type_encodings[["Type_DT","Type_FC","Type_Other"]]
    type_encodings
    
Merging Type_DT, Type_FC, Type_Other with essential data
   
    train_data = pd.merge(train_data, type_encodings, left_index = True, right_index = True)
    train_data.drop(["Type"],axis=1,inplace=True)
    train_data.head()
    
removing open date
   
    train_data.drop(["Open Date"],axis=1,inplace=True)
    train_data.head()    
    
keeping P2,P6 and P28 and removing rest all unnecesasry features from train_data
     
     train_data.drop(["P1","P3","P4","P5","P7",	"P8",	"P9",	"P10",	"P11",	"P12",	"P13",	"P14",	"P15",	"P16",	"P17",	"P18",	"P19",	"P20",	"P21",	"P22",	"P23",	"P24",	"P25","P26","P27","P29",	"P30",	"P31",	"P32",	"P33",	"P34",	"P35",	"P36",	"P37"],axis=1,inplace=True)
        train_data.head()
