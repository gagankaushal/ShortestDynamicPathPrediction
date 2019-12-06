# Submitted by: Gagan Kaushal (gk@bu.edu)
# Week 9 Assignment 2, naive bayesian



import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

def AI_BayesianNetwork_predictWastedTimepathWithCityName(pathWithCityName,departureTime, departureWeekDay) -> int:

#    print("\n")
#    print('\t', '\t', '\033[1m' + "Submitted by 'Gagan Kaushal' (gk@bu.edu)" + '\033[0m')
#    print('\t', '\t', '\033[1m' + "BU MET CS-677: Data Science With Python" + '\033[0m')
#    print('\t', '\t', '\t',  '\033[1m' + "Week 9: Assignment 2 " + '\033[0m')
#    print('\t', '\t', '\t', '\033[1m' + "naive bayesian" + '\033[0m')
#    print('')
    
    pd.set_option('display.max_rows', 1000)
    
#    ques = '''implement a Gaussian naive bayesian classifier and compute
#    its accuracy for 2018'''
    
#    print('Ques. 1')
#    print('\033[1m' + ques + '\033[0m','\n')
    
    
    #df1 = pd.read_csv('Dataset\\SubsetOfOriginalData_with3cityNames.csv')
    df1 = pd.read_csv('Dataset\\Motor_Vehicle_Crashes_-_Case_Information__Three_Year_Window - Copy_with3citiesV3.csv')
    
    #le = LabelEncoder ()
    #
    ##print(df)
    #
    ##X = df [[ 'Time','Day of Week']].values
    #X = le. fit_transform (df [[ 'Road Surface Conditions','Day of Week']].values)
    #print(X)
    #
    #
    #Y = df [[ 'Number of Vehicles Involved']].values
    #print(Y)
    #
    #df['Predicted Number of Vehicles Involved'] = np.nan
    #new_x =  df [[ 'Time','Day of Week']].values
    #Y_test = df [['Predicted Number of Vehicles Involved']].values
    #
    #
    
    
    
    pd.set_option('display.max_columns', 999)
    pd.set_option('display.width', 1000)
    #df = df1  
    
    timeWasted = 0
    
    ## Starting city and Destination city's traffic prediction is not done
    for IndexOfcitY in range(1,len(pathWithCityName)-1) :
#        df = df1[df1['Municipality'] == citY]
        df = df1[df1['City'] == pathWithCityName[IndexOfcitY]]  
        #df = df1[df1['Municipality'] == 'Kansas City']
        #print(df1[df1['Municipality'] == 'MARTINSBURG']) #.df[df['Year'] == 2017].here the variable will be added for city
        # initially based on teh cities.....sort which ...maybe find a couple of cities along teh way
        input_data = df [[ 'Time Label', 'Day of Week']]
        #print(input_data)
        dummies = [pd. get_dummies ( df [c]) for c in input_data.columns ]
        binary_data = pd.concat ( dummies , axis =1)
    #    print(binary_data)
        X = binary_data [0:len(binary_data)].values
    #    print(X)
        le = LabelEncoder ()
        Y = le.fit_transform ( df ['Number of Vehicles Involved'].values )
        #Y1 = df ['Number of Vehicles Involved'].values 
        #print(Y)
        #print(Y1)
        #NB_classifier = MultinomialNB().fit (X, Y)
        NB_classifier = GaussianNB().fit (X, Y)
        
        
        
        
        
        # new_instance is defined as
#        departureTime = '10:00'
#        departureWeekDay = 'Sunday'
        
        #=IF(C2>=2/3,"Night",IF(C2>=0.5,"Afternoon","Morning"))
        
        if float(departureTime[:2]) >= 16:
            timeLabel = 'Night'
        
        elif float(departureTime[:2]) >= 12:
            timeLabel = 'Afternoon'
        
        else:
            timeLabel = 'Morning'
        
        #print(timeLabel)
        #new_x = np.empty((10,))
        #new_x [:] = np.nan
        
        # input consisiting of day of week and time of day (afternoon)
        new_x = np.zeros((10,),dtype = int)
        
        ## Mapping user input to 1x10 array
        if timeLabel == 'Afternoon':
            new_x[0] = 1
        elif timeLabel == 'Morning':
            new_x[1] = 1
        elif timeLabel == 'Night':
            new_x[2] = 1
        
        
        ## Mapping user input to 1x10 array
        ##
        if departureWeekDay == 'Friday':
            new_x[3] = 1
        elif departureWeekDay == 'Monday':
            new_x[4] = 1
        elif departureWeekDay == 'Saturday':
            new_x[5] = 1
        elif departureWeekDay == 'Sunday':
            new_x[6] = 1
        elif departureWeekDay == 'Thursday':
            new_x[7] = 1
        elif departureWeekDay == 'Tuesday':
            new_x[8] = 1
        elif departureWeekDay == 'Wednesday':
            new_x[9] = 1
        
        #print(binary_data [0:len(binary_data)].values)
        #new_x = np.nan(10)
    #    print('User input mapped',new_x)
        
        #new_x = np.reshape(new_x, (-1, 2))
        new_instance = np.asmatrix (new_x)
        #new_instance = np. asmatrix (binary_data [0:len(binary_data)].values)
        prediction = NB_classifier.predict ( new_instance )
    #    print(prediction)
        
        numberOfCarsInvolvedInAccident = int(prediction + 1)
        print('--> Predicted number of cars involved in accident: ', numberOfCarsInvolvedInAccident)
        
        
        ## Mapping prediction of [number of cars involved in accident] to [time wasted]
        if numberOfCarsInvolvedInAccident == 1:
            timeWasted += 2
        
        elif numberOfCarsInvolvedInAccident == 2:
            timeWasted += 3
        
        elif numberOfCarsInvolvedInAccident == 3:
            timeWasted += 4
        
        elif numberOfCarsInvolvedInAccident == 4:
            timeWasted += 5
        print('--> Wasted time in traffic jam at',pathWithCityName[IndexOfcitY],'   :', timeWasted, 'hours')
        print()
    print()
    print('Total Wasted time in traffic jam: ', timeWasted, 'hours')
    return timeWasted

#accuracy = 1-np.mean ( prediction != Y )
#print(accuracy)

#[0 ,0 ,1 ,1 ,0 ,0 ,0 ,1]# basically convert the user input into ...e.g. if afternoon....1st element should be 1 and so on
#
#
#
#
#print(X)
#



#pathWithCityName = ['Minneapolis','Kansas City', 'Houston']
#timeWasted = AI_BayesianNetwork_predictWastedTimepathWithCityName(pathWithCityName)
#
#print(timeWasted)
#








#NB_classifier = GaussianNB().fit (X, Y)
#predicted = NB_classifier.predict (new_x)
#accuracy = NB_classifier.score (new_x, Y_test)

#
#################### TRAINING DATA - 2017 ####################
#df3 = df[df['Year'] == 2017]
#df2 = df3.groupby(['Week_Number'])['Return','label'].mean()
#df4 = df3.groupby(['Week_Number'])['Return','label'].std()
#df5 = df3.groupby(['Week_Number'])['label'].agg(['last']).stack()
#df5 = df5.to_frame()
#df2  = df2.rename(columns={"Return": "[Mean of Weekly Returns]"})
#df2['[Std Dev of Weekly Returns]'] = df4['Return']
#df5 = df5.droplevel(level = 1)
#df5  = df5.rename(columns={0: "[Mean of Weekly Returns]"})
#df2['[label]'] = df5['[Mean of Weekly Returns]'] 
#X = df2 [[ '[Mean of Weekly Returns]','[Std Dev of Weekly Returns]']].values
#
#################### TESTING DATA - 2018 ####################
#df13 = df[df['Year'] == 2018]
#df12 = df13.groupby(['Week_Number'])['Return','label'].mean()
#df14 = df13.groupby(['Week_Number'])['Return','label'].std()
#df15 = df13.groupby(['Week_Number'])['label'].agg(['last']).stack()
#df15 = df15.to_frame()
#df12  = df12.rename(columns={"Return": "[Mean of Weekly Returns]"})
#df12['[Std Dev of Weekly Returns]'] = df14['Return']
#df15 = df15.droplevel(level = 1)
#df15  = df15.rename(columns={0: "[Mean of Weekly Returns]"})
#df12['[label]'] = df15['[Mean of Weekly Returns]'] 
#new_x = df12 [[ '[Mean of Weekly Returns]','[Std Dev of Weekly Returns]']].values
#
#
#
##### TRUE LABELS FOR YEAR - 2017 ###########
#Y = df2 [[ '[label]']].values
#
##### TRUE LABELS FOR YEAR - 2018 ###########
#Y_test = df12 [[ '[label]']].values
#
##### GAUSSIAN NAIVE BAYESIAN CLASSIFIER ######
#NB_classifier = GaussianNB().fit (X, Y)
#predicted = NB_classifier.predict (new_x)
#accuracy = NB_classifier.score (new_x, Y_test)
#
#
#Conclusion= '''
#Ans. 
#Gaussian naive bayesian classifier was implemented and the accuracy for year 2 is ->: '''
#
#print(Conclusion, round(accuracy*100,2), '%')
