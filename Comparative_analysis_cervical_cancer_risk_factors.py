"""
Title:  Classification Techniques for Cervical Cancer Detection Utilising At Risk Factors and Screening Test Results
Author: Seán Quinlan - sean.a.quinlan@mycit.ie
Supervisor: Dr. Ruairi O'Reilly - ruairi.oreilly@cit.ie - https://scholar.google.com/citations?user=86x5oQgAAAAJ&hl=en

 If you find this code useful in your research, please consider citing:

@article{quinlan2019comparative,
  title={A Comparative Analysis of Classification Techniques for Cervical Cancer Utilising At Risk Factors and Screening Test Results},
  author={Quinlan, Sean and Afli, Haithem and O’Reilly, Ruairi},
  year={2019}
}

"""

""" Import Libraries """

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import tree  
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics 
from sklearn.metrics import confusion_matrix   
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_validate
import warnings
warnings.filterwarnings("ignore")
RANDOM_STATE = 3



"""  General Info on the Data """ 
# Dimensions of the data set - rows , columns 
# Summary statistics of variables - uncomment section to get a variable by varaible print out. 
# Countplot of respone variable "Biopsy", with number and percent

def genInfo():
    global dfDescribe, N, P
    print("\n***** General Information of Dataset and Response Variable *****")
    
    # Observationss and variables 
    nRow, nCol = df.shape
    print("\nThe Data consists of",nRow,"observations and", nCol,"variables.\n")
    
    # Variable, observations per column (excluding Na's)) & data type 
    print(df.info())

    #print(df.describe().transpose())
    
    # Summary Statistics for each variable  
    # uncomment below to get a better output of describe 
    dfDescribe = df.describe()   
    for col in dfDescribe:
       print (dfDescribe[col])    
    
    # Response Summary - Countplot by response category, and figures by response category
    print("\nCountplot of the response variable 'Biopsy'")
    sns.countplot(x ='Biopsy',label = "Count", data = df).set_title("Biopsy Classification") 
    plt.show()
    N, P = df['Biopsy'].value_counts()
    nPc = round(N/len(df)*100,2)
    pPc = round(P/len(df)*100,2)
    print("\nBiposy results from the data set included: \nNegative", N,"(",nPc,"%)","\nPositive",P,"(",pPc,"%).")  

    #print(df.groupby('Biopsy').size())

#""" Analysis """

from matplotlib_venn import venn2
# First way to call the 2 group Venn diagram:
#venn2(subsets = (49, 12, 6), set_labels = ('Positive Biopsy Result', 'Positive HPV'))
#venn2(subsets = (6,791,49), set_labels = ('Positive Biopsy Result', 'Negative HPV'))


#venn2(subsets = (45, 113, 10), set_labels = ('Positive Biopsy Result', 'Smoker'))
#venn2(subsets = (11,678,44), set_labels = ('Positive Biopsy Result', 'Non Smokes'))

""" Outliers """

def outliers():
    print("\n***** 4.1 Outliers *****",   
    "\n\nIt was decided that even though the data contains outliers that they would be included in the analysis.",   
    "\nThe reason for this is that it is hoped that the model will be able to account for simialar observations if they occur.")
    
    # Box Plots 
    fig1, ax1 = plt.subplots(1,2)
    sns.boxplot(df.iloc[:,0], ax = ax1[0])
    sns.boxplot(df.iloc[:,1], ax = ax1[1])
    plt.show()
    
    fig2, ax2 = plt.subplots(1,2)
    sns.boxplot(df.iloc[:,2], ax = ax2[0])
    sns.boxplot(df.iloc[:,3], ax = ax2[1])
    plt.show()



""" Missing Data """
# Number and percent of missing data by variable 
# Number of missing data points
# Number of populated data points 
# Total data points in data set 
# Percent of data that is missing 
# Plot Showing percent missing data

def missingData():  
    print("\n***** 4.2 Missing Data*****")
    totalMissing = df.isnull().sum().sort_values(ascending=False)
    percentMissing =  (df.isna().mean().round(4) * 100).sort_values(ascending=False)
    MissingData = pd.DataFrame({'Number' : totalMissing, 'Percent' : percentMissing}).head(26)
    print("\nThe number and percent of missing data per variable (26 variables as 10 have no missing values): \n",MissingData)
    
    # Number of missing data points
    dfMissing = df.isna().sum().sum()
    print("\nThere are", dfMissing, "missing data points.")
    
    # Number of populated data points 
    dfObservations = int(dfDescribe.iloc[0].sum())
    print("\nExcluding NA values, there are",dfObservations,"data points.")
    
    # Total data points in data set 
    dfTotalDataPoints = dfMissing + dfObservations
    print("\nThere are a total of",dfTotalDataPoints, "data points including na's.")
    
    # Percent of data that is missing 
    dfTotalPercentMissingData = ((dfMissing / dfTotalDataPoints)*100).round(2)
    print("\nThere is a total of",dfTotalPercentMissingData,"% data missing from this dataset.")

    # Plot Showing percent missing data  
    print("\nThe percentage of missing data can be seen graphically in the below figure.")
    f, ax = plt.subplots(figsize=(15, 6))
    plt.xticks(rotation='90')
    sns.barplot(x=MissingData.index, y=MissingData['Percent'])
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent Missing Data by Variable', fontsize=15)
    plt.show()

""" Edit Data """
# Create new dataframe and drop variables located at points 26 and 27 (due to missing 92% data)    
# create 3 lists - column names, variables for mean imputation, variables for median imputation
# for loop to iterate through dataframe and perform mean /  median imputation on specified variables
 

def replaceData():
    global df1 , columHeaders
    
    # create new dataframe while dropping variabels located at points 26 and 27 
    df1 = df.drop(df.columns[[26, 27]], axis=1)
    
    print("\nAs two variables (",df.columns[26],"and",df.columns[27],") have greater than 50% missing data we remove those.",
          "The Data now consists of",df1.shape[0],"observations and", df1.shape[1],"variables.")
    
    print("\nWe replace the missing data of the other variables with their respective mean / median")    
    # Three lists that contain column names that require imputation by mean and median methods
    columHeaders = list(df1)
    
    columnsMean = ['Number of sexual partners','First sexual intercourse','Smokes (years)',
                   'Smokes (packs/year)','Hormonal Contraceptives (years)','IUD (years)']
    
    columnsMedian = ['Num of pregnancies','Smokes','Hormonal Contraceptives','IUD','STDs','STDs (number)',
                     'STDs:condylomatosis','STDs:cervical condylomatosis','STDs:vaginal condylomatosis',
                     'STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease',
                     'STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV',
                     'STDs:Hepatitis B','STDs:HPV']
    
    # Loop that iterates over the dataframe, if the column belowns to the mean list as described above
    # fill the missing values with mean, if belongs to median list fill with median, otherwise leave it alone
    for colHeader in columHeaders:
        if colHeader in columnsMean:
            df1[colHeader] = (df1[colHeader].fillna(df1[colHeader].mean()).round(0))
        elif colHeader in columnsMedian:
            df1[colHeader] = df1[colHeader].fillna(df1[colHeader].median())
        else:
            pass

""" Imblanced Data """

def imb():   
    global X, y
    print("\n***** 4.3 Imbalance *****",
      "\n\nBiposy results from the data set included \n Negative", N,"\n Positive",P,
      "\nFrom the above we can see that the dataset is heavily imbalanced.",
      "\nTo address this imbalance over, under and combination resampling methods are used.")
    
    X = df1.drop('Biopsy',1)
    y = df1['Biopsy']

""" Oversampling """
# create 2 dataframes based on 2 over-sampling methods 
# print the shape of the dataframes response


def ovrSmpleDfs():
    global dfRndOvrSam, dfAda
    # Random Over Sampling 
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state = RANDOM_STATE)
    X_rndOvrSam, y_rndOvrSam = ros.fit_sample(X, y)
    
    # Create new dataframe 
    m1a = pd.DataFrame(X_rndOvrSam)
    m1b = pd.DataFrame(y_rndOvrSam)
    m1b.columns = ['Biopsy']
    dfRndOvrSam = m1a.join(m1b)
    dfRndOvrSam.columns = columHeaders
    
    #  Adaptive Synthetic Sampling
    from imblearn.over_sampling import ADASYN
    ada = ADASYN(random_state = RANDOM_STATE)
    X_ada, y_ada = ada.fit_sample(X, y)
    
    # Create new dataframe 
    m2a = pd.DataFrame(X_ada)
    m2b = pd.DataFrame(y_ada)
    m2b.columns = ['Biopsy']
    dfAda = m2a.join(m2b)
    dfAda.columns = columHeaders
    
    # round data frame values 
    dfAda = dfAda.round(decimals=0)  
    
    print("\n**Over Sampling**")
    print('Random Over Sampling dataset Biopsy classification {}'.format(Counter(y_rndOvrSam)))
    print('Adaptive Synthetic Sampling dataset Biopsy classification {}'.format(Counter(y_ada)))

""" Under-Sampling """
# create 2 dataframes based on 2 under-sampling methods
# print the shape of the dataframes response


def undrSmpleDfs():
    global dfRndUndrSam, dfNcl
    # Random Under Sampling
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state = RANDOM_STATE)
    X_rndUndrSam, y_rndUndrSam = rus.fit_sample(X, y)
    
    # Create new dataframe 
    m3a = pd.DataFrame(X_rndUndrSam)
    m3b = pd.DataFrame(y_rndUndrSam)
    m3b.columns = ['Biopsy']
    dfRndUndrSam = m3a.join(m3b)
    dfRndUndrSam.columns = columHeaders
    
    # Neighbourhood Cleaning Rule
    from imblearn.under_sampling import NeighbourhoodCleaningRule
    ncl = NeighbourhoodCleaningRule(random_state = RANDOM_STATE)
    X_ncl, y_ncl = ncl.fit_sample(X, y)
    
    # Create new dataframe 
    m4a = pd.DataFrame(X_ncl)
    m4b = pd.DataFrame(y_ncl)
    m4b.columns = ['Biopsy']
    dfNcl = m4a.join(m4b)
    dfNcl.columns = columHeaders
    
    print("\n**Under Sampling**")
    print('Random Under Sample dataset Biopsy classification {}'.format(Counter(y_rndUndrSam)))
    print('NCL dataset Biopsy classification {}'.format(Counter(y_ncl)))

""" Combination """

# create 2 dataframes based on 2 combination-sampling methods 
# print the shape of the dataframes response

def cmbiSmplDfs():
    global dfSmEnn, dfSmTom
    
    # SmoteTomek 
    from imblearn.combine import SMOTETomek
    smTom = SMOTETomek(random_state = RANDOM_STATE)
    X_smTom, y_smTom = smTom.fit_sample(X, y)
    
    # Create new dataframe 
    m5a = pd.DataFrame(X_smTom)
    m5b = pd.DataFrame(y_smTom)
    m5b.columns = ['Biopsy']
    dfSmTom = m5a.join(m5b)
    dfSmTom.columns = columHeaders
    
    # round data frame values 
    dfSmTom = dfSmTom.round(decimals=0)
    
    # SMOTE edited nearest-neighbours
    from imblearn.combine import SMOTEENN
    smote_enn = SMOTEENN(random_state = RANDOM_STATE)
    X_smNn, y_smNn = smote_enn.fit_sample(X, y)
    
    # Create new dataframe 
    m6a = pd.DataFrame(X_smNn)
    m6b = pd.DataFrame(y_smNn)
    m6b.columns = ['Biopsy']
    dfSmEnn = m6a.join(m6b)
    dfSmEnn.columns = columHeaders
    
    # round data frame values 
    dfSmEnn = dfSmEnn.round(decimals=0)
    
    print("\n**Combination**")
    print('SMOTETomek dataset Biopsy classification {}'.format(Counter(y_smTom)))
    print('SMOTE Edited Nearest Neighbour dataset Biopsy classification {}'.format(Counter(y_smNn)))
    print("")

""" Cross Validation """

def crossValModels(aDataset):
    global models, names
    Scoring = ['accuracy','precision', 'recall','f1']
    X = aDataset.drop('Biopsy',1)
    Y = aDataset['Biopsy']

    models = [] 
    models.append(("Decision Tree (Entropy)",tree.DecisionTreeClassifier(criterion='entropy',random_state = RANDOM_STATE)))
    models.append(("Decision Tree (Gini)",tree.DecisionTreeClassifier(criterion='gini', random_state= RANDOM_STATE)))    
    models.append(("Gaussian Naive Bayes",GaussianNB())) 
    models.append(("Gradient Boosting",GradientBoostingClassifier(n_estimators = 100, learning_rate = 2.0, max_depth = 1, random_state = RANDOM_STATE)))
    models.append(("Kmeans",KMeans(n_clusters = 2, random_state = RANDOM_STATE)))  
    models.append(("K-Nearest Neighbours",KNeighborsClassifier(n_neighbors = 5, p=2, weights = 'distance'))) 
    models.append(("Linear Discriminant Analysis",LinearDiscriminantAnalysis(solver = 'svd')))
    models.append(("Logistic Regression",LogisticRegression(solver= "liblinear")))           
    models.append(("Random Forest",RandomForestClassifier(max_features = 7, n_estimators = 100, random_state = RANDOM_STATE)))
    models.append(('SVC', svm.SVC(random_state = RANDOM_STATE)))
 
    for name, model in models: 
      print("The cross validation score for model",name,"is:")      
      scores = cross_validate(model, X, Y, cv=5, scoring = Scoring)
      print("Average Accuracy:", round((scores["test_accuracy"]).mean(),4), "+/-", round((scores["test_accuracy"]).std(),4))
      print("Average Precision:", round((scores["test_precision"]).mean(),4), "+/-", round((scores["test_precision"]).std(),4))
      print("Average Recall:", round((scores["test_recall"]).mean(),4), "+/-", round((scores["test_recall"]).std(),4))
      print("Average F1:", round((scores["test_f1"]).mean(),4), "+/-", round((scores["test_f1"]).std(),4))
      print("")

def CallcrossValModels():
    print("***Cross validation results for the original (cleanded) dataset***")    
    crossValModels(df1)
    
    print("\n***Cross validation results For the random over-sampled dataset***") 
    crossValModels(dfRndOvrSam)

    print("\n***Cross validation results For the Adaptive Synthetic Sampling over-sampled dataset***") 
    crossValModels(dfAda)

    print("\n***Cross validation results For the random undersampled dataset***") 
    crossValModels(dfRndUndrSam)

    print("\n***Cross validation results For the Neighbourhood Cleaning Rule under-sampled dataset***") 
    crossValModels(dfNcl)

    print("\n***Cross validation results For the SMOTETomek combi-sampled dataset") 
    crossValModels(dfSmTom)

    print("\n***Cross validation results For the SMOTE Edited Nearest Neighbour combination-sampled dataset") 
    crossValModels(dfSmEnn)

""" 4.4 Splitting Data """
# shuffle and Split the 7 datasets into training and test sets 
# Create 7 lists, each list contains the 4 split elements from their respective data frames

def splitDfs():   
    global ListDfOrg, ListDfRndOvrSam,ListDfAda,ListDfRndmUndrSam, ListDfNcl, ListDfSmTom, ListDfSmEnn 
    # separate dataframes into features(X) and response(Y)
    XdfRndOvrSam = dfRndOvrSam.drop('Biopsy',1)
    YdfRndOvrSam = dfRndOvrSam['Biopsy']
    
    XdfAda = dfAda.drop('Biopsy',1)
    YdfAda = dfAda['Biopsy']
    
    XdfRndUndrSam = dfRndUndrSam.drop('Biopsy',1)
    YdfRndUndrSam = dfRndUndrSam['Biopsy']
    
    XdfNcl = dfNcl.drop('Biopsy',1)
    YdfNcl = dfNcl['Biopsy']
    
    XdfSmTom = dfSmTom.drop('Biopsy',1)
    YdfSmTom = dfSmTom['Biopsy']
    
    XdfSmEnn = dfSmEnn.drop('Biopsy',1)
    YdfSmEnn = dfSmEnn['Biopsy']
    
    from sklearn import model_selection
    TEST_SIZE = 0.2
    
    trainXdf1, testXdf1, trainYdf1, testYdf1 = model_selection.train_test_split( X, y, test_size = TEST_SIZE, random_state = RANDOM_STATE, shuffle=True)
    trainXdfRndOvrSam, testXdfRndOvrSam, trainYdfRndOvrSam, testYdfRndOvrSam = model_selection.train_test_split( XdfRndOvrSam, YdfRndOvrSam, test_size = TEST_SIZE, random_state = RANDOM_STATE, shuffle=True)
    trainXdfAda, testXdfAda, trainYdfAda, testYdfAda = model_selection.train_test_split( XdfAda, YdfAda, test_size = TEST_SIZE, random_state = RANDOM_STATE, shuffle=True)
    trainXdfRndUndrSam, testXdfRndUndrSam, trainYdfRndUndrSam, testYdfRndUndrSam = model_selection.train_test_split( XdfRndUndrSam, YdfRndUndrSam, test_size = TEST_SIZE, random_state = RANDOM_STATE, shuffle=True)    
    trainXdfNcl, testXdfNcl, trainYdfNcl, testYdfNcl = model_selection.train_test_split( XdfNcl, YdfNcl, test_size = TEST_SIZE, random_state = RANDOM_STATE, shuffle=True)
    trainXdfSmTom, testXdfSmTom, trainYdfSmTom, testYdfSmTom = model_selection.train_test_split( XdfSmTom, YdfSmTom, test_size = TEST_SIZE, random_state = RANDOM_STATE, shuffle=True)
    trainXdfSmEnn, testXdfSmEnn, trainYdfSmEnn, testYdfSmEnn = model_selection.train_test_split( XdfSmEnn, YdfSmEnn, test_size = TEST_SIZE, random_state = RANDOM_STATE, shuffle=True)
    
    # add each split data sets components to a list to allow for access and globalisation 
    ListDfOrg = [trainXdf1, testXdf1, trainYdf1, testYdf1]
    ListDfRndOvrSam = [trainXdfRndOvrSam, testXdfRndOvrSam, trainYdfRndOvrSam, testYdfRndOvrSam]
    ListDfAda= [trainXdfAda, testXdfAda, trainYdfAda, testYdfAda]
    ListDfRndmUndrSam = [trainXdfRndUndrSam, testXdfRndUndrSam, trainYdfRndUndrSam, testYdfRndUndrSam]
    ListDfNcl = [trainXdfNcl, testXdfNcl, trainYdfNcl, testYdfNcl]
    ListDfSmTom = [trainXdfSmTom, testXdfSmTom, trainYdfSmTom, testYdfSmTom]
    ListDfSmEnn = [trainXdfSmEnn, testXdfSmEnn, trainYdfSmEnn, testYdfSmEnn]

""" Tuning KNN Tuning """

### The function call for the following 3 functions for parameter tuning are commented out 
print("\n*** Note the code for printing the parameter tuning for KNN and Random Forest have been commented out***\n")

# Produces a plot to determine the optimim value for k in Knn 
def knnTuner(alist):
    v=[]
    for i in range(1, 20):
        knn = KNeighborsClassifier(n_neighbors=i)
        # fit the model with data
        knn.fit(alist[0], alist[2])
        pred = knn.predict(alist[1])
        y_acc = round(metrics.accuracy_score(pred,alist[3])*100,2)
        v.append(y_acc)

    plt.plot(v,c='Orange',)
    plt.show()
    print(v)

#knnTuner(ListDfOrg)
#knnTuner(ListDfRndOvrSam)
#knnTuner(ListDfAda)
#knnTuner(ListDfRndmUndrSam)
#knnTuner(ListDfNcl)
#knnTuner(ListDfSmTom)
#knnTuner(ListDfSmEnn)

""" RF Tuning - Number of Trees """

# The below two functions can be used to fine tune the random forest model 
# This function can be used to determine the optimum number of trees to used in the rf model 
# the function takes one of the input lists created from the previous train/test split section 
def rfTreeNumber(alist):
    results=[]
    n_estimator_values=range(10,200,10)
    for trees in n_estimator_values:
        model=RandomForestClassifier(trees, n_jobs=-1,random_state=RANDOM_STATE)
        model.fit(alist[0], alist[2])
        print(trees, "trees")    
        pred = model.predict(alist[1])
        y_acc = round(metrics.accuracy_score(pred,alist[3])*100,2)
        print("accuracy : ", y_acc)
        results.append(y_acc)

    pd.Series(results, n_estimator_values).plot();

#rfTreeNumber(ListDfOrg)
#rfTreeNumber(ListDfRndOvrSam)
#rfTreeNumber(ListDfAda)
#rfTreeNumber(ListDfRndmUndrSam)
#rfTreeNumber(ListDfNcl)
#rfTreeNumber(ListDfSmTom)
#rfTreeNumber(ListDfSmEnn)

""" RF - Number of Features """

# This function can be used to determine the optimum number of features to be used in the rf model  
# the function takes one of the input lists created from the previous train/test split section 
def rfFeatureNumber(alist):   
    results=[]
    max_features_values=["auto", "sqrt", "log2", 5,10 ,15,20,25]
    for item in max_features_values:        
        model = RandomForestClassifier(n_estimators=170, n_jobs=-1, random_state=RANDOM_STATE, max_features=item)                              
        model.fit(alist[0], alist[2])
        print(item, "option")
        pred = model.predict(alist[1])
        y_acc = round(metrics.accuracy_score(pred,alist[3])*100,2)
        print("accuracy : ", y_acc)
        results.append(y_acc)
        
    pd.Series(results, max_features_values).plot(kind="barh");

#rfFeatureNumber(ListDfOrg)
#rfFeatureNumber(ListDfRndOvrSam)
#rfFeatureNumber(ListDfAda)
#rfFeatureNumber(ListDfRndmUndrSam)
#rfFeatureNumber(ListDfNcl)
#rfFeatureNumber(ListDfSmTom)
#rfFeatureNumber(ListDfSmEnn)

""" Model Making """

# The end part of this code to create the models in a list was taken from https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

# function containing the various models to be appleid to the data 
# the function takes one of the input lists created from the previous train/test split section 
def modelMaker(aList):
    global models
    models = [] 
    models.append(("Decision Tree (Entropy)",tree.DecisionTreeClassifier(criterion='entropy',random_state = RANDOM_STATE)))
    models.append(("Decision Tree (Gini)",tree.DecisionTreeClassifier(criterion='gini', random_state= RANDOM_STATE)))    
    models.append(("Gaussian Naive Bayes",GaussianNB())) 
    models.append(("Gradient Boosting",GradientBoostingClassifier(n_estimators = 100, learning_rate = 2.0, max_depth = 1, random_state = RANDOM_STATE)))
    models.append(("Kmeans",KMeans(n_clusters = 2, random_state = RANDOM_STATE)))  
    models.append(("K-Nearest Neighbours",KNeighborsClassifier(n_neighbors = 5, p=2, weights = 'distance'))) 
    models.append(("Linear Discriminant Analysis",LinearDiscriminantAnalysis(solver = 'svd')))
    models.append(("Logistic Regression",LogisticRegression(solver= "liblinear")))           
    models.append(("Random Forest",RandomForestClassifier(max_features = 7, n_estimators = 100, random_state = RANDOM_STATE)))
    models.append(('SVC', svm.SVC(random_state = RANDOM_STATE)))
    
    # for loop that takes a list that contains the split 4 test/training sets 
    # and iterates through the model methods and returns accuracy and confusion matrix for each 
    for name, model in models:
        model_fit = model.fit(aList[0], aList[2])
        model_pred = model_fit.predict(aList[1])
        
        showMetrics(aList,model_pred,name)

""" Helper Function to show metrics """

def conf_matrix_metrics(y_true, y_pred):
  cnf_matrix = confusion_matrix(y_true, y_pred)
  #print(classification_report(y_pred, y_true))

  TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()

  # Sensitivity, hit rate, recall, or true positive rate
  TPR = round(TP/(TP+FN),4)
  # Specificity or true negative rate
  TNR = round(TN/(TN+FP),4)
  # Precision or positive predictive value
  PPV = round(TP/(TP+FP),4)
  # Negative predictive value
  NPV = round(TN/(TN+FN),4)
  # Fall out or false positive rate
  FPR = round(FP/(FP+TN),4)
  # False negative rate
  FNR = round(FN/(TP+FN),4)
  # False discovery rate
  FDR = round(FP/(TP+FP),4)
  # Overall accuracy
  ACC = round((TP+TN)/(TP+FP+FN+TN),4)
  # Diagnostic Odds Ratio
  DOR = round((TP/FP)/(FN/TN),4)

  print("Sensitivity / Recall",TPR)
  print("Specificity",TNR)
  print("Precision",PPV)
  #print("NPV",NPV)
  #print("FPR",FPR)
  #print("FNR",FNR)
  print("ACC",ACC)
  print("DOR",DOR)

def showMetrics(aList,model_pred,name):
  y_acc = round(metrics.accuracy_score(model_pred,aList[3])*100,2)  
  print("The",name,"model has an accuracy of",y_acc,"%.")
  cf_mat = confusion_matrix(aList[3], model_pred) 
  print("Confusion matrix for the",name,"model\n", cf_mat)
  print(classification_report(aList[3], model_pred))
  print("Precision",round(precision_score(aList[3], model_pred),2))
  print("Recall",round(recall_score(aList[3], model_pred),2))  
  conf_matrix_metrics(aList[3],model_pred)
  print("")

# use the modelMaker function for each of the 7 dataframes 
def useModelMaker():
    print("***Models for the original (cleanded) data frame***")    
    modelMaker(ListDfOrg)
    
    print("\n***Models For the random over-sampled data frame***") 
    modelMaker(ListDfRndOvrSam)

    print("\n***Models For the Adaptive Synthetic Sampling over-sampled data frame***") 
    modelMaker(ListDfAda)

    print("\n***Models For the random undersampled data frame***") 
    modelMaker(ListDfRndmUndrSam)

    print("\n***Models For the Neighbourhood Cleaning Rule under-sampled data frame***") 
    modelMaker(ListDfNcl)

    print("\n***Models For the SMOTETomek combi-sampled data frame") 
    modelMaker(ListDfSmTom)

    print("\n***Models For the SMOTE Edited Nearest Neighbour combination-sampled data frame") 
    modelMaker(ListDfSmEnn)

""" Call Functions """

def main():
    global df, models
    df = pd.read_csv('CervicalCancer.csv',na_values=["?"])
    genInfo()
    outliers()
    missingData()    
    replaceData()
    imb()
    ovrSmpleDfs()
    undrSmpleDfs()
    cmbiSmplDfs()
    CallcrossValModels()
    splitDfs()
    useModelMaker() 
        
main()

""" Variable Importance Plot """

X = ListDfSmTom[0]
Y = ListDfSmTom[2]

clf = RandomForestClassifier(max_features = 7, n_estimators = 100, random_state = RANDOM_STATE)
rf = clf.fit(X, Y)
                                                      
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = X.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)

features = X.columns
importance = rf.feature_importances_
indices = np.argsort(importance)
indices = indices[23:]

plt.title('Feature Importances')
plt.barh(range(len(indices)), importance[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

""" Comparison with removing screening test results """

Scoring = ['accuracy','precision', 'recall','f1']

z = ['Biopsy','Hinselmann','Schiller','Citology']
X = dfSmEnn.drop(z,1)
Y = dfSmEnn['Biopsy']

models = []     
models.append(("Random Forest",RandomForestClassifier(max_features = 7, n_estimators = 100, random_state = RANDOM_STATE)))

for name, model in models: 
  print("The cross validation score for model",name,"is:")      
  scores = cross_validate(model, X, Y, cv=5, scoring = Scoring)
  print("Average Accuracy:", round((scores["test_accuracy"]).mean(),4), "+/-", round((scores["test_accuracy"]).std(),4))
  print("Average Precision:", round((scores["test_precision"]).mean(),4), "+/-", round((scores["test_precision"]).std(),4))
  print("Average Recall:", round((scores["test_recall"]).mean(),4), "+/-", round((scores["test_recall"]).std(),4))
  print("Average F1:", round((scores["test_f1"]).mean(),4), "+/-", round((scores["test_f1"]).std(),4))
  print("")


""" Voting Classifier """

from sklearn.ensemble import VotingClassifier
from sklearn import model_selection

def VoteClassifier(aList):
  VCmodels = [] 
  VCmodels.append(("Decision Tree (Entropy)",tree.DecisionTreeClassifier(criterion='entropy',random_state = RANDOM_STATE)))
  VCmodels.append(("Decision Tree (Gini)",tree.DecisionTreeClassifier(criterion='gini', random_state= RANDOM_STATE)))    
  #VCmodels.append(("Gaussian Naive Bayes",GaussianNB())) 
  VCmodels.append(("Gradient Boosting",GradientBoostingClassifier(n_estimators = 100, learning_rate = 2.0, max_depth = 1, random_state = RANDOM_STATE)))
  #VCmodels.append(("Kmeans",KMeans(n_clusters = 2, random_state = RANDOM_STATE)))  
  VCmodels.append(("K-Nearest Neighbours",KNeighborsClassifier(n_neighbors = 5, p=2, weights = 'distance'))) 
  #VCmodels.append(("Linear Discriminant Analysis",LinearDiscriminantAnalysis(solver = 'svd')))
  VCmodels.append(("Logistic Regression",LogisticRegression(solver= "liblinear")))           
  #VCmodels.append(("Random Forest",RandomForestClassifier(max_features = 7, n_estimators = 100, random_state = RANDOM_STATE)))
  VCmodels.append(('SVC', svm.SVC(random_state = RANDOM_STATE)))

  ensemble = VotingClassifier(VCmodels, voting = "hard")
  model_fit = ensemble.fit(aList[0], aList[2])
  model_pred = model_fit.predict(aList[1])

  showMetrics(aList,model_pred,"Voting Classifier")

# Call the voteClassifier on the 7 dataframes 
def callVoteClassifier():  
    print("***Result using voting classifier for the original (cleanded) data frame***")    
    VoteClassifier(ListDfOrg)
    
    print("\n***Result using voting classifier for the random over-sampled data frame***") 
    VoteClassifier(ListDfRndOvrSam)
    
    print("\n***Result using voting classifier For the Adaptive Synthetic Sampling over-sampled data frame***") 
    VoteClassifier(ListDfAda)
    
    print("\n***Result using voting classifier For the random undersampled data frame***") 
    VoteClassifier(ListDfRndmUndrSam)
    
    print("\n***Result using voting classifier For the Neighbourhood Cleaning Rule under-sampled data frame***") 
    VoteClassifier(ListDfNcl)
    
    print("\n***Result using voting classifier For the SMOTETomek combi-sampled data frame") 
    VoteClassifier(ListDfSmTom)
    
    print("\n***Result using voting classifier For the SMOTE Edited Nearest Neighbour combination-sampled data frame") 
    VoteClassifier(ListDfSmEnn)

callVoteClassifier()

""" Neural Network Model """

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()  

x_train = scaler.fit_transform(ListDfRndmUndrSam[0])
x_test  = ListDfRndmUndrSam[2]
y_train = scaler.fit_transform(ListDfRndmUndrSam[1])
y_test = ListDfRndmUndrSam[3]

mlp = MLPClassifier(hidden_layer_sizes=(33,33,33), activation='relu', solver='lbfgs', max_iter=500, random_state= RANDOM_STATE)
model_fit = mlp.fit(x_train,x_test)

model_pred = model_fit.predict(y_train)

y_acc = round(metrics.accuracy_score(model_pred,y_test)*100,2)
print("The model has an accuracy of", y_acc,"%")   
cf_mat = confusion_matrix(y_test,  model_pred) 
print("Confusion matrix for the model\n", cf_mat)
print(classification_report(y_test, model_pred)) 


""" Keras """

import keras
from keras.models import Sequential
from keras.layers import Dense

def keras_model(aList):

  # Initialize the constructor  
  keras_model = Sequential()
  # we have 33 input variables , the first numerical value here represents the freedom the model has, high values lead to overfitting. 

  # Add an input layer 
  keras_model.add(Dense(32, activation='relu', input_dim=33))
  keras_model.add(Dense(16, activation='relu'))
  # Add an output layer - ensure it will be either 0 or 1 
  keras_model.add(Dense(1, activation='sigmoid'))

  # Compile Model - binary_crossentropy for binary classification 
  keras_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  keras_model.fit(aList[0], aList[2], epochs=20, verbose = 0)

  model_pred = keras_model.predict_classes(aList[1])

  # Call metrics function 
  showMetrics(aList,model_pred,"Keras")

def KerasClassifier():
    print("***Result using Keras Model for the original (cleanded) data frame***")    
    keras_model(ListDfOrg)
    
    print("\n***Result using Keras Model for the random over-sampled data frame***") 
    keras_model(ListDfRndOvrSam)
    
    print("\n***Result using Keras Model For the Adaptive Synthetic Sampling over-sampled data frame***") 
    keras_model(ListDfAda)
    
    print("\n***Result using Keras Model For the random undersampled data frame***") 
    keras_model(ListDfRndmUndrSam)
    
    print("\n***Result using Keras Model For the Neighbourhood Cleaning Rule under-sampled data frame***") 
    keras_model(ListDfNcl)
    
    print("\n***Result using Keras Model For the SMOTETomek combi-sampled data frame") 
    keras_model(ListDfSmTom)
    
    print("\n***Result using Keras Model For the SMOTE Edited Nearest Neighbour combination-sampled data frame") 
    keras_model(ListDfSmEnn)

KerasClassifier()

""" TPOT """

# Install tpot on the server
!pip install tpot

global Tpot_Model
from tpot import TPOTClassifier

tpot = TPOTClassifier(generations=5, population_size=100, verbosity=2, random_state= RANDOM_STATE, scoring = 'f1')
#tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state= RANDOM_STATE, scoring = 'recall')

tpot.fit(ListDfSmTom[0], ListDfSmTom[2])
tpot.score(ListDfSmTom[1], ListDfSmTom[3])

tpot_model = tpot.fitted_pipeline_

""" Call the Tpot Model and Evaluate It """

tpot_model

# TPOT Model 
models = [] 
models.append(("TPOT Model",GradientBoostingClassifier(criterion='friedman_mse', init=None,
                                            learning_rate=0.5, loss='deviance',
                                            max_depth=8, max_features=0.55,
                                            max_leaf_nodes=None,
                                            min_impurity_decrease=0.0,
                                            min_impurity_split=None,
                                            min_samples_leaf=15,
                                            min_samples_split=12,
                                            min_weight_fraction_leaf=0.0,
                                            n_estimators=100,
                                            n_iter_no_change=None,
                                            presort='auto', random_state=3,
                                            subsample=0.45, tol=0.0001,
                                            validation_fraction=0.1, verbose=0,
                                            warm_start=False)))
models.append(("Random Forest",RandomForestClassifier(max_features = 7, n_estimators = 100, random_state = RANDOM_STATE)))


for name, model in models:
    model_fit = model.fit(ListDfSmTom[0], ListDfSmTom[2])
    model_pred = model_fit.predict(ListDfSmTom[1])  
    
    # Call Metrics Function 
    showMetrics(ListDfSmTom,model_pred,name)

global Tpot_Model
from tpot import TPOTClassifier

#scoring = {'accuracy': 'accuracy','prec': 'precision', 'rec':'recall' }
#tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state= RANDOM_STATE)
#tpot = TPOTClassifier(generations=5, population_size=100, verbosity=2, random_state= RANDOM_STATE)

tpot.fit(ListDfSmTom[0], ListDfSmTom[2])
tpot.score(ListDfSmTom[1], ListDfSmTom[3])

tpot_model = tpot.fitted_pipeline_



""" Get Module Versions """
# Check the versions of libraries

def getModuleVersion():
  # Python version
  import sys
  print('Python: {}'.format(sys.version))
  # numpy
  import numpy
  print('numpy: {}'.format(numpy.__version__))
  # matplotlib
  import matplotlib
  print('matplotlib: {}'.format(matplotlib.__version__))
  # pandas
  import pandas
  print('pandas: {}'.format(pandas.__version__))
  # scikit-learn
  import sklearn
  print('sklearn: {}'.format(sklearn.__version__))
  # seaborn
  import seaborn 
  print('seaborn: {}'.format(seaborn.__version__))

getModuleVersion()