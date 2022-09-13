import pandas as pd
#load data
df_test = pd.read_csv('Mushroom_Test.csv')
df_train = pd.read_csv('Mushroom_Train.csv')
#find number of data with specific feature in each class
def calculate_number(df, col, val):
    classp = len(df[(df['class']=='p')&(df[col]==val)])
    classe = len(df[(df['class']=='e')&(df[col]==val)])
    return classp , classe
#calculate probability and choose class by probability and predict
def calculate_p(train, test):
    train_num = train['class'].value_counts()
    num_of_p = train_num[0]#number of all data with class p
    num_of_e = train_num[1]#number of all data with class e
    prior_p = num_of_p/sum(train_num)#find prior probability
    prior_e = num_of_e/sum(train_num)
    post_p = prior_p
    post_e = prior_e
    test = test.drop('class', axis = 1)
    columns = test.columns
    for col in columns:#use all features
        val = test[col].values
        p, e = calculate_number(train, col, val[0])
        v = df_test[col].nunique()
        temp_p = (p+1)/(num_of_p+v)
        temp_e = (e+1)/(num_of_e+v)
        post_p = post_p * temp_p
        post_e = post_e * temp_e
    #predict by probability
    if  post_p >= post_e:
        out = 'p'
    if post_e > post_p:
        out = 'e'
    return out

#predict all data test
def predict(train, test):
    preds = []
    for i in range(len(test)):
        temp = test[i:i+1]
        pred = calculate_p(train, temp)
        preds.append(pred)
    return preds
#use actual and predict list to plot confusion matrix
def confusion_matrix(pred, actual):
    Te = 0
    Tp = 0
    Fe = 0
    Fp = 0
    if  len(pred) != len(actual):
        print('prediction and actual arent same size')
        return None
    for i in range(len(pred)):
        if actual[i] == 'e':
            if pred[i] == 'e':
                Te = Te+1
            if pred[i] == 'p':
                Fp = Fp +1
        if actual[i] == 'p':
            if pred[i] == 'p':
                Tp = Tp +1
            if pred[i] == 'e':
                Fe = Fe +1
    total = Te + Tp + Fe + Fp
    ac = (Te+Tp)/total
    print('Accuracy = ', ac)
    print('                       Actual')
    print('                       p         e')
    print('              p      ', Tp, '     ' ,Fp)
    print('Predict')
    print('              e      ', Fe, '      ' ,Te)

p = predict(df_train, df_test)#predict test data
act = df_test['class']
act = list(act)
confusion_matrix(p,act)#plot confusion matrix