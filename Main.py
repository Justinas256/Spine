#describtion and more information about dataset can be found https://www.kaggle.com/sammy123/lower-back-pain-symptoms-dataset

from Graphs import *
from Models import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#reading data from file
data = pd.read_csv('spine.csv')

#------Preparing data-------

data.dropna(axis = 1, inplace = True)

#dividing dataset into features and target
x = data.iloc[:,:12]
y = data.iloc[:,12]

#string to int
y = [0 if e == 'Normal' else 1 for e in y]

#scaling
#scaler = StandardScaler()
#x = scaler.fit_transform(x)

#spliting data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 20)


#------Declaring functions-------

#printing basic information about dataset
def dataset_info():
    print(data.dtypes)
    print(data.shape)
    print(data.head())

def plot_hist():
    draw_hist(data)

def plot_heatmap():
    draw_heatmap(data)

def lr_model():
    model = logistic_regression(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    coef =  model.coef_
    print('Logistic regression accuracy on training set: ', train_score)
    print('Logistic regression accuracy on test set: ', test_score)
    plot_coefficients(list(data), coef)

def knn_model():
    model = knn(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print('KNN Classifier accuracy on training set: ', train_score)
    print('KNN Classifier accuracy on test set: ', test_score)

def svm_model():
    model = support_vector_machine(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    coef = model.coef_
    print('Support Vector Machine Classifier accuracy on training set: ', train_score)
    print('Support Vector Machine Classifier accuracy on test set: ', test_score)
    plot_coefficients(list(data), coef)

def rfc_model():
    model = random_forest(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    importances = model.feature_importances_
    print('Random Forest Classifier Classifier accuracy on training set: ', train_score)
    print('Random Forest Classifier Classifier accuracy on test set: ', test_score)
    plot_features_importances(list(data), importances)

#----------Switcher---------------

def numbers_to_functions(argument):
    switcher = {
        1: dataset_info,
        2: plot_hist,
        3: plot_heatmap,
        4: lr_model,
        5: svm_model,
        6: knn_model,
        7: rfc_model
    }
    func = switcher.get(argument)
    func()

#-------------------------------
print('1. Print basic information about dataset')
print('2. Draw histogram')
print('3. Draw features correlation heatmap')
print('4. Calculate logistic regression accuracy and plot features coefficients')
print('5. Calculate Support Vector Machine Classifier accuracy and plot features coefficients')
print('6. Calculate KNN Classifier accuracy')
print('7. Calculate Random Forest Classifier accuracy and plot features importances')
print('Type z to exit')
print('Enter: ')

command = input()

while command != 'z':
    try:
        command_int = int(command)
        if command_int < 1 or command_int > 7:
            raise ValueError
        print('Executing..')
        numbers_to_functions(command_int)
    except ValueError:
        print('Wrong input. Try again')
    print('Enter: ')
    command = input()

print('Bye bye!')
