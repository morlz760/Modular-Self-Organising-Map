# This script will hhelp me implemented the kfold valuation framework 

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


standard_som = create_train_som(data=X_train.values, n_features = 6, convolutional_layer=False)

def cros_validate(model, X, y, cvKFold):

    for train_index, test_index in cvKFold.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        

model_performance = cross_val_score(clf, X, y, cv=cvKFold)

print("{:.4f}".format(model_performance.mean())) 


standard_som = create_train_som(data=X_train.values, n_features = 6, convolutional_layer=False)

# area_som = trained_soms_layer_1['area'][0]
# X_test_area = np.array([[i] for i in X_train['area'].values])

# evaluate_purity(som, X_train, y_train)
cvKFold=StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

for train_index, test_index in cvKFold.split(X_train.values, y_train):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train_cv, X_test_cv = X_train.values[train_index], X_train.values[test_index]
    y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
    standard_som = create_train_som(data=X_train_cv, n_features = 6, convolutional_layer=False)
    final_purity = evaluate_purity(standard_som, X_test_cv, y_test_cv, convolutional_layer=False)
    print(final_purity)

# Single layer SOM 
evaluate_purity(standard_som, X_test.values, y_test)
