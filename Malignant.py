import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import matthews_corrcoef

torch.manual_seed(0)


cv_n_split = 5
random_state = 40
test_train_split_part = 0.3
metrics_all = {
    1: 'accuracy',
    2: 'precision',
    3: 'recall',
    4: 'F1_score',
    5: 'specificity',
    6: 'MCC',
    7: 'ROC_AUC'
}
metrics_now = [1, 2, 3, 4, 5, 6, 7]


data = pd.read_csv('C:/Users/dell/Desktop/feature/feature_40/Malignant_feature_train.csv')
main_cols = data.columns.tolist()
label = pd.read_csv('C:/Users/dell/Desktop/feature/feature_40/Malignant_train_label.csv')
test = pd.read_csv('C:/Users/dell/Desktop/feature/feature_40/Malignant_feature_val.csv')
target_test = pd.read_csv('C:/Users/dell/Desktop/feature/feature_40/Malignant_val_label.csv')


target_name = 'Malignant'
target = label[target_name]
target_test = target_test[target_name]
train = data[main_cols]
train, test = train[train.columns], test[train.columns]


kfold = KFold(n_splits=5, shuffle=True,random_state=14)


scaler2 = StandardScaler()
train = pd.DataFrame(scaler2.fit_transform(train), columns=train.columns)
test = pd.DataFrame(scaler2.fit_transform(test), columns=test.columns)



kfold = KFold(n_splits=5, shuffle=True,random_state=14)
acc_train_list = []
acc_test_list = []
# MinMaxScaler

svr_CV = SVC(probability=True,C=1.95,gamma=0.07,kernel='rbf',random_state=8)

score=cross_val_score(svr_CV,train,target,cv=kfold,scoring='roc_auc')

acc_train_list = []
acc_test_list = []
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
specificity_list = []
mcc_list = []
confusion_matrices = []
for i in range(5):

    print(f"fold:{i}")
    train_index = pd.read_csv('fold_new'+str(i)+'/'+'train_index.csv',header=None).iloc[:, 0].tolist()
    test_index = pd.read_csv('fold_new' +str(i) + '/' + 'test_index.csv',header=None).iloc[:, 0].tolist()

    this_train_x, this_train_y = train.iloc[train_index], target.iloc[train_index]  # 本组训练集
    this_test_x, this_test_y = train.iloc[test_index], target.iloc[test_index]  # 本组验证集
    # 训练本组的数据，并计算准确率
    svr_CV.fit(this_train_x, this_train_y)
    # 五折中训练集
    ytrain_pred = svr_CV.predict(this_train_x)
    ytest_pred = svr_CV.predict(this_test_x)

    ytrain_prob = svr_CV.predict_proba(this_train_x)[:, 1]
    ytest_prob = svr_CV.predict_proba(this_test_x)[:, 1]
    ##五折中训练集结果
    acc_train = round(roc_auc_score(this_train_y, ytrain_prob) * 100, 2)
    ##五折中内部验证集结果
    acc_test = round(roc_auc_score(this_test_y, ytest_prob) * 100, 2)
    # print(f"SVM train&test AUC: {acc_test}")
    # 存储每个分折的结果
    acc_train_list.append(acc_train)
    acc_test_list.append(acc_test)


    cm = confusion_matrix(this_test_y, ytest_pred)
    confusion_matrices.append(cm)

    acc_test = round(roc_auc_score(this_test_y, ytest_prob) * 100, 2)
    accuracy = round(accuracy_score(this_test_y, ytest_pred) * 100, 2)
    precision = round(precision_score(this_test_y, ytest_pred) * 100, 2)
    recall = round(recall_score(this_test_y, ytest_pred) * 100, 2)
    f1 = round(f1_score(this_test_y, ytest_pred) * 100, 2)
    tn, fp, fn, tp = confusion_matrix(this_test_y, ytest_pred).ravel()
    specificity = round((tn / (tn + fp)) * 100, 2)
    sensitivity = round((tp / (tp + fn)) * 100, 2)
    mcc = round(matthews_corrcoef(this_test_y, ytest_pred) * 100, 2)

    print(f"第{i}折")
    print(f"SVM internal Accuracy: {accuracy}")
    print(f"SVM internal Precision: {precision}")
    print(f"SVM internal Recall: {recall}")
    print(f"SVM internal F1 Score: {f1}")
    print(f"SVM internal Specificity: {specificity}")
    print(f"SVM internal sensitivity:{sensitivity}")
    print(f"SVM internal MCC: {mcc}")

    acc_test_list.append(acc_test)
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    specificity_list.append(specificity)
    mcc_list.append(mcc)
    #
    test_indices = np.arange(len(test))

    drop = [0,11,24,61,28,29,33,35,40,46,48,49,60,65,77,43,55]


    # 根据要删除的索引列表，过滤出需要保留的测试集索引
    to_keep_indices = list(filter(lambda num: num not in drop, test_indices))

    # 根据保留的索引获取新的测试集
    new_test = test.iloc[to_keep_indices]

    # 对新的测试集进行预测
    ytest_val_new = svr_CV.predict_proba(new_test)[:, 1]

    # 计算新的测试集上的 AUC
    acc_val_new = round(roc_auc_score(target_test[to_keep_indices], ytest_val_new) * 100, 2)
    # print(f"SVM val AUC: {acc_val_new}")
    ytest_val_pred = svr_CV.predict(new_test)

    cm_external = confusion_matrix(target_test[to_keep_indices], ytest_val_pred)

    # 计算模型在外部验证集上的性能指标
    ytest_val_new =svr_CV.predict_proba(test)[:, 1]
    acc_val_new = round(roc_auc_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    accuracy_val_new = round(accuracy_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    precision_val_new = round(precision_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    recall_val_new = round(recall_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    f1_val_new = round(f1_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    specificity_val_new = round((cm_external[0, 0] / (cm_external[0, 0] + cm_external[0, 1])) * 100, 2)
    mcc_val_new = round(matthews_corrcoef(target_test[to_keep_indices], ytest_val_pred) * 100, 2)

    # print(f"SVM val AUC: {acc_val_new}")
    # print(f"SVM val Accuracy: {accuracy_val_new}")
    # print(f"SVM val Precision: {precision_val_new}")
    # print(f"SVM val Recall: {recall_val_new}")
    # print(f"SVM val F1 Score: {f1_val_new}")
    # print(f"SVM val Specificity: {specificity_val_new}")
    # print(f"SVM val MCC: {mcc_val_new}")



avg_acc_test = np.mean(acc_test_list)
avg_accuracy = np.mean(accuracy_list)
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)
avg_f1 = np.mean(f1_list)
avg_specificity = np.mean(specificity_list)
avg_mcc = np.mean(mcc_list)

# print(f"Avg Test AUC: {avg_acc_test}")
# print(f"Avg ACCURACY: {avg_accuracy}")
# print(f"Avg Precision: {avg_precision}")
# print(f"Avg Recall: {avg_recall}")
# print(f"Avg F1 Score: {avg_f1}")
# print(f"Avg Specificity: {avg_specificity}")
# print(f"Avg MCC: {avg_mcc}")

acc_train_list = []
acc_test_list = []
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
specificity_list = []
mcc_list = []
confusion_matrices = []

# MLP
for i in range(5):
    print(f"fold:{i}")
    train_index = pd.read_csv('fold_new' + str(i) + '/' + 'train_index.csv', header=None).iloc[:, 0].tolist()
    test_index = pd.read_csv('fold_new' + str(i) + '/' + 'test_index.csv', header=None).iloc[:, 0].tolist()

    this_train_x, this_train_y = train.iloc[train_index], target.iloc[train_index]  # 本组训练集
    this_test_x, this_test_y = train.iloc[test_index], target.iloc[test_index]  # 本组验证

    mlp = MLPClassifier()
    param_grid = {'hidden_layer_sizes': [1],
                  'solver': ['lbfgs'],
                  'learning_rate': ['invscaling'],
                  'max_iter': [300]
                  }
    mlp_GS = GridSearchCV(mlp, param_grid=param_grid, verbose=False)

    mlp_GS.fit(this_train_x, this_train_y)

    ytrain_pred = mlp_GS.predict(this_train_x)
    ytest_pred = mlp_GS.predict(this_test_x)

    ytrain_prob = mlp_GS.predict_proba(this_train_x)[:, 1]
    ytest_prob = mlp_GS.predict_proba(this_test_x)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(this_test_y, ytest_pred)
    confusion_matrices.append(cm)


    acc_test = round(roc_auc_score(this_test_y, ytest_prob) * 100,2)
    accuracy = round(accuracy_score(this_test_y, ytest_pred) * 100,2)
    precision = round(precision_score(this_test_y, ytest_pred) * 100,2)
    recall = round(recall_score(this_test_y, ytest_pred) * 100,2)
    f1 = round(f1_score(this_test_y, ytest_pred) * 100,2)
    tn, fp, fn, tp = confusion_matrix(this_test_y, ytest_pred).ravel()
    specificity = round((tn / (tn + fp)) * 100, 2)
    sensitivity = round((tp / (tp + fn)) * 100, 2)
    mcc = round(matthews_corrcoef(this_test_y, ytest_pred) * 100,2)

    print(f"第{i}折")
    print(f"MLP internal Accuracy: {accuracy}")
    print(f"MLP internal Precision: {precision}")
    print(f"MLP internal Recall: {recall}")
    print(f"MLP internal F1 Score: {f1}")
    print(f"MLP internal Specificity: {specificity}")
    print(f"MLP internal Sensitivity: {sensitivity}")
    print(f"MLP internal MCC: {mcc}")

    acc_test_list.append(acc_test)
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    specificity_list.append(specificity)
    mcc_list.append(mcc)



    test_indices = np.arange(len(test))
    drop = [0, 11, 24, 61, 28, 29, 33, 35, 40, 46, 48, 49, 60, 65, 77, 43, 55]

    to_keep_indices = list(filter(lambda num: num not in drop, test_indices))
    new_test = test.iloc[to_keep_indices]

    ytest_val_new = mlp_GS.predict_proba(new_test)[:, 1]
    acc_val_new = round(roc_auc_score(target_test[to_keep_indices], ytest_val_new) * 100,2)
    # print(f"MLP val AUC: {acc_val_new}")

    ytest_val_pred = mlp_GS.predict(new_test)

    cm_external = confusion_matrix(target_test[to_keep_indices], ytest_val_pred)

    # 计算模型在外部验证集上的性能指标
    ytest_val_new = mlp_GS.predict_proba(test)[:, 1]
    acc_val_new = round(roc_auc_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    accuracy_val_new = round(accuracy_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    precision_val_new = round(precision_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    recall_val_new = round(recall_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    f1_val_new = round(f1_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    specificity_val_new = round((cm_external[0, 0] / (cm_external[0, 0] + cm_external[0, 1])) * 100, 2)
    mcc_val_new = round(matthews_corrcoef(target_test[to_keep_indices], ytest_val_pred) * 100, 2)

    # # print(f"MLP val AUC: {acc_val_new}")
    # print(f"MLP val Accuracy: {accuracy_val_new}")
    # print(f"MLP val Precision: {precision_val_new}")
    # print(f"MLP val Recall: {recall_val_new}")
    # print(f"MLP val F1 Score: {f1_val_new}")
    # print(f"MLP val Specificity: {specificity_val_new}")
    # print(f"MLP val MCC: {mcc_val_new}")


avg_acc_test = np.mean(acc_test_list)
avg_accuracy = np.mean(accuracy_list)
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)
avg_f1 = np.mean(f1_list)
avg_specificity = np.mean(specificity_list)
avg_mcc = np.mean(mcc_list)

# print(f"Avg Test AUC: {avg_acc_test}")
# print(f"Avg ACCURACY: {avg_accuracy}")
# print(f"Avg Precision: {avg_precision}")
# print(f"Avg Recall: {avg_recall}")
# print(f"Avg F1 Score: {avg_f1}")
# print(f"Avg Specificity: {avg_specificity}")
# print(f"Avg MCC: {avg_mcc}")


from sklearn.linear_model import LogisticRegression
acc_train_list = []
acc_test_list = []
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
specificity_list = []
mcc_list = []
confusion_matrices = []

# logistic
for i in range(5):
    print(f"fold:{i}")
    train_index = pd.read_csv('fold_new' + str(i) + '/' + 'train_index.csv', header=None).iloc[:, 0].tolist()
    test_index = pd.read_csv('fold_new' + str(i) + '/' + 'test_index.csv', header=None).iloc[:, 0].tolist()

    this_train_x, this_train_y = train.iloc[train_index], target.iloc[train_index]  # 本组训练集
    this_test_x, this_test_y = train.iloc[test_index], target.iloc[test_index]  # 本组验证

    logistic_model = LogisticRegression(C=5,max_iter=20, random_state=5)

    # Train the model
    logistic_model.fit(this_train_x,this_train_y)


    ytrain_pred = logistic_model.predict(this_train_x)
    ytest_pred = logistic_model.predict(this_test_x)

    ytrain_prob = logistic_model.predict_proba(this_train_x)[:, 1]
    ytest_prob = logistic_model.predict_proba(this_test_x)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(this_test_y, ytest_pred)
    confusion_matrices.append(cm)


    acc_test = round(roc_auc_score(this_test_y, ytest_prob) * 100,2)
    accuracy = round(accuracy_score(this_test_y, ytest_pred) * 100,2)
    precision = round(precision_score(this_test_y, ytest_pred) * 100,2)
    recall = round(recall_score(this_test_y, ytest_pred) * 100,2)
    f1 = round(f1_score(this_test_y, ytest_pred) * 100,2)
    tn, fp, fn, tp = confusion_matrix(this_test_y, ytest_pred).ravel()
    specificity = round((tn / (tn + fp)) * 100, 2)
    sensitivity = round((tp / (tp + fn)) * 100, 2)
    mcc = round(matthews_corrcoef(this_test_y, ytest_pred) * 100,2)

    print(f"第{i}折")
    print(f"Logistic Regression internal Accuracy: {accuracy}")
    print(f"Logistic Regression internal Precision: {precision}")
    print(f"Logistic Regression internal Recall: {recall}")
    print(f"Logistic Regression internal F1 Score: {f1}")
    print(f"Logistic Regression internal Specificity: {specificity}")
    print(f"Logistic Regression internal Sensitivity: {sensitivity}")
    print(f"Logistic Regression internal MCC: {mcc}")


    acc_test_list.append(acc_test)
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    specificity_list.append(specificity)
    mcc_list.append(mcc)



    test_indices = np.arange(len(test))
    drop = [0, 11, 24, 61, 28, 29, 33, 35, 40, 46, 48, 49, 60, 65, 77, 43, 55]

    to_keep_indices = list(filter(lambda num: num not in drop, test_indices))
    new_test = test.iloc[to_keep_indices]

    ytest_val_new = logistic_model.predict_proba(new_test)[:, 1]
    acc_val_new = round(roc_auc_score(target_test[to_keep_indices], ytest_val_new) * 100,2)
    # print(f"Logistic Regression val AUC: {acc_val_new}")

    ytest_val_pred = logistic_model.predict(new_test)

    cm_external = confusion_matrix(target_test[to_keep_indices], ytest_val_pred)

    # 计算模型在外部验证集上的性能指标
    ytest_val_new = logistic_model.predict_proba(test)[:, 1]
    acc_val_new = round(roc_auc_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    accuracy_val_new = round(accuracy_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    precision_val_new = round(precision_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    recall_val_new = round(recall_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    f1_val_new = round(f1_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    specificity_val_new = round((cm_external[0, 0] / (cm_external[0, 0] + cm_external[0, 1])) * 100, 2)
    mcc_val_new = round(matthews_corrcoef(target_test[to_keep_indices], ytest_val_pred) * 100, 2)

    # print(f"Logistic Regression val AUC: {acc_val_new}")
    # print(f"Logistic Regression val Accuracy: {accuracy_val_new}")
    # print(f"Logistic Regression val Precision: {precision_val_new}")
    # print(f"Logistic Regression val Recall: {recall_val_new}")
    # print(f"Logistic Regression val F1 Score: {f1_val_new}")
    # print(f"Logistic Regression val Specificity: {specificity_val_new}")
    # print(f"Logistic Regression val MCC: {mcc_val_new}")


avg_acc_test = np.mean(acc_test_list)
avg_accuracy = np.mean(accuracy_list)
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)
avg_f1 = np.mean(f1_list)
avg_specificity = np.mean(specificity_list)
avg_mcc = np.mean(mcc_list)


# print(f"Avg Test AUC: {avg_acc_test}")
# print(f"Avg ACCURACY: {avg_accuracy}")
# print(f"Avg Precision: {avg_precision}")
# print(f"Avg Recall: {avg_recall}")
# print(f"Avg F1 Score: {avg_f1}")
# print(f"Avg Specificity: {avg_specificity}")
# print(f"Avg MCC: {avg_mcc}")

acc_train_list = []
acc_test_list = []
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
specificity_list = []
mcc_list = []
confusion_matrices = []
from sklearn.tree import DecisionTreeClassifier
# Decision Tree
for i in range(5):
    print(f"fold:{i}")
    train_index = pd.read_csv('fold_new' + str(i) + '/' + 'train_index.csv', header=None).iloc[:, 0].tolist()
    test_index = pd.read_csv('fold_new' + str(i) + '/' + 'test_index.csv', header=None).iloc[:, 0].tolist()

    this_train_x, this_train_y = train.iloc[train_index], target.iloc[train_index]  # 本组训练集
    this_test_x, this_test_y = train.iloc[test_index], target.iloc[test_index]  # 本组验证

    tree_model = DecisionTreeClassifier(random_state=24)

    # Train the model
    tree_model.fit(this_train_x,this_train_y)


    ytrain_pred = tree_model.predict(this_train_x)
    ytest_pred = tree_model.predict(this_test_x)

    ytrain_prob = tree_model.predict_proba(this_train_x)[:, 1]
    ytest_prob = tree_model.predict_proba(this_test_x)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(this_test_y, ytest_pred)
    confusion_matrices.append(cm)


    acc_test = round(roc_auc_score(this_test_y, ytest_prob) * 100,2)
    accuracy = round(accuracy_score(this_test_y, ytest_pred) * 100,2)
    precision = round(precision_score(this_test_y, ytest_pred) * 100,2)
    recall = round(recall_score(this_test_y, ytest_pred) * 100,2)
    f1 = round(f1_score(this_test_y, ytest_pred) * 100,2)
    tn, fp, fn, tp = confusion_matrix(this_test_y, ytest_pred).ravel()
    specificity = round((tn / (tn + fp)) * 100, 2)
    sensitivity = round((tp / (tp + fn)) * 100, 2)
    mcc = round(matthews_corrcoef(this_test_y, ytest_pred) * 100,2)

    print(f"第{i}折")
    print(f"Decision Tree internal Accuracy: {accuracy}")
    print(f"Decision Tree internal Precision: {precision}")
    print(f"Decision Tree internal Recall: {recall}")
    print(f"Decision Tree internal F1 Score: {f1}")
    print(f"Decision Tree internal Specificity: {specificity}")
    print(f"Decision Tree internal Sensitivity: {sensitivity}")
    print(f"Decision Tree internal MCC: {mcc}")


    acc_test_list.append(acc_test)
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    specificity_list.append(specificity)
    mcc_list.append(mcc)



    test_indices = np.arange(len(test))
    drop = [0, 11, 24, 61, 28, 29, 33, 35, 40, 46, 48, 49, 60, 65, 77, 43, 55]

    to_keep_indices = list(filter(lambda num: num not in drop, test_indices))
    new_test = test.iloc[to_keep_indices]

    ytest_val_new = tree_model.predict_proba(new_test)[:, 1]
    acc_val_new = round(roc_auc_score(target_test[to_keep_indices], ytest_val_new) * 100,2)
    # print(f"Decision Tree val AUC: {acc_val_new}")

    ytest_val_pred = tree_model.predict(new_test)

    cm_external = confusion_matrix(target_test[to_keep_indices], ytest_val_pred)

    # 计算模型在外部验证集上的性能指标
    ytest_val_new = tree_model.predict_proba(test)[:, 1]
    acc_val_new = round(roc_auc_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    accuracy_val_new = round(accuracy_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    precision_val_new = round(precision_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    recall_val_new = round(recall_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    f1_val_new = round(f1_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    specificity_val_new = round((cm_external[0, 0] / (cm_external[0, 0] + cm_external[0, 1])) * 100, 2)
    mcc_val_new = round(matthews_corrcoef(target_test[to_keep_indices], ytest_val_pred) * 100, 2)

    # print(f"Decision Tree val AUC: {acc_val_new}")
    # print(f"Decision Tree val Accuracy: {accuracy_val_new}")
    # print(f"Decision Tree val Precision: {precision_val_new}")
    # print(f"Decision Tree val Recall: {recall_val_new}")
    # print(f"Decision Tree val F1 Score: {f1_val_new}")
    # print(f"Decision Tree val Specificity: {specificity_val_new}")
    # print(f"Decision Tree val MCC: {mcc_val_new}")



avg_acc_test = np.mean(acc_test_list)
avg_accuracy = np.mean(accuracy_list)
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)
avg_f1 = np.mean(f1_list)
avg_specificity = np.mean(specificity_list)
avg_mcc = np.mean(mcc_list)

# print(f"Avg Test AUC: {avg_acc_test}")
# print(f"Avg ACCURACY: {avg_accuracy}")
# print(f"Avg Precision: {avg_precision}")
# print(f"Avg Recall: {avg_recall}")
# print(f"Avg F1 Score: {avg_f1}")
# print(f"Avg Specificity: {avg_specificity}")
# print(f"Avg MCC: {avg_mcc}")


acc_train_list = []
acc_test_list = []
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
specificity_list = []
mcc_list = []
confusion_matrices = []
from sklearn.ensemble import RandomForestClassifier
# Random Forest
for i in range(5):
    print(f"fold:{i}")
    train_index = pd.read_csv('fold_new' + str(i) + '/' + 'train_index.csv', header=None).iloc[:, 0].tolist()
    test_index = pd.read_csv('fold_new' + str(i) + '/' + 'test_index.csv', header=None).iloc[:, 0].tolist()

    this_train_x, this_train_y = train.iloc[train_index], target.iloc[train_index]  # 本组训练集
    this_test_x, this_test_y = train.iloc[test_index], target.iloc[test_index]  # 本组验证

    rf_model = RandomForestClassifier(n_estimators=10, random_state=10)

    # Train the model
    rf_model.fit(this_train_x,this_train_y)


    ytrain_pred = rf_model.predict(this_train_x)
    ytest_pred = rf_model.predict(this_test_x)

    ytrain_prob = rf_model.predict_proba(this_train_x)[:, 1]
    ytest_prob = rf_model.predict_proba(this_test_x)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(this_test_y, ytest_pred)
    confusion_matrices.append(cm)


    acc_test = round(roc_auc_score(this_test_y, ytest_prob) * 100,2)
    accuracy = round(accuracy_score(this_test_y, ytest_pred) * 100,2)
    precision = round(precision_score(this_test_y, ytest_pred) * 100,2)
    recall = round(recall_score(this_test_y, ytest_pred) * 100,2)
    f1 = round(f1_score(this_test_y, ytest_pred) * 100,2)
    tn, fp, fn, tp = confusion_matrix(this_test_y, ytest_pred).ravel()
    specificity = round((tn / (tn + fp)) * 100, 2)
    sensitivity = round((tp / (tp + fn)) * 100, 2)
    mcc = round(matthews_corrcoef(this_test_y, ytest_pred) * 100,2)

    print(f"第{i}折")
    print(f"Random Forest internal Accuracy: {accuracy}")
    print(f"Random Forest internal Precision: {precision}")
    print(f"Random Forest internal Recall: {recall}")
    print(f"Random Forest internal F1 Score: {f1}")
    print(f"Random Forest internal Specificity: {specificity}")
    print(f"Random Forest internal Sensitivity: {sensitivity}")
    print(f"Random Forest internal MCC: {mcc}")


    acc_test_list.append(acc_test)
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    specificity_list.append(specificity)
    mcc_list.append(mcc)



    test_indices = np.arange(len(test))
    drop = [0, 11, 24, 61, 28, 29, 33, 35, 40, 46, 48, 49, 60, 65, 77, 43, 55]

    to_keep_indices = list(filter(lambda num: num not in drop, test_indices))
    new_test = test.iloc[to_keep_indices]

    ytest_val_new = rf_model.predict_proba(new_test)[:, 1]
    acc_val_new = round(roc_auc_score(target_test[to_keep_indices], ytest_val_new) * 100,2)
    # print(f"Random Forest val AUC: {acc_val_new}")

    ytest_val_pred = rf_model.predict(new_test)

    cm_external = confusion_matrix(target_test[to_keep_indices], ytest_val_pred)

    # 计算模型在外部验证集上的性能指标
    ytest_val_new = rf_model.predict_proba(test)[:, 1]
    acc_val_new = round(roc_auc_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    accuracy_val_new = round(accuracy_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    precision_val_new = round(precision_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    recall_val_new = round(recall_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    f1_val_new = round(f1_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    specificity_val_new = round((cm_external[0, 0] / (cm_external[0, 0] + cm_external[0, 1])) * 100, 2)
    mcc_val_new = round(matthews_corrcoef(target_test[to_keep_indices], ytest_val_pred) * 100, 2)

    # print(f"Random Forest val AUC: {acc_val_new}")
    # print(f"Random Forest val Accuracy: {accuracy_val_new}")
    # print(f"Random Forest val Precision: {precision_val_new}")
    # print(f"Random Forest val Recall: {recall_val_new}")
    # print(f"Random Forest val F1 Score: {f1_val_new}")
    # print(f"Random Forest val Specificity: {specificity_val_new}")
    # print(f"Random Forest val MCC: {mcc_val_new}")


avg_acc_test = np.mean(acc_test_list)
avg_accuracy = np.mean(accuracy_list)
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)
avg_f1 = np.mean(f1_list)
avg_specificity = np.mean(specificity_list)
avg_mcc = np.mean(mcc_list)

# print(f"Avg Test AUC: {avg_acc_test}")
# print(f"Avg ACCURACY: {avg_accuracy}")
# print(f"Avg Precision: {avg_precision}")
# print(f"Avg Recall: {avg_recall}")
# print(f"Avg F1 Score: {avg_f1}")
# print(f"Avg Specificity: {avg_specificity}")
# print(f"Avg MCC: {avg_mcc}")

acc_train_list = []
acc_test_list = []
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
specificity_list = []
mcc_list = []
confusion_matrices = []
from sklearn.ensemble import GradientBoostingClassifier
# Gradient Boosting
for i in range(5):
    print(f"fold:{i}")
    train_index = pd.read_csv('fold_new' + str(i) + '/' + 'train_index.csv', header=None).iloc[:, 0].tolist()
    test_index = pd.read_csv('fold_new' + str(i) + '/' + 'test_index.csv', header=None).iloc[:, 0].tolist()

    this_train_x, this_train_y = train.iloc[train_index], target.iloc[train_index]  # 本组训练集
    this_test_x, this_test_y = train.iloc[test_index], target.iloc[test_index]  # 本组验证

    gbm_model = GradientBoostingClassifier(learning_rate=0.5,n_estimators=10, random_state=10)

    # Train the model
    gbm_model.fit(this_train_x, this_train_y)


    ytrain_pred = gbm_model.predict(this_train_x)
    ytest_pred = gbm_model.predict(this_test_x)

    ytrain_prob = gbm_model.predict_proba(this_train_x)[:, 1]
    ytest_prob = gbm_model.predict_proba(this_test_x)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(this_test_y, ytest_pred)
    confusion_matrices.append(cm)


    acc_test = round(roc_auc_score(this_test_y, ytest_prob) * 100,2)
    accuracy = round(accuracy_score(this_test_y, ytest_pred) * 100,2)
    precision = round(precision_score(this_test_y, ytest_pred) * 100,2)
    recall = round(recall_score(this_test_y, ytest_pred) * 100,2)
    f1 = round(f1_score(this_test_y, ytest_pred) * 100,2)
    tn, fp, fn, tp = confusion_matrix(this_test_y, ytest_pred).ravel()
    specificity = round((tn / (tn + fp)) * 100, 2)
    sensitivity = round((tp / (tp + fn)) * 100, 2)
    mcc = round(matthews_corrcoef(this_test_y, ytest_pred) * 100,2)

    print(f"第{i}折")
    print(f"Gradient Boosting internal Accuracy: {accuracy}")
    print(f"Gradient Boosting internal Precision: {precision}")
    print(f"Gradient Boosting internal Recall: {recall}")
    print(f"Gradient Boosting internal F1 Score: {f1}")
    print(f"Gradient Boosting internal Specificity: {specificity}")
    print(f"Gradient Boosting internal Sensitivity: {sensitivity}")
    print(f"Gradient Boosting internal MCC: {mcc}")


    acc_test_list.append(acc_test)
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    specificity_list.append(specificity)
    mcc_list.append(mcc)



    test_indices = np.arange(len(test))
    drop = [0, 11, 24, 61, 28, 29, 33, 35, 40, 46, 48, 49, 60, 65, 77, 43, 55]

    to_keep_indices = list(filter(lambda num: num not in drop, test_indices))
    new_test = test.iloc[to_keep_indices]

    ytest_val_new = gbm_model.predict_proba(new_test)[:, 1]
    acc_val_new = round(roc_auc_score(target_test[to_keep_indices], ytest_val_new) * 100,2)
    # print(f"Gradient Boosting val AUC: {acc_val_new}")

    ytest_val_pred = gbm_model.predict(new_test)

    cm_external = confusion_matrix(target_test[to_keep_indices], ytest_val_pred)

    # 计算模型在外部验证集上的性能指标
    ytest_val_new = gbm_model.predict_proba(test)[:, 1]
    acc_val_new = round(roc_auc_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    accuracy_val_new = round(accuracy_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    precision_val_new = round(precision_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    recall_val_new = round(recall_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    f1_val_new = round(f1_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    specificity_val_new = round((cm_external[0, 0] / (cm_external[0, 0] + cm_external[0, 1])) * 100, 2)
    mcc_val_new = round(matthews_corrcoef(target_test[to_keep_indices], ytest_val_pred) * 100, 2)

    # print(f"Gradient Boosting val AUC: {acc_val_new}")
    # print(f"Gradient Boosting val Accuracy: {accuracy_val_new}")
    # print(f"Gradient Boosting val Precision: {precision_val_new}")
    # print(f"Gradient Boosting val Recall: {recall_val_new}")
    # print(f"Gradient Boosting val F1 Score: {f1_val_new}")
    # print(f"Gradient Boosting val Specificity: {specificity_val_new}")
    # print(f"Gradient Boosting val MCC: {mcc_val_new}")



avg_acc_test = np.mean(acc_test_list)
avg_accuracy = np.mean(accuracy_list)
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)
avg_f1 = np.mean(f1_list)
avg_specificity = np.mean(specificity_list)
avg_mcc = np.mean(mcc_list)

# print(f"Avg Test AUC: {avg_acc_test}")
# print(f"Avg ACCURACY: {avg_accuracy}")
# print(f"Avg Precision: {avg_precision}")
# print(f"Avg Recall: {avg_recall}")
# print(f"Avg F1 Score: {avg_f1}")
# print(f"Avg Specificity: {avg_specificity}")
# print(f"Avg MCC: {avg_mcc}")

acc_train_list = []
acc_test_list = []
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
specificity_list = []
mcc_list = []
confusion_matrices = []
from sklearn.neighbors import KNeighborsClassifier
# K-Nearest Neighbors
for i in range(5):
    print(f"fold:{i}")
    train_index = pd.read_csv('fold_new' + str(i) + '/' + 'train_index.csv', header=None).iloc[:, 0].tolist()
    test_index = pd.read_csv('fold_new' + str(i) + '/' + 'test_index.csv', header=None).iloc[:, 0].tolist()

    this_train_x, this_train_y = train.iloc[train_index], target.iloc[train_index]  # 本组训练集
    this_test_x, this_test_y = train.iloc[test_index], target.iloc[test_index]  # 本组验证

    knn_model = KNeighborsClassifier(n_neighbors=3)

    # Train the model
    knn_model.fit(this_train_x,this_train_y)


    ytrain_pred = knn_model.predict(this_train_x)
    ytest_pred = knn_model.predict(this_test_x)

    ytrain_prob = knn_model.predict_proba(this_train_x)[:, 1]
    ytest_prob = knn_model.predict_proba(this_test_x)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(this_test_y, ytest_pred)
    confusion_matrices.append(cm)


    acc_test = round(roc_auc_score(this_test_y, ytest_prob) * 100,2)
    accuracy = round(accuracy_score(this_test_y, ytest_pred) * 100,2)
    precision = round(precision_score(this_test_y, ytest_pred) * 100,2)
    recall = round(recall_score(this_test_y, ytest_pred) * 100,2)
    f1 = round(f1_score(this_test_y, ytest_pred) * 100,2)
    tn, fp, fn, tp = confusion_matrix(this_test_y, ytest_pred).ravel()
    specificity = round((tn / (tn + fp)) * 100, 2)
    sensitivity = round((tp / (tp + fn)) * 100, 2)
    mcc = round(matthews_corrcoef(this_test_y, ytest_pred) * 100,2)

    print(f"第{i}折")
    print(f"K-Nearest Neighbors internal Accuracy: {accuracy}")
    print(f"K-Nearest Neighbors internal Precision: {precision}")
    print(f"K-Nearest Neighbors internal Recall: {recall}")
    print(f"K-Nearest Neighbors internal F1 Score: {f1}")
    print(f"K-Nearest Neighbors internal Specificity: {specificity}")
    print(f"K-Nearest Neighbors internal Sensitivity: {sensitivity}")
    print(f"K-Nearest Neighbors internal MCC: {mcc}")


    acc_test_list.append(acc_test)
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    specificity_list.append(specificity)
    mcc_list.append(mcc)



    test_indices = np.arange(len(test))
    drop = [0, 11, 24, 61, 28, 29, 33, 35, 40, 46, 48, 49, 60, 65, 77, 43, 55]

    to_keep_indices = list(filter(lambda num: num not in drop, test_indices))
    new_test = test.iloc[to_keep_indices]

    ytest_val_new = knn_model.predict_proba(new_test)[:, 1]
    acc_val_new = round(roc_auc_score(target_test[to_keep_indices], ytest_val_new) * 100,2)
    # print(f"K-Nearest Neighbors val AUC: {acc_val_new}")

    ytest_val_pred = knn_model.predict(new_test)

    cm_external = confusion_matrix(target_test[to_keep_indices], ytest_val_pred)

    # 计算模型在外部验证集上的性能指标
    ytest_val_new = knn_model.predict_proba(test)[:, 1]
    acc_val_new = round(roc_auc_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    accuracy_val_new = round(accuracy_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    precision_val_new = round(precision_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    recall_val_new = round(recall_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    f1_val_new = round(f1_score(target_test[to_keep_indices], ytest_val_pred) * 100, 2)
    specificity_val_new = round((cm_external[0, 0] / (cm_external[0, 0] + cm_external[0, 1])) * 100, 2)
    mcc_val_new = round(matthews_corrcoef(target_test[to_keep_indices], ytest_val_pred) * 100, 2)

    # print(f"K-Nearest Neighbors val AUC: {acc_val_new}")
    # print(f"K-Nearest Neighbors val Accuracy: {accuracy_val_new}")
    # print(f"K-Nearest Neighbors val Precision: {precision_val_new}")
    # print(f"K-Nearest Neighbors val Recall: {recall_val_new}")
    # print(f"K-Nearest Neighbors val F1 Score: {f1_val_new}")
    # print(f"K-Nearest Neighbors val Specificity: {specificity_val_new}")
    # print(f"K-Nearest Neighbors val MCC: {mcc_val_new}")

    save_dir = "plot_1"
    # Plot Confusion Matrix for External Validation Set
    fig, ax = plt.subplots(figsize=(6, 4))  # 创建单个子图
    sns.heatmap(cm_external, annot=True, cmap='Blues', fmt='g', ax=ax, xticklabels=['No Tumor', 'Tumor'],
                yticklabels=['No Tumor', 'Tumor'])
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('K-Nearest Neighbors')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/K-Nearest Neighbors_fold_external_{i+1}.png")  # 保存外部验证集的混淆矩阵图像
    plt.close()  # 关闭当前图形

# Calculate average evaluation metrics

avg_acc_test = np.mean(acc_test_list)
avg_accuracy = np.mean(accuracy_list)
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)
avg_f1 = np.mean(f1_list)
avg_specificity = np.mean(specificity_list)
avg_mcc = np.mean(mcc_list)

# print(f"Avg Test AUC: {avg_acc_test}")
# print(f"Avg ACCURACY: {avg_accuracy}")
# print(f"Avg Precision: {avg_precision}")
# print(f"Avg Recall: {avg_recall}")
# print(f"Avg F1 Score: {avg_f1}")
# print(f"Avg Specificity: {avg_specificity}")
# print(f"Avg MCC: {avg_mcc}")