
import pandas as pd
import os
from sklearn.feature_selection import VarianceThreshold,  f_classif, SelectKBest



def date_process():

    date_train = pd.read_csv("C:/Users/dell/Desktop/train2.csv",index_col=0)
    date_val = pd.read_excel("C:/Users/dell/Desktop/val.xlsx",index_col=0)

    date_train_Malignant=date_train.dropna(subset=['Malignant'],axis=0, how='any')
    date_train_Malignant.drop('IA', axis=1,inplace=True)
    date_train_Malignant.drop('HI', axis=1, inplace=True)

    date_val_Malignant = date_val.dropna(subset=['Malignant'], axis=0, how='any')
    date_val_Malignant.drop('IA', axis=1, inplace=True)
    date_val_Malignant.drop('HI', axis=1, inplace=True)


    date_train_IA = date_train.dropna(subset=['IA'], axis=0, how='any')
    date_train_IA.drop('Malignant', axis=1,inplace=True)
    date_train_IA.drop('HI', axis=1, inplace=True)

    date_val_IA = date_val.dropna(subset=['IA'], axis=0, how='any')
    date_val_IA.drop('Malignant', axis=1,inplace=True)
    date_val_IA.drop('HI', axis=1, inplace=True)



    date_train_HI = date_train.dropna(subset=['HI'], axis=0, how='any')
    date_train_HI.drop('Malignant', axis=1,inplace=True)
    date_train_HI.drop('IA', axis=1, inplace=True)

    date_val_HI = date_val.dropna(subset=['HI'], axis=0, how='any')
    date_val_HI.drop('Malignant', axis=1,inplace=True)
    date_val_HI.drop('IA', axis=1, inplace=True)


    Malignant_feature_com = pd.concat([date_train_Malignant, date_val_Malignant], join='inner', axis=0)
    Malignant_feature_com_train = Malignant_feature_com.iloc[0:date_train_Malignant.shape[0]]
    Malignant_feature_com_val = Malignant_feature_com.iloc[date_train_Malignant.shape[0]:]


    IA_feature_com = pd.concat([date_train_IA, date_val_IA], join='inner', axis=0)
    IA_feature_com_train = IA_feature_com.iloc[0:date_train_IA.shape[0]]
    IA_feature_com_val = IA_feature_com.iloc[date_train_IA.shape[0]:]


    HI_feature_com = pd.concat([date_train_HI, date_val_HI], join='inner', axis=0)
    HI_feature_com_train = HI_feature_com.iloc[0:date_train_HI.shape[0]]
    HI_feature_com_val = HI_feature_com.iloc[date_train_HI.shape[0]:]

    Malignant_feature_com_train_label=pd.DataFrame(Malignant_feature_com_train['Malignant'])
    Malignant_feature_com_val_label = pd.DataFrame(Malignant_feature_com_val['Malignant'])

    IA_feature_com_train_label=pd.DataFrame(IA_feature_com_train['IA'])
    IA_feature_com_val_label = pd.DataFrame(IA_feature_com_val['IA'])

    HI_feature_com_train_label=pd.DataFrame(HI_feature_com_train['HI'])
    HI_feature_com_val_label = pd.DataFrame(HI_feature_com_val['HI'])


    Malignant_feature_com_train = Malignant_feature_com_train.drop(columns='Malignant')
    Malignant_feature_com_val = Malignant_feature_com_val.drop(columns='Malignant')

    IA_feature_com_train = IA_feature_com_train.drop(columns='IA')
    IA_feature_com_val = IA_feature_com_val.drop(columns='IA')

    HI_feature_com_train = HI_feature_com_train.drop(columns='HI')
    HI_feature_com_val = HI_feature_com_val.drop(columns='HI')



    # 过滤方差
    selMalignant = VarianceThreshold(threshold=0.01)
    selMalignant.fit(Malignant_feature_com_train)
    Malignant_feature_com_train_filter = Malignant_feature_com_train.iloc[:, selMalignant.get_support(True)]


    selIA = VarianceThreshold(threshold=0.01)
    selIA.fit(IA_feature_com_train)
    IA_feature_com_train_filter = IA_feature_com_train.iloc[:, selIA.get_support(True)]


    selHI = VarianceThreshold(threshold=0.01)
    selHI.fit(HI_feature_com_train)
    HI_feature_com_train_filter = HI_feature_com_train.iloc[:, selHI.get_support(True)]

    # os.makedirs(1500)
    feature_list=[40]
    for k in feature_list:
        path="feature1_"+str(k)
        if not os.path.exists(path):
            os.mkdir(path)
            print("目录已创建：", path)
        #调整特征数量
        selectorMalignant = SelectKBest(f_classif, k=k)
        selectorMalignant.fit(Malignant_feature_com_train_filter, Malignant_feature_com_train_label)
        Malignant_feature_com_train_selector = Malignant_feature_com_train_filter.iloc[:, selectorMalignant.get_support(True)]

        selectorIA = SelectKBest(f_classif, k=k)  # 0.05
        selectorIA.fit(IA_feature_com_train_filter, IA_feature_com_train_label)
        IA_feature_com_train_selector = IA_feature_com_train_filter.iloc[:, selectorIA.get_support(True)]


        selectorHI = SelectKBest(f_classif, k=k)  # 0.05
        selectorHI.fit(HI_feature_com_train_filter, HI_feature_com_train_label)
        HI_feature_com_train_selector = HI_feature_com_train_filter.iloc[:, selectorHI.get_support(True)]



        Malignant_feature_com_selector = pd.concat([Malignant_feature_com_train_selector, Malignant_feature_com_val], join='inner', axis=0)
        Malignant_feature_train = Malignant_feature_com_selector.iloc[0:Malignant_feature_com_train_selector.shape[0]]
        Malignant_feature_val = Malignant_feature_com_selector.iloc[Malignant_feature_com_train_selector.shape[0]:]

        IA_feature_com_selector = pd.concat([IA_feature_com_train_selector, IA_feature_com_val], join='inner', axis=0)
        IA_feature_train = IA_feature_com_selector.iloc[0:IA_feature_com_train_selector.shape[0]]
        IA_feature_val = IA_feature_com_selector.iloc[IA_feature_com_train_selector.shape[0]:]

        HI_feature_com_selector = pd.concat([HI_feature_com_train_selector, HI_feature_com_val], join='inner', axis=0)
        HI_feature_train = HI_feature_com_selector.iloc[0:HI_feature_com_train_selector.shape[0]]
        HI_feature_val = HI_feature_com_selector.iloc[HI_feature_com_train_selector.shape[0]:]

        print(Malignant_feature_train.shape, Malignant_feature_val.shape)
        print(Malignant_feature_com_train_label.shape, Malignant_feature_com_val_label.shape)

        print(IA_feature_train.shape, IA_feature_val.shape)
        print(IA_feature_com_train_label.shape, IA_feature_com_val_label.shape)

        print(HI_feature_train.shape, HI_feature_val.shape)
        print(HI_feature_com_train_label.shape, HI_feature_com_val_label.shape)


        Malignant_feature_train.to_csv(path+'/'+'Malignant_feature_train.csv', index=False)
        Malignant_feature_val.to_csv(path+'/'+'Malignant_feature_val.csv', index=False)
        Malignant_feature_com_train_label.to_csv(path+'/'+'Malignant_train_label.csv', index=False)
        Malignant_feature_com_val_label.to_csv(path+'/'+'Malignant_val_label.csv', index=False)
        IA_feature_train.to_csv(path+'/'+'IA_feature_train.csv', index=False)
        IA_feature_val.to_csv(path+'/'+'IA_feature_val.csv', index=False)
        IA_feature_com_train_label.to_csv(path+'/'+'IA_train_label.csv', index=False)
        IA_feature_com_val_label.to_csv(path+'/'+'IA_val_label.csv', index=False)


        HI_feature_train.to_csv(path+'/'+'HI_feature_train.csv', index=False)
        HI_feature_val.to_csv(path+'/'+'HI_feature_val.csv', index=False)
        HI_feature_com_train_label.to_csv(path+'/'+'HI_train_label.csv', index=False)
        HI_feature_com_val_label.to_csv(path+'/'+'HI_val_label.csv', index=False)



if __name__ == '__main__':
     date_process()
