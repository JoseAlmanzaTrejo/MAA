import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def fnc_histograma(data, variable, descripcion):
    try:
        print(pd.DataFrame(data.groupby(variable)['folioviv'].count()))
        resumen = pd.DataFrame(data.groupby(by=variable)['folioviv'].count()).reset_index().sort_values(by=variable)
        resumen.rename(columns={'folioviv':'registros'}, inplace=True)
        resumen.plot.bar(x=variable, 
                         y='registros', 
                         figsize=(15, 8), 
                         title=f'Histograma de {descripcion}')
        plt.savefig(f"Prácticas/Práctica3/img/hist_{descripcion}.png")
        plt.close()
        print(f'Se imprime la imagen hist_{descripcion}.png')
    except:
        print(f'No se puede imprimir la imagen hist_{descripcion}.png')

def fnc_resumen_viviendas(data, dict_data):

    data_copy = data.loc[:,dict_data.keys()].replace(['',' ','&'],-999999)
    print(data_copy.describe())

    for variable, descripcion in dict_data.items():
        n_valores = len(data_copy.loc[:,variable].unique())
        fnc_histograma(data, variable, descripcion)

def fnc_imputador_viviendas(data, dict_data):

    data_copy = data.loc[:,dict_data.keys()].replace(['',' ','&'],np.nan)
    imp = SimpleImputer(strategy="most_frequent")
    data_copy = pd.DataFrame(imp.fit_transform(data_copy.values)).rename(columns={i:columna for i, columna in enumerate(data_copy.columns)})
    return data_copy.astype(int)

def fnc_knn_classifier_model(data: pd.DataFrame, X: list, y: str, test_size: float=0.3, neighbors: int=5) -> dict:
    print('Usando sklearn KNN Classifier model predict')
    dict_estadisticos = dict()

    #Partimos el dataset en train y test para hacer Validación Cruzada
    X_train, X_test, y_train, y_test = train_test_split(data[X], data[y],test_size=test_size, random_state = 123, stratify=data[y])

    for n_neigbors in neighbors:
        trained_knn_model = KNeighborsClassifier(n_neighbors=n_neigbors)
        trained_knn_model.fit(X_train, y_train)
        score_test = trained_knn_model.score(X_test, y_test)
        score_train = trained_knn_model.score(X_train, y_train)
        y_pred = trained_knn_model.predict(X_test)
        print(pd.pivot_table(pd.concat([y_test, pd.DataFrame(y_pred)], axis=1).rename(columns={0:'y_pred'}).reset_index(), index=y, columns='y_pred', values='index', aggfunc='count'))
        print(f"accuracy score knn classifier n_neighbors {n_neigbors}: [train {score_train:.4}, test {score_test:.4}]")

        #Guardamos la confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=trained_knn_model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=trained_knn_model.classes_)
        disp.plot()
        plt.savefig(f"Prácticas/Práctica3/img/Confusion Matrix KNN n_neighbors {n_neigbors}.png")
        plt.close()

        dict_estadisticos[f'accuracy train n_neighbors: {n_neigbors}'] = score_train
        dict_estadisticos[f'accuracy test n_neighbors: {n_neigbors}']  = score_test
        
    return dict_estadisticos

def fnc_decision_tree_classifier_model(data: pd.DataFrame, X: list, y: str, test_size: float=0.3, depths:list=[5], min_samples_leafs: list=[5], min_samples_splits: list=[6]) -> dict:
    print('Usando sklearn KNN Classifier model predict')
    dict_estadisticos = dict()

    #Partimos el dataset en train y test para hacer Validación Cruzada
    X_train, X_test, y_train, y_test = train_test_split(data[X], data[y],test_size=test_size, random_state = 123, stratify=data[y])

    for depth in depths:
        for min_samples_leaf in min_samples_leafs:
            for min_samples_split in min_samples_splits:
                dt_classifier = DecisionTreeClassifier(random_state=123, max_depth=depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
                trained_dt_model = OneVsRestClassifier(estimator=dt_classifier)
                trained_dt_model.fit(X_train, y_train)
                score_test = trained_dt_model.score(X_test, y_test)
                score_train = trained_dt_model.score(X_train, y_train)
                y_pred = trained_dt_model.predict(X_test)
                print(pd.pivot_table(pd.concat([y_test, pd.DataFrame(y_pred)], axis=1).rename(columns={0:'y_pred'}).reset_index(), index=y, columns='y_pred', values='index', aggfunc='count'))
                print(f"accuracy score decision tree depth {depth} min_samples_leaf {min_samples_leaf} min_samples_split {min_samples_split}: [train {score_train:.4}, test {score_test:.4}]")

                #Guardamos la confusion matrix
                cm = confusion_matrix(y_test, y_pred, labels=trained_dt_model.classes_)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=trained_dt_model.classes_)
                disp.plot()
                plt.savefig(f"Prácticas/Práctica3/img/Confusion Matrix Decision tree depth {depth} min_samples_leaf {min_samples_leaf} min_samples_split {min_samples_split}.png")
                plt.close()

                dict_estadisticos[f'accuracy train depth {depth} min_s_s {min_samples_split} min_s_l {min_samples_leaf}'] = score_train
                dict_estadisticos[f'accuracy test depth {depth} min_s_s {min_samples_split} min_s_l {min_samples_leaf}']  = score_test    

    return dict_estadisticos

def fnc_svm_classifier_model(data: pd.DataFrame, X: list, y: str, test_size: float=0.3, C_list:list=[1.0]) -> dict:
    print('Usando sklearn SVM Classifier model predict')
    dict_estadisticos = dict()

    #Partimos el dataset en train y test para hacer Validación Cruzada
    X_train, X_test, y_train, y_test = train_test_split(data[X], data[y],test_size=test_size, random_state = 123, stratify=data[y])

    for C in C_list:
        svm_classifier = SVC(random_state=123, C=C)
        trained_svm_model = OneVsRestClassifier(estimator=svm_classifier)
        trained_svm_model.fit(X_train, y_train)
        score_test = trained_svm_model.score(X_test, y_test)
        score_train = trained_svm_model.score(X_train, y_train)
        y_pred = trained_svm_model.predict(X_test)
        print(pd.pivot_table(pd.concat([y_test, pd.DataFrame(y_pred)], axis=1).rename(columns={0:'y_pred'}).reset_index(), index=y, columns='y_pred', values='index', aggfunc='count'))
        print(f"accuracy score Support Vector Machines C {C}: [train {score_train:.4}, test {score_test:.4}]")

        #Guardamos la confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=trained_svm_model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=trained_svm_model.classes_)
        disp.plot()
        plt.savefig(f"Prácticas/Práctica3/img/Confusion Matrix Support Vector Machines C {C}.png")
        plt.close()

        dict_estadisticos[f'accuracy train C {C}'] = score_train
        dict_estadisticos[f'accuracy test C {C}']  = score_test    

    return dict_estadisticos

def fnc_log_reg_classifier_model(data: pd.DataFrame, X: list, y: str, test_size: float=0.3, C_list:list=[1.0]) -> dict:
    print('Usando sklearn Logistic Regression Classifier model predict')
    dict_estadisticos = dict()

    #Partimos el dataset en train y test para hacer Validación Cruzada
    X_train, X_test, y_train, y_test = train_test_split(data[X], data[y],test_size=test_size, random_state = 123, stratify=data[y])

    for C in C_list:
        log_reg_classifier = LogisticRegression(random_state=123, C=C)
        trained_log_reg_model = OneVsRestClassifier(estimator=log_reg_classifier)
        trained_log_reg_model.fit(X_train, y_train)
        score_test = trained_log_reg_model.score(X_test, y_test)
        score_train = trained_log_reg_model.score(X_train, y_train)
        y_pred = trained_log_reg_model.predict(X_test)
        print(pd.pivot_table(pd.concat([y_test, pd.DataFrame(y_pred)], axis=1).rename(columns={0:'y_pred'}).reset_index(), index=y, columns='y_pred', values='index', aggfunc='count'))
        print(f"accuracy score Logistic Regression C {C}: [train {score_train:.4}, test {score_test:.4}]")

        #Guardamos la confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=trained_log_reg_model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=trained_log_reg_model.classes_)
        disp.plot()
        plt.savefig(f"Prácticas/Práctica3/img/Confusion Matrix Logistic Regression C {C}.png")
        plt.close()

        dict_estadisticos[f'accuracy train C {C}'] = score_train
        dict_estadisticos[f'accuracy test C {C}']  = score_test    

    return dict_estadisticos

if __name__ == "__main__":
    ##Leemos la información que utilizaremos para la tarea 2
    
    ##Leemos la información de la tabla de viviendas
    viviendas = pd.read_csv('csv/65_enigh2022_ns_viviendas_csv/viviendas.csv')
    llaves=['folioviv']
    variable_objetivo='est_socio'
    dict_variables_descriptivas = {'est_socio':'estrato socioeconómico',
        'tipo_viv':'tipo de vivienda',
        'mat_pared':'material de pared',
        'mat_techos':'material de techos',
        'mat_pisos':'material de pisos',
        'antiguedad':'antigüedad de la vivienda',
        'cocina':'cuenta con cocina',
        'cuart_dorm':'dormitorios',
        'num_cuarto':'número de cuartos',
        'disp_agua':'forma de abastecimiento de agua',
        'excusado':'tiene excusado',
        'bano_comp':'cuantos baños completos tiene',
        'bano_excus':'cuantos baños con excusado tiene',
        'bano_regad':'cuantos baños con regadera tiene',
        'drenaje':'destino del drenaje',
        'disp_elect':'fuente de donde se obtiene energía eléctrica',
        'focos_inca':'número de focos incandescentes',
        'focos_ahor':'número de focos ahorradores',
        'combustible':'tipo de combustible usado en la cocina',
        'tipo_adqui':'tipo de adquisición de la vivienda',
        'tipo_finan':'tipo de financiamiento',
        'calent_sol':'cuenta con calentador solar',
        'calent_gas':'cuenta con calentador de gas',
        'medidor_luz':'cuenta con medidor de luz',
        'bomba_agua':'cuenta con bomba de agua',
        'tanque_gas':'cuenta con tanque de gas',
        'aire_acond':'cuenta con aire acondicionado',
        'calefacc':'cuenta con calefaccion',
        'tot_resid':'número de residentes en la vivienda',
        'tot_hom':'número de residentes hombres en la vivienda',
        'tot_muj':'número de mujeres en la vivienda',
        'tot_hog':'número de hogares en la vivienda',
        'tam_loc':'tamaño de la localidad'}
    #fnc_resumen_viviendas(viviendas, dict_variables_descriptivas)
    #imputamos los valores no válidos con la moda
    viviendas = fnc_imputador_viviendas(viviendas, dict_variables_descriptivas)

    #Corremos un modelo KNN para predecir la etiqueta de estrato socioeconómico con las características de la vivienda
    X_vars = [variable for variable in dict_variables_descriptivas.keys() if variable!='est_socio']
    accuracy_kmm_classifier = fnc_knn_classifier_model(data = viviendas, X = X_vars, y = variable_objetivo, test_size=0.4, neighbors=[1,2,3,4,5,6])

    #Corremos un modelo de Decision Tree para predecir la etiqueta de estrato socioeconómico 
    accuracy_decision_tree_classifier = fnc_decision_tree_classifier_model(data = viviendas, X = X_vars, y = variable_objetivo, test_size=0.4, depths=[i for i in range(1,50,5)], min_samples_leafs=range(50,500,50), min_samples_splits=range(50,500,50))

    #Corremos un modelo de Support Vector Machines para predecir la etiqueta de estrato socioeconómico 
    accuracy_svm_classifier = fnc_svm_classifier_model(data = viviendas, X = X_vars, y = variable_objetivo, test_size=0.4, C_list=[1.0,0.5])

    #Corremos un modelo de Regresión Logística para predecir la etiqueta de estrato socioeconómico 
    accuracy_log_reg_classifier = fnc_log_reg_classifier_model(data = viviendas, X = X_vars, y = variable_objetivo, test_size=0.4, C_list=[1.0,0.9,0.8,0.7,0.6,0.5])

    mejor_modelo = ''
    acc_mejor_modelo = 0
    for llave, valor in accuracy_kmm_classifier.items():
        if llave.find('test')>0:
            if acc_mejor_modelo<valor:
                mejor_modelo=llave
                acc_mejor_modelo=valor
                print(llave+':'+str(valor))
    mejor_modelo = ''
    acc_mejor_modelo = 0
    for llave, valor in accuracy_decision_tree_classifier.items():
        if llave.find('test')>0:
            if acc_mejor_modelo<valor:
                mejor_modelo=llave
                acc_mejor_modelo=valor
                print(llave+':'+str(valor))
    mejor_modelo = ''
    acc_mejor_modelo = 0
    for llave, valor in accuracy_svm_classifier.items():
        if llave.find('test')>0:
            if acc_mejor_modelo<valor:
                mejor_modelo=llave
                acc_mejor_modelo=valor
                print(llave+':'+str(valor))
    mejor_modelo = ''
    acc_mejor_modelo = 0
    for llave, valor in accuracy_log_reg_classifier.items():
        if llave.find('test')>0:
            if acc_mejor_modelo<valor:
                mejor_modelo=llave
                acc_mejor_modelo=valor
                print(llave+':'+str(valor))

    accuracy_kmm_classifier['accuracy train n_neighbors: 6']
    accuracy_decision_tree_classifier['accuracy train depth 11 min_s_s 50 min_s_l 50']

    accuracy_log_reg_classifier['accuracy train C 0.9']