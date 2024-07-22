import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, silhouette_score
from sklearn.cluster import KMeans

def fnc_imputador_viviendas(data, dict_data):

    data_copy = data.loc[:,dict_data.keys()].replace(['',' ','&'],np.nan)
    imp = SimpleImputer(strategy="most_frequent")
    data_copy = pd.DataFrame(imp.fit_transform(data_copy.values)).rename(columns={i:columna for i, columna in enumerate(data_copy.columns)})
    return data_copy.astype(int)

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
                plt.savefig(f"Prácticas/Práctica4/img/Confusion Matrix Decision tree depth {depth} min_samples_leaf {min_samples_leaf} min_samples_split {min_samples_split}.png")
                plt.close()

                dict_estadisticos[f'accuracy train depth {depth} min_s_s {min_samples_split} min_s_l {min_samples_leaf}'] = score_train
                dict_estadisticos[f'accuracy test depth {depth} min_s_s {min_samples_split} min_s_l {min_samples_leaf}']  = score_test    

    return dict_estadisticos

def fnc_kmeans_cluster_model(data: pd.DataFrame, X: list, y: str, test_size: float=0.3, clusters:list=[2]) -> dict:
    print('Usando sklearn KMeans Cluster model predict')
    dict_estadisticos = dict()

    #Partimos el dataset en train y test para hacer Validación Cruzada
    X_train, X_test, y_train, y_test = train_test_split(data[X], data[y],test_size=test_size, random_state = 123, stratify=data[y])

    for n_clusters in clusters:
        kmeans_cluster = KMeans(n_clusters=n_clusters, random_state=123)
        kmeans_cluster.fit(X_train)
        y_pred_train = kmeans_cluster.predict(X_train)+1
        y_pred_test = kmeans_cluster.predict(X_test)+1
        
        score_train = accuracy_score(y_train, y_pred_train)
        score_test = accuracy_score(y_test, y_pred_test)

        silhouette_score_train = float(silhouette_score(X_train,y_pred_train))
        silhouette_score_test = float(silhouette_score(X_test,y_pred_test))
        
        print(pd.pivot_table(pd.concat([y_test, pd.DataFrame(y_pred_test)], axis=1).rename(columns={0:'y_pred'}).reset_index(), index=y, columns='y_pred', values='index', aggfunc='count'))
        print(f"accuracy score kmeans cluster n_clusters {n_clusters}: [train {score_train:.4}, test {score_test:.4}]")
        print(f"silhouette score kmeans cluster n_clusters {n_clusters}: [train {silhouette_score_train:.4}, test {silhouette_score_test:.4}]")
        

        #Guardamos la confusion matrix
        cm = confusion_matrix(y_train, y_pred_train)
        print(cm)
        cm = confusion_matrix(y_test, y_pred_test)
        print(cm)
        print(np.sort(y_pred_train))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(f"Prácticas/Práctica4/img/Confusion Matrix KMeans n_clusters {n_clusters}.png")
        plt.close()

        dict_estadisticos[f'accuracy train n_clusters {n_clusters}'] = score_train
        dict_estadisticos[f'accuracy test n_clusters {n_clusters}']  = score_test

        dict_estadisticos[f'silhouette train n_clusters {n_clusters}'] = silhouette_score_train
        dict_estadisticos[f'silhouette test n_clusters {n_clusters}']  = silhouette_score_test    

    return dict_estadisticos

def fnc_grafico_codo_kmeans(data, X, clusters):

    dict_estadisticos = dict()
    for n_clusters in clusters:

        kmeans_cluster = KMeans(n_clusters=n_clusters, random_state=123)
        silhouette_score_X = silhouette_score(data[X],kmeans_cluster.fit_predict(data[X]))
        print(f'Score silhouette {silhouette_score_X} para n_clusters {n_clusters}')

        dict_estadisticos[n_clusters] = silhouette_score_X

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

    #Corremos un modelo de Decision Tree para predecir la etiqueta de estrato socioeconómico 
    X_vars = [variable for variable in dict_variables_descriptivas.keys() if variable!='est_socio']
    accuracy_decision_tree_classifier = fnc_decision_tree_classifier_model(data = viviendas, X = X_vars, y = variable_objetivo, test_size=0.4, depths=[11], min_samples_leafs=[50], min_samples_splits=[50])

    accuracy_kmeans_cluster = fnc_kmeans_cluster_model(data = viviendas, X = X_vars, y = variable_objetivo, test_size=0.4, clusters=[4])
    #Se recalculan los accuracy de train y test en excel debido a que no empatan los clústeres generados con la variable est_socio

    dict_estadisticos = fnc_grafico_codo_kmeans(data = viviendas, X = X_vars, clusters=[2,3,4,5,6,7,8,9,10])


    for llave, variable in dict_estadisticos.items():
        print(f'Para n_clusters {llave} tenemos un score_silhouette de {variable}')

    plt.figure()
    plt.plot(list(dict_estadisticos.keys()), list(dict_estadisticos.values()))
    plt.xlabel("k = number of cluster")
    plt.ylabel("Silhouette Score")
    plt.savefig(f"Prácticas/Práctica4/img/Silhouette score para selección de k en kmeans clúster.png")
    plt.close()