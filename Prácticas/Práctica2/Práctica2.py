import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import math
from scipy import stats
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor 

import matplotlib.pyplot as plt

def fnc_asigna_corte(data, cortes, variable, redondeo):
    for corte in cortes:
        if data.loc[variable] <= corte:
            return round(corte,redondeo)

def fnc_histograma(data, variable, descripcion, redondeo=0, particiones=None):
    try:
        data_copy = data[~data.loc[:,variable].isna()]
        minimo = data_copy.loc[:,variable].min()
        maximo = data_copy.loc[:,variable].max()
        rango = maximo - minimo
        if particiones:
            n_particiones = particiones
        else:
            n_particiones = math.ceil(math.sqrt(len(data_copy)))
            
        rango_particion = rango/n_particiones

        cortes = [minimo + (i + 1) * rango_particion for i in range(n_particiones)]

        data_copy.loc[:,'corte'] = data_copy.apply(fnc_asigna_corte, cortes=cortes, variable=variable, redondeo=redondeo, axis=1)

        resumen = pd.DataFrame(data_copy.groupby(by='corte')[variable].count()).reset_index().sort_values(by='corte')
        resumen.rename(columns={variable:'registros'}, inplace=True)
        resumen.plot.bar(x='corte', 
                         y='registros', 
                         figsize=(20, 10), 
                         title=f'Histograma de {descripcion}')
        plt.savefig(f"Prácticas/Práctica2/img/hist_{descripcion}.png")
        plt.close()
    except:
        print(f'No se puede imprimir la imagen hist_{descripcion}.png')

def fnc_resumen_ingresos(data):
    
    ingreso_por_salario = data.clave=='P001'
    personas = len(data.loc[:,['folioviv','foliohog','numren']].drop_duplicates())
    print(f'Número de personas que tienen algún tipo de ingreso en el hogar: {personas:,.0f}')
    personas_salarios = len(data.loc[ingreso_por_salario,['folioviv','foliohog','numren']].drop_duplicates())
    print(f'Número de personas que tienen un ingreso por salarios: {personas_salarios:,.0f}')
    
    #Obtenemos el gráfico del histograma de ingresos por salarios así como su logaritmo natural
    fnc_histograma(data.loc[ingreso_por_salario,:], variable='ing_tri', descripcion='ingreso trimestral por salarios', redondeo=1, particiones=None)
    print('Se imprime histograma de ingreso trimestral por salarios.')
    fnc_histograma(data.loc[ingreso_por_salario,:], variable='ln_ing_tri', descripcion='logaritmo natural del ingreso trimestral por salarios', redondeo=5, particiones=100)
    print('Se imprime histograma de logaritmo natural de los ingresos trimestrales por salarios.')
    fnc_histograma(data.loc[ingreso_por_salario,:], variable='entidad', descripcion='entidad donde trabaja', redondeo=0, particiones=32)
    print('Se imprime histograma de entidad donde trabaja.')

def fnc_resumen_trabajos(data):
    
    trabajo_principal = data.id_trabajo==1
    personas = len(data.loc[trabajo_principal,['folioviv','foliohog','numren']].drop_duplicates())
    print(f'Número de personas que tienen algún tipo de trabajo principal: {personas:,.0f}')
    horas_trabajadas = data.loc[trabajo_principal,'htrab'].mean()
    print(f'Número de horas trabajadas en promedio en trabajo principal: {horas_trabajadas:,.2f}')

    #Obtenemos el gráfico del histograma de número de horas trabajadas
    fnc_histograma(data.loc[trabajo_principal,:], variable='htrab', descripcion='horas trabajadas a la semana', redondeo=0, particiones=84)
    print('Se imprime histograma del número de horas trabajadas a la semana')

def fnc_poblacion(data):

    poblacion = len(data)
    print(f'Tenemos {poblacion:,.0f} personas identificadas en la encuesta.')
    genero_h, genero_m = len(data[data.sexo==1]), len(data[data.sexo==2])
    print(f'Hay {genero_h:,.0f} hombres y {genero_m:,.0f} mujeres identificados en la encuesta.')
    edad_promedio, edad_promedio_h, edad_promedio_m = data.edad.mean(), data.loc[data.sexo==1,'edad'].mean(), data.loc[data.sexo==2,'edad'].mean()
    print(f'La edad promedio de las personas en la encuesta es de {edad_promedio:,.2f} años, {edad_promedio_h:,.2f} años hombres y {edad_promedio_m:,.2f} años mujeres')
    hijos = data[data.hijos>0].hijos.mean()
    print(f'Número de hijos promedio en la encuesta  (sin contar los que no tienen hijos): {hijos:,.2f}')
    con_estudios = sum(data.nivelaprob>0)
    print(f'Número de personas con algún grado de estudios (conocido): {con_estudios:,.0f}')

    #Obtenemos el gráfico del histograma de edad y número de hijos
    fnc_histograma(data, variable='edad', descripcion='edad', redondeo=0, particiones=25)
    print('Se imprime histograma de la edad.')
    fnc_histograma(data[data.sexo==1], variable='edad', descripcion='edad por género masculino', redondeo=0, particiones=25)
    print('Se imprime histograma de la edad por género masculino.')
    fnc_histograma(data[data.sexo==2], variable='edad', descripcion='edad por género femenino', redondeo=0, particiones=25)
    print('Se imprime histograma de la edad por género femenino.')
    fnc_histograma(data[data.hijos>0], variable='hijos', descripcion='número de hijos', redondeo=0, particiones=5)
    print('Se imprime histograma del número de hijos.')
    fnc_histograma(data, variable='nivelaprob', descripcion='nivel de estudios aprobado', redondeo=0, particiones=10)
    print('Se imprime histograma del nivel de estudios aprobado.')    
    fnc_histograma(data, variable='edo_conyug', descripcion='estado conyugal', redondeo=0, particiones=6)
    print('Se imprime histograma del estado conyugal.')

def fnc_encode_nivelaprob(data):

    enc = OneHotEncoder(handle_unknown='ignore')
    encoding =  enc.fit_transform(data.nivelaprob.values.reshape(-1, 1)).toarray()
    encoding = pd.DataFrame(encoding).rename(columns={i:x for i, x in enumerate(enc.get_feature_names_out(['nivel']))})
    encoding = encoding.loc[:,[columna for columna in encoding.columns if columna!='nivel_nan']]

    return pd.concat([data, encoding], axis=1)

def fnc_linear_regression_model(data: pd.DataFrame, X: list, y: str, test_size: float=0.3) -> dict:
    print('Usando sklearn Linear Regression model predict')
    
    #Partimos el dataset en train y test para hacer Validación Cruzada
    X_train, X_test, y_train, y_test = train_test_split(data[X], data[y],test_size=test_size, random_state = 123)
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    print(model_lr.coef_)
    #coef, intercept = model_lr.coef_, model_lr.intercept_
    #for coeficiente, variable in zip(coef, X):
    #    print(f"coef de {variable}: {coeficiente}")
    
    #print(f"intercept: {intercept}", flush=True)
    print('R-squared score (training): {:.9f}'.format(model_lr.score(X_train, y_train)))
    print('R-squared score (test): {:.9f}'.format(model_lr.score(X_test, y_test)))

    return {'R-squared train lr':model_lr.score(X_train, y_train), 'R-squared test lr':model_lr.score(X_test, y_test)}

def fnc_polynomial_model(data: pd.DataFrame, X: list, y: str, test_size: float=0.3, grados: list=[1]) -> dict:
    dict_estadisticos = dict()
    print('Usando sklearn Polynomial model predict')
    
    #Partimos el dataset en train y test para hacer Validación Cruzada
    X_train, X_test, y_train, y_test = train_test_split(data[X], data[y],test_size=test_size, random_state = 123)
    
    for grado in grados:
        model_poly = PolynomialFeatures(degree=grado)
        X_transformed_train = model_poly.fit_transform(X_train)
        X_transformed_test = model_poly.fit_transform(X_test)
        model_poly = LinearRegression()
        model_poly.fit(X_transformed_train, y_train)
        #coef, intercept = model_poly.coef_, model_poly.intercept_
        # for coeficiente, variable in zip(coef, X):
        #    print(f"coef de {variable}: {coeficiente}")
        
        #print(f"intercept: {intercept}", flush=True)
        print('R-squared score poly{:,.0f} (training): {:.9f}'.format(grado,model_poly.score(X_transformed_train, y_train)))
        print('R-squared score poly{:,.0f} (test): {:.9f}'.format(grado,model_poly.score(X_transformed_test, y_test)))

        dict_estadisticos[f'R-squared train poly{grado}'] = model_poly.score(X_transformed_train, y_train)
        dict_estadisticos[f'R-squared test poly{grado}']  = model_poly.score(X_transformed_test,  y_test) 

    return dict_estadisticos

def fnc_polynomial_ridge_model(data: pd.DataFrame, X: list, y: str, test_size: float=0.3, grados: list=[1], alphas: list=[1]) -> dict:
    dict_estadisticos = dict()
    print('Usando sklearn Polynomial Ridge model predict')
    
    #Partimos el dataset en train y test para hacer Validación Cruzada
    X_train, X_test, y_train, y_test = train_test_split(data[X], data[y],test_size=test_size, random_state = 123)
    
    for grado in grados:
        for alpha in alphas:
            model_poly = PolynomialFeatures(degree=grado)
            X_transformed_train = model_poly.fit_transform(X_train)
            print(X_transformed_train)
            X_transformed_test = model_poly.fit_transform(X_test)
            model_ridge = Ridge(alpha=alpha)
            model_ridge.fit(X_transformed_train, y_train)
            #coef, intercept = model_ridge.coef_, model_ridge.intercept_
            #for coeficiente, variable in zip(coef, X_transformed_train):
            #    print(f"coef de {variable}: {coeficiente}")
        
            #print(f"intercept: {intercept}", flush=True)
            print('R-squared score ridge poly{:.0f} alpha{:.0f} (training): {:.9f}'.format(grado, alpha, model_ridge.score(X_transformed_train, y_train)))
            print('R-squared score ridge poly{:.0f} alpha{:.0f} (test): {:.9f}'.format(grado, alpha, model_ridge.score(X_transformed_test, y_test)))

            dict_estadisticos[f'R-squared train ridge{grado} alpha{alpha}'] = model_ridge.score(X_transformed_train, y_train)
            dict_estadisticos[f'R-squared test ridge{grado} alpha{alpha}']  = model_ridge.score(X_transformed_test,  y_test) 

    return dict_estadisticos

def fnc_polynomial_lasso_model(data: pd.DataFrame, X: list, y: str, test_size: float=0.3, grados: list=[1], alphas: list=[1]) -> dict:
    dict_estadisticos = dict()
    print('Usando sklearn Polynomial Lasso model predict')
    
    #Partimos el dataset en train y test para hacer Validación Cruzada
    X_train, X_test, y_train, y_test = train_test_split(data[X], data[y],test_size=test_size, random_state = 123)
    
    for grado in grados:
        for alpha in alphas:
            model_poly = PolynomialFeatures(degree=grado)
            X_transformed_train = model_poly.fit_transform(X_train)
            X_transformed_test = model_poly.fit_transform(X_test)
            model_lasso = Lasso(alpha=alpha)
            model_lasso.fit(X_transformed_train, y_train)
            #coef, intercept = model_lasso.coef_, model_lasso.intercept_
            #for coeficiente, variable in zip(coef, X):
            #    print(f"coef de {variable}: {coeficiente}")
        
            #print(f"intercept: {intercept}", flush=True)
            print('R-squared score lasso poly{:.0f} alpha{:.0f} (training): {:.9f}'.format(grado, alpha, model_lasso.score(X_transformed_train, y_train)))
            print('R-squared score lasso poly{:.0f} alpha{:.0f} (test): {:.9f}'.format(grado, alpha, model_lasso.score(X_transformed_test, y_test)))

            dict_estadisticos[f'R-squared train lasso poly{grado} alpha{alpha}'] = model_lasso.score(X_transformed_train, y_train)
            dict_estadisticos[f'R-squared test lasso poly{grado} alpha{alpha}']  = model_lasso.score(X_transformed_test,  y_test) 

    return dict_estadisticos

def fnc_knn_regression_model(data: pd.DataFrame, X: list, y: str, test_size: float=0.3, neighbors=[1]) -> dict:
    dict_estadisticos = dict()
    print('Usando sklearn KNN Regression model predict')
    
    #Partimos el dataset en train y test para hacer Validación Cruzada
    X_train, X_test, y_train, y_test = train_test_split(data[X], data[y],test_size=test_size, random_state = 123)
    
    for neighbors_ in neighbors:
        model_knn = KNeighborsRegressor(n_neighbors=neighbors_)
        model_knn.fit(X_train, y_train)
        #coef, intercept = model_knn.coef_, model_knn.intercept_
        #for coeficiente, variable in zip(coef, X):
        #    print(f"coef de {variable}: {coeficiente}")
    
        #print(f"intercept: {intercept}", flush=True)
        print('R-squared score KNN Regression neighbors{:.0f} (training): {:.9f}'.format(neighbors_, model_knn.score(X_train, y_train)))
        print('R-squared score KNN Regression neighbors{:.0f} (test): {:.9f}'.format(neighbors_, model_knn.score(X_test, y_test)))

        dict_estadisticos[f'R-squared train KNN Regression neighbors{neighbors_}'] = model_knn.score(X_train, y_train)
        dict_estadisticos[f'R-squared test KNN Regression neighbors{neighbors_}']  = model_knn.score(X_test,  y_test) 

    return dict_estadisticos

def fnc_decision_tree_regression_model(data: pd.DataFrame, X: list, y: str, test_size: float=0.3, depths:list=[5], min_samples_leafs: list=[5], min_samples_splits: list=[6]) -> dict:
    dict_estadisticos = dict()
    print('Usando sklearn KNN Regression model predict')
    
    #Partimos el dataset en train y test para hacer Validación Cruzada
    X_train, X_test, y_train, y_test = train_test_split(data[X], data[y],test_size=test_size, random_state = 123)
    
    for depth in depths:
        for min_samples_leaf in min_samples_leafs:
            for min_samples_split in min_samples_splits:
                model_dt = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split)
                model_dt.fit(X_train, y_train)
                #coef, intercept = model_dt.coef_, model_dt.intercept_
                #for coeficiente, variable in zip(coef, X):
                #    print(f"coef de {variable}: {coeficiente}")
            
                #print(f"intercept: {intercept}", flush=True)
                print('R-squared score DT Regression depth{:.0f} min_s_l{:.0f} min_s_s{:.0f} (training): {:.9f}'.format(depth, min_samples_leaf,min_samples_split, model_dt.score(X_train, y_train)))
                print('R-squared score DT Regression depth{:.0f} min_s_l{:.0f} min_s_s{:.0f} (test): {:.9f}'.format(depth, min_samples_leaf,min_samples_split, model_dt.score(X_test, y_test)))

                dict_estadisticos[f'R-squared train DT Regression depth{depth} min_s_l{min_samples_leaf} min_s_s{min_samples_split}'] = model_dt.score(X_train, y_train)
                dict_estadisticos[f'R-squared test DT Regression depth{depth} min_s_l{min_samples_leaf} min_s_s{min_samples_split}']  = model_dt.score(X_test,  y_test) 

    return dict_estadisticos

if __name__ == "__main__":
    ##Leemos la información que utilizaremos para la tarea 2
    
    ##Leemos la información de la tabla de ingresos
    ingresos = pd.read_csv('csv/41_enigh2022_ns_ingresos_csv/ingresos.csv')
    #Conversión de ingresos a su logaritmo natural
    ingresos.loc[:,'ln_ing_tri'] = ingresos.loc[:,'ing_tri'].apply(lambda x: np.log(x) if x!=0 else np.nan)
    woe_entidad = pd.DataFrame(ingresos.groupby(by='entidad')['ing_tri'].agg(['count','sum'])).reset_index()
    woe_entidad.loc[:,'%count'] = woe_entidad.loc[:,'count']/woe_entidad.loc[:,'count'].sum()
    woe_entidad.loc[:,'%sum'] = woe_entidad.loc[:,'sum']/woe_entidad.loc[:,'sum'].sum()
    woe_entidad.loc[:,'WoE_entidad'] = np.log(woe_entidad.loc[:,'%sum']/woe_entidad.loc[:,'%count'])
    woe_entidad.loc[:,'IV'] = woe_entidad.loc[:,'WoE_entidad'] * (woe_entidad.loc[:,'%sum']-woe_entidad.loc[:,'%count'])
    woe_entidad.sum()
    ingresos= ingresos.merge(woe_entidad.loc[:,['entidad','WoE_entidad']], how='left', on='entidad')
    #fnc_resumen_ingresos(ingresos)

    ##Leemos la información de la tabla de trabajos
    trabajos = pd.read_csv('csv/61_enigh2022_ns_trabajos_csv/trabajos.csv')
    #fnc_resumen_trabajos(trabajos)
    
    ##Leemos la información de la tabla de población
    poblacion = pd.read_csv('csv/57_enigh2022_ns_poblacion_csv/poblacion.csv')
    poblacion.loc[:, 'madre_id'] = poblacion.loc[:, 'madre_id'].apply(lambda x: int(x) if x not in [' ','&'] else np.nan)
    poblacion.loc[:, 'padre_id'] = poblacion.loc[:, 'padre_id'].apply(lambda x: int(x) if x not in [' ','&'] else np.nan)
    poblacion.loc[:, 'nivelaprob'] = poblacion.loc[:, 'nivelaprob'].apply(lambda x: int(x) if x not in [' ','&'] else np.nan)
    hijos = pd.concat([
        pd.DataFrame(poblacion.groupby(
            by=['folioviv','foliohog','padre_id'])['sexo'].count()
            ).reset_index().rename(columns={'sexo':'hijos','padre_id':'numren'}),
        pd.DataFrame(poblacion.groupby(
            by=['folioviv','foliohog','madre_id'])['sexo'].count()
            ).reset_index().rename(columns={'sexo':'hijos','madre_id':'numren'})],
        axis=0)
    poblacion = poblacion.merge(hijos, how='left', on=['folioviv','foliohog','numren'])
    poblacion.loc[:,'hijos'] = poblacion.loc[:,'hijos'].replace(np.nan,0)

    llaves = ['folioviv','foliohog','numren']
    woe_nivelaprob = pd.DataFrame(poblacion.merge(ingresos.loc[:,llaves+['ing_tri']], how='left',on=llaves).groupby(by='nivelaprob')['ing_tri'].agg(['count','sum'])).reset_index()
    woe_nivelaprob.loc[:,'%count'] = woe_nivelaprob.loc[:,'count']/woe_nivelaprob.loc[:,'count'].sum()
    woe_nivelaprob.loc[:,'%sum'] = woe_nivelaprob.loc[:,'sum']/woe_nivelaprob.loc[:,'sum'].sum()
    woe_nivelaprob.loc[:,'WoE_nivelaprob'] = np.log(woe_nivelaprob.loc[:,'%sum']/woe_nivelaprob.loc[:,'%count'])
    woe_nivelaprob.loc[:,'IV'] = woe_nivelaprob.loc[:,'WoE_nivelaprob'] * (woe_nivelaprob.loc[:,'%sum']-woe_nivelaprob.loc[:,'%count'])
    woe_nivelaprob.sum()
    poblacion= poblacion.merge(woe_nivelaprob.loc[:,['nivelaprob','WoE_nivelaprob']], how='left', on='nivelaprob')

    woe_edo_conyug = pd.DataFrame(poblacion.merge(ingresos.loc[:,llaves+['ing_tri']], how='left',on=llaves).groupby(by='edo_conyug')['ing_tri'].agg(['count','sum'])).reset_index()
    woe_edo_conyug.loc[:,'%count'] = woe_edo_conyug.loc[:,'count']/woe_edo_conyug.loc[:,'count'].sum()
    woe_edo_conyug.loc[:,'%sum'] = woe_edo_conyug.loc[:,'sum']/woe_edo_conyug.loc[:,'sum'].sum()
    woe_edo_conyug.loc[:,'WoE_edo_conyug'] = np.log(woe_edo_conyug.loc[:,'%sum']/woe_edo_conyug.loc[:,'%count'])
    woe_edo_conyug.loc[:,'IV'] = woe_edo_conyug.loc[:,'WoE_edo_conyug'] * (woe_edo_conyug.loc[:,'%sum']-woe_edo_conyug.loc[:,'%count'])
    woe_edo_conyug.sum()
    poblacion= poblacion.merge(woe_edo_conyug.loc[:,['edo_conyug','WoE_edo_conyug']], how='left', on='edo_conyug')

    poblacion = fnc_encode_nivelaprob(poblacion)
    #fnc_poblacion(poblacion)

    #Conjuntamos la información para realizar el análisis
    #Nuestra base constará de las personas que reportan un ingreso en el trimestre por salarios
    #tomando en cuenta de su trabajo principal las horas trabajadas en la semana
    #también su edad, género (1 si es masculino y 2 si es femenino), el número de hijos y grado de estudios

    llaves = ['folioviv','foliohog','numren']
    ingreso_por_salario = ingresos.clave=='P001'
    variables_ingresos = ['ing_tri','ln_ing_tri','WoE_entidad']
    registros = ingresos.loc[ingreso_por_salario,llaves+variables_ingresos]
    print(f'Número de personas con ingresos por salarios {len(registros):,.0f}')
    trabajo_principal = trabajos.id_trabajo==1
    variables_trabajos = ['htrab']
    registros = registros.merge(trabajos.loc[trabajo_principal,llaves+variables_trabajos],
                                how='inner',
                                on=llaves)
    print(f'Número de personas con trabajo principal {len(registros):,.0f}')
    variables_poblacion = ['edad','sexo','hijos','WoE_nivelaprob','WoE_edo_conyug']#,'nivel_0.0','nivel_1.0','nivel_2.0','nivel_3.0','nivel_4.0','nivel_5.0','nivel_6.0','nivel_7.0','nivel_8.0','nivel_9.0']
    registros = registros.merge(poblacion.loc[:,llaves+variables_poblacion],
                                how='left',
                                on=llaves)
    print(f'Número de personas total {len(registros):,.0f}')
    variables_para_modelo = registros.columns[5:]
    variable_independiente = 'ing_tri'

    registros.describe().T
    #Corremos varios modelos 
    scaler = StandardScaler()
    scaler.fit(registros.iloc[:,3:])
    registros_scaled = pd.concat([registros.iloc[:,:3], pd.DataFrame(scaler.transform(registros.iloc[:,3:]))], axis=1)
    registros_scaled.columns=registros.columns
    r_squared_lr = fnc_linear_regression_model(registros, variables_para_modelo, variable_independiente, test_size=0.4)
    r_squared_lr_stscale = fnc_linear_regression_model(registros_scaled, variables_para_modelo, variable_independiente, test_size=0.4)
    r_squared_poly = fnc_polynomial_model(registros, variables_para_modelo, variable_independiente, test_size=0.4, grados=[1,2,3,4,5])
    r_squared_poly_ridge = fnc_polynomial_ridge_model(registros, variables_para_modelo, variable_independiente, test_size=0.4, grados=[1,2,3,4,5], alphas=[10, 100, 1000, 10000])
    r_squared_poly_lasso = fnc_polynomial_lasso_model(registros, variables_para_modelo, variable_independiente, test_size=0.4, grados=[1,2,3,4,5], alphas=[10, 100, 1000, 10000])
    r_squared_knnregression = fnc_knn_regression_model(registros, variables_para_modelo, variable_independiente, test_size=0.4, neighbors=[i for i in range(1,15)]) 
    r_squared_decisiontreeregression = fnc_decision_tree_regression_model(registros, variables_para_modelo, variable_independiente, test_size=0.4,depths=[i for i in range(1,50,5)], min_samples_leafs=range(50,500,50), min_samples_splits=range(50,500,50))
    
    mejor_modelo = ''
    r2_mejor_modelo = 0
    for llave, valor in r_squared_lr.items():
        if llave.find('test')>0:
            if r2_mejor_modelo<valor:
                mejor_modelo=llave
                r2_mejor_modelo=valor
                print(llave+':'+str(valor))
    mejor_modelo = ''
    r2_mejor_modelo = 0
    for llave, valor in r_squared_lr_stscale.items():
        if llave.find('test')>0:
            if r2_mejor_modelo<valor:
                mejor_modelo=llave
                r2_mejor_modelo=valor
                print(llave+':'+str(valor))
    mejor_modelo = ''
    r2_mejor_modelo = 0
    for llave, valor in r_squared_poly.items():
        if llave.find('test')>0:
            if r2_mejor_modelo<valor:
                mejor_modelo=llave
                r2_mejor_modelo=valor
                print(llave+':'+str(valor))
    mejor_modelo = ''
    r2_mejor_modelo = 0
    for llave, valor in r_squared_poly_ridge.items():
        if llave.find('test')>0:
            if r2_mejor_modelo<valor:
                mejor_modelo=llave
                r2_mejor_modelo=valor
                print(llave+':'+str(valor))
    mejor_modelo = ''
    r2_mejor_modelo = 0
    for llave, valor in r_squared_poly_lasso.items():
        if llave.find('test')>0:
            if r2_mejor_modelo<valor:
                mejor_modelo=llave
                r2_mejor_modelo=valor
                print(llave+':'+str(valor))
    mejor_modelo = ''
    r2_mejor_modelo = 0
    for llave, valor in r_squared_knnregression.items():
        if llave.find('test')>0:
            if r2_mejor_modelo<valor:
                mejor_modelo=llave
                r2_mejor_modelo=valor
                print(llave+':'+str(valor))
    mejor_modelo = ''
    r2_mejor_modelo = 0
    for llave, valor in r_squared_decisiontreeregression.items():
        if llave.find('test')>0:
            if r2_mejor_modelo<valor:
                mejor_modelo=llave
                r2_mejor_modelo=valor
                print(llave+':'+str(valor))

    fnc_polynomial_ridge_model(registros, variables_para_modelo, variable_independiente, test_size=0.4, grados=[4], alphas=[10000])

    vif_data = pd.DataFrame() 
    vif_data["feature"] = registros.iloc[:,5:].columns 
    
    # calculating VIF for each feature 
    vif_data["VIF"] = [variance_inflation_factor(registros.iloc[:,5:].values, i) for i in range(len(registros.iloc[:,5:].columns))] 
    
    print(vif_data)
      