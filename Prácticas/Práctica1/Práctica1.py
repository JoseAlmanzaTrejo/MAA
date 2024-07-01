import pandas as pd
import seaborn as sns
import matplotlib
import math
from scipy import stats

import matplotlib.pyplot as plt

def fnc_analisis_inicial(data):

    #Obtenemos los estadísticos de sueldo neto total
    resumen1 = data.loc[:, 'Sueldo Neto'].describe()
    print("Resumen de Sueldo Neto")
    print(resumen1)

    #Generamos tabla de paso para el cálculo del número de empleados por Fecha/dependencia
    resumen2 = data.groupby(by=['Fecha','dependencia'])['Sueldo Neto'].count()
    resumen2 = resumen2.describe()
    print("Resumen de Número de empleados por Fecha/dependencia")
    print(resumen2)

    #Generamos tabla de paso para el cálculo del número de dependencias por Fecha/Tipo
    resumen3 = data.groupby(by=['Fecha','Tipo','dependencia'])['Sueldo Neto'].count().reset_index()
    #Ya que tenemos agrupado por las tres llaves, generamos el resumen que necesitamos
    resumen3 = resumen3.groupby(by=['Fecha','Tipo'])['dependencia'].count()
    resumen3 = resumen3.describe()
    print("Resumen de Número de empleados por Fecha/dependencia")
    print(resumen3)

def fnc_obten_estadisticos(data):

    return data.describe()

def fnc_compara_mean_std(data, media, desvest):
    return abs(data['Sueldo Neto']-media)/desvest

def fnc_grafico_inicial(data):
    resumen1 = pd.DataFrame(data.groupby(by='dependencia')['Sueldo Neto'].mean()).reset_index().sort_values(by='Sueldo Neto', ascending=False)
    print('Resumen de gráfico inicial: principaes dependencias con Sueldo Neto promedio más alto')
    estadisticos = fnc_obten_estadisticos(resumen1).T
    estadisticos_mean = estadisticos['mean']
    estadisticos_std = estadisticos['std']
    
    resumen1.loc[:,'veces std de media'] = resumen1.apply(fnc_compara_mean_std, media=estadisticos_mean, desvest=estadisticos_std, axis=1)
    print(resumen1.head(5))
    
    print('Se muestra gráfico de barras de Sueldo Neto por dependencia independientemente de la Fecha')
    resumen1.plot.bar(x='dependencia', 
                     y='Sueldo Neto', 
                     figsize=(25, 18),
                     title='Sueldo Neto promedio todas las dependencias')
    
    plt.savefig("Prácticas/Práctica1/img/lt_dependencias.png")
    plt.close()

    #Generamos resumen para % de empleados por Tipo de dependencia total
    resumen2 = pd.DataFrame(data.groupby(by=['Tipo'])['Sueldo Neto'].count()).rename(columns={'Sueldo Neto':'Empleados'})
    #Convertimos el dtype de Empleados ya que python nos muestra un warning por el cambio de tipo
    resumen2.loc[:,"%Empleados"] = resumen2.loc[:,'Empleados']/resumen2.loc[:,'Empleados'].sum()
    resumen2=resumen2.sort_values(by='%Empleados', ascending=False)
    print(resumen2)
    
    resumen2.plot(
        kind='pie',
        y='%Empleados',
        figsize=(12, 9),
        title='Porcentaje de población de empleados por Tipo de dependencia', 
        autopct='%1.1f%%')
    plt.savefig("Prácticas/Práctica1/img/pie_Empleados por Tipo.png")
    plt.close()

    #Generamos resumen para % de dependencias por Tipo de dependencia total
    resumen3 = pd.DataFrame(data.groupby(by=['Tipo','dependencia'])['Sueldo Neto'].count()).reset_index()
    resumen3 = pd.DataFrame(resumen3.groupby(by='Tipo')['dependencia'].count())
    resumen3.loc[:,'%dependencias'] = resumen3.loc[:,'dependencia'] / resumen3.loc[:,'dependencia'].sum()
    resumen3 = resumen3.sort_values(by='%dependencias', ascending=False)
    print(resumen3)
    
    resumen3.plot(
        kind='pie',
        y='%dependencias',
        figsize=(12, 9),
        title='Porcentaje de dependencias por Tipo de dependencia', 
        autopct='%1.1f%%')
    plt.savefig("Prácticas/Práctica1/img/pie_dependencias por Tipo.png")
    plt.close()

    #También podemos generar un gráfico de distribución total por Tipo de dependencia del Sueldo Neto de los Empleados
    sns.set(style="whitegrid")
    plt.figure(figsize=(20,13))
    sns.violinplot(x="Tipo", y="Sueldo Neto", data=data, hue="Tipo")
    plt.savefig("Prácticas/Práctica1/img/violín_Sueldo Neto por Tipo.png")

def fnc_asigna_corte(data, cortes, variable):
    for corte in cortes:
        if data.loc[variable] <= corte:
            return corte

def fnc_grafico_dependencia_mensual_distribucion(data, dependencia, variable, Fecha):
    try:
        data_copy = data.copy()
        sueldo_minimo = data_copy.loc[:,variable].min()
        sueldo_maximo = data_copy.loc[:,variable].max()
        rango = sueldo_maximo - sueldo_minimo
        n_particiones = math.ceil(math.sqrt(len(data_copy)))
        rango_particion = rango/n_particiones

        cortes = [sueldo_minimo + (i + 1) * rango_particion + 0.01 for i in range(n_particiones)]

        data_copy.loc[:,'corte'] = data_copy.apply(fnc_asigna_corte, cortes=cortes, variable=variable, axis=1)

        resumen = pd.DataFrame(data_copy.groupby(by='corte')[variable].count()).reset_index().sort_values(by='corte')
        resumen.rename(columns={variable:'registros'}, inplace=True)
        resumen.plot.bar(x='corte', 
                         y='registros', 
                         figsize=(25, 15), 
                         title=f'Histograma de {variable} {dependencia} de la fecha {Fecha}')
        plt.savefig(f"Prácticas/Práctica1/img/{dependencia}_{Fecha}_distribución_{variable}.png")
        plt.close()
    except:
        print(f'No se puede imprimir la imagen {dependencia}_{Fecha}_distribución.png')

def fnc_grafico_dependencia_mensual(data):
    for dependencia in list(data.dependencia.unique()):
        for Fecha in list(data.Fecha.unique()):
            filtro_dependencia = (data.dependencia == dependencia)
            filtro_fecha = (data.Fecha == Fecha)
            fnc_grafico_dependencia_mensual_distribucion(data.loc[filtro_dependencia&filtro_fecha,:], dependencia=dependencia, variable='Sueldo Neto', Fecha=Fecha)

def fnc_grafico_tipo_dependencia_mensual(data):
    for Tipo in list(data.Tipo.unique()):
        for Fecha in list(data.Fecha.unique()):
            filtro_dependencia = (data.Tipo == Tipo)
            filtro_fecha = (data.Fecha == Fecha)
            fnc_grafico_dependencia_mensual_distribucion(data.loc[filtro_dependencia&filtro_fecha,:], dependencia=Tipo, variable='Sueldo Neto', Fecha=Fecha)

def fnc_grafico_mensual(data):
    for Fecha in list(data.Fecha.unique()):
        filtro_fecha = (data.Fecha == Fecha)
        fnc_grafico_dependencia_mensual_distribucion(data.loc[filtro_fecha,:], dependencia='Todos', variable='Sueldo Neto', Fecha=Fecha)

def fnc_grafico_mensual_poblacion(data):
    for Fecha in list(data.Fecha.unique()):
        filtro_fecha = (data.Fecha == Fecha)
        data_copy = pd.DataFrame(data.loc[filtro_fecha,:].groupby(by='dependencia')['Sueldo Neto'].count()).reset_index().rename(columns={'Sueldo Neto':'poblacion'})
        fnc_grafico_dependencia_mensual_distribucion(data_copy, dependencia='Todos', variable='poblacion', Fecha=Fecha)

def fnc_grafico_pie(data, Fecha):

    resumen = pd.DataFrame(data.groupby(by=['Tipo','dependencia'])['Sueldo Neto'].count()).reset_index()
    resumen = pd.DataFrame(resumen.groupby(by='Tipo')['dependencia'].count())
    resumen.loc[:,'%dependencias'] = resumen.loc[:,'dependencia'] / resumen.loc[:,'dependencia'].sum()
    resumen = resumen.sort_values(by='%dependencias', ascending=False)
    
    resumen.plot(
        kind='pie',
        y='%dependencias',
        figsize=(12, 9),
        title=f'Porcentaje de dependencias por Tipo de dependencia del mes{Fecha}', 
        autopct='%1.1f%%')
    plt.savefig(f"Prácticas/Práctica1/img/pie_dependencias por Tipo del mes {Fecha}.png")
    plt.close()

def fnc_grafico_mensual_poblacion_tipo(data):
    for Fecha in list(data.Fecha.unique()):
        #Generamos gráfico de Pie de distribución de Número de empleados por Tipo de dependencia por mes
        filtro_fecha = (data.Fecha == Fecha)

        fnc_grafico_pie(data.loc[filtro_fecha,:], Fecha=Fecha)
        for Tipo in list(data.Tipo.unique()):
            filtro_tipo = (data.Tipo == Tipo)
            data_copy = pd.DataFrame(data.loc[filtro_fecha&filtro_tipo,:].groupby(by='dependencia')['Sueldo Neto'].count()).reset_index().rename(columns={'Sueldo Neto':'poblacion'})
            fnc_grafico_dependencia_mensual_distribucion(data_copy, dependencia=Tipo, variable='poblacion', Fecha=Fecha)

def fnc_grafico_mensual_tipo(data):
    for Tipo in list(data.Tipo.unique()):
        for Fecha in list(data.Fecha.unique()):
            filtro_tipo = (data.Tipo == Tipo)
            filtro_fecha = (data.Fecha == Fecha)
            data_copy = data.loc[filtro_tipo&filtro_fecha,:]
            fnc_grafico_dependencia_mensual_distribucion(data_copy, dependencia=Tipo, variable='Sueldo Neto', Fecha=Fecha)

def fnc_prueba_kruskal_wallis_tipo(data):

    #La prueba no paramétrica de Kruskal Wallis se utiliza para contrastar
    #H0: Las medianas de las poblaciones son iguales
    #vs
    #H1: Las medianas de las poblaciones no son iguales

    Tipos = data.Tipo.unique()
    print(Tipos)
    for n1, Tipo1 in enumerate(Tipos):
        for n2, Tipo2 in enumerate(Tipos):
            if n2>=n1:
                continue
            else:
                data1 = data.loc[data.Tipo==Tipo1,'Sueldo Neto']
                data2 = data.loc[data.Tipo==Tipo2,'Sueldo Neto']
                
                print(f'Prueba Kruskal Wallis de {Tipo1} vs {Tipo2}: {stats.kruskal(data1, data2)}')

if __name__ == "__main__":
    #Leemos la información de los pagos individuales
    df_pagos_uanl = pd.read_csv("csv/typed_uanl.csv")
    print(df_pagos_uanl.info())

    #Las entidades que podemos identificar son: 
    # sueldo neto, número de empleados, número de dependencias
    #fnc_analisis_inicial(df_pagos_uanl)
    #fnc_grafico_inicial(df_pagos_uanl)
    
    #Generamos los gráficos de distribución de Sueldos Netos totales por mes
    #fnc_grafico_mensual(df_pagos_uanl)

    #Generamos los gráficos de distribución de Sueldos Netos por dependencia por mes
    #fnc_grafico_dependencia_mensual(df_pagos_uanl)

    #Generamos los gráficos de distribución de Sueldos Netos por Tipo de dependencia por mes
    #fnc_grafico_tipo_dependencia_mensual(df_pagos_uanl)

    #Generamos los gráficos de distribución de número de empleados totales por mes
    #fnc_grafico_mensual_poblacion(df_pagos_uanl)

    #Generamos los gráficos de distribución de número de empleados por dependencia por Tipo de dependencia por mes
    #fnc_grafico_mensual_poblacion_tipo(df_pagos_uanl)

    #Generamos los gráficos de distribución de Sueldo Neto por Tipo de dependencia por mes
    #fnc_grafico_mensual_tipo(df_pagos_uanl)

    #Generamos una prueba de Kruskal Wallis a cada par de Tipos de dependencia
    fnc_prueba_kruskal_wallis_tipo(df_pagos_uanl)