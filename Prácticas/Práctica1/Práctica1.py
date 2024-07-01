import pandas as pd
import matplotlib
import math

import matplotlib.pyplot as plt

def fnc_analisis_inicial(data):

    #Obtenemos los estadísticos de sueldo neto total
    resumen1 = data.loc[:, 'Sueldo Neto'].describe()
    print("Resumen de Sueldo Neto")
    print(resumen1)

    #Generamos tabla de paso para el cálculo del número de empleados por Fecha/dependencia
    resumen2 = data.groupby(by=['Fecha','dependencia'])['Sueldo Neto'].count()
    print("Resumen de Número de empleados por Fecha/dependencia")
    print(resumen2)


def fnc_grafico_inicial(data):
    resumen = pd.DataFrame(data.groupby(by='dependencia')['Sueldo Neto'].mean()).reset_index().sort_values(by='Sueldo Neto', ascending=False)
    resumen.plot.bar(x='dependencia', 
                     y='Sueldo Neto', 
                     figsize=(10, 8),
                     title='Sueldo Neto promedio todas las dependeincias',
                     fontsize=3)
    plt.savefig(f"Prácticas/Práctica1/img/lt_dependencias.png")
    plt.close()
    print("Se crean gráficos de Sueldo Neto promedio total, y conteo promedio de empleados por dependencia total por mes")

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
        resumen.plot.bar(x='corte', y='registros', figsize=(32, 18))
        plt.savefig(f"img/{dependencia}_{Fecha}_distribución_{variable}.png")
        plt.close()
    except:
        print(f'No se puede imprimir la imagen {dependencia}_{Fecha}_distribución.png')

def fnc_grafico_dependencia_mensual(data):
    for dependencia in list(data.dependencia.unique()):
        for Fecha in list(data.Fecha.unique()):
            filtro_dependencia = (data.dependencia == dependencia)
            filtro_fecha = (data.Fecha == Fecha)
            fnc_grafico_dependencia_mensual_distribucion(data.loc[filtro_dependencia&filtro_fecha,:], dependencia=dependencia, variable='Sueldo Neto', Fecha=Fecha)

def fnc_grafico_mensual(data):
    for Fecha in list(data.Fecha.unique()):
        filtro_fecha = (data.Fecha == Fecha)
        fnc_grafico_dependencia_mensual_distribucion(data.loc[filtro_fecha,:], dependencia='Todos', variable='Sueldo Neto', Fecha=Fecha)

def fnc_grafico_mensual_poblacion(data):
    for Fecha in list(data.Fecha.unique()):
        filtro_fecha = (data.Fecha == Fecha)
        data_copy = pd.DataFrame(data.loc[filtro_fecha,:].groupby(by='dependencia')['Sueldo Neto'].count()).reset_index().rename(columns={'Sueldo Neto':'poblacion'})
        #print(data_copy)
        fnc_grafico_dependencia_mensual_distribucion(data_copy, dependencia='Todos', variable='poblacion', Fecha=Fecha)

def fnc_grafico_mensual_tipo(data):
    for Tipo in list(data.Tipo.unique()):
        for Fecha in list(data.Fecha.unique()):
            filtro_tipo = (data.Tipo == Tipo)
            filtro_fecha = (data.Fecha == Fecha)
            fnc_grafico_dependencia_mensual_distribucion(data.loc[filtro_tipo&filtro_fecha,:], dependencia=Tipo, variable='Sueldo Neto', Fecha=Fecha)

if __name__ == "__main__":
    #Leemos la información de los pagos individuales
    df_pagos_uanl = pd.read_csv("csv/typed_uanl.csv")
    print(df_pagos_uanl.info())

    #Las entidades que podemos identificar son: 
    # sueldo neto, número de empleados, número de dependencias
    fnc_analisis_inicial(df_pagos_uanl)
    #fnc_grafico_inicial(df_pagos_uanl)
    #fnc_grafico_mensual(df_pagos_uanl)
    #fnc_grafico_dependencia_mensual(df_pagos_uanl)
    #fnc_grafico_mensual_poblacion(df_pagos_uanl)
    #fnc_grafico_mensual_tipo(df_pagos_uanl)