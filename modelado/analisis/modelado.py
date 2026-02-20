import pandas as pd
import numpy as np
import statsmodels.api as sm

print("Iniciando el modelado de regresión lineal...")

# 1. Generar datos simulados para el ejercicio
np.random.seed(42)
experiencia = np.random.rand(100) * 10          
salario = 1000 + 150 * experiencia + np.random.randn(100) * 200  

df = pd.DataFrame({'Experiencia': experiencia, 'Salario': salario})

# 2. Modelo de Regresión Lineal Simple (Minceriano básico)
X = sm.add_constant(df['Experiencia']) 
y = df['Salario']

modelo = sm.OLS(y, X).fit()

# 3. Mostrar resultados
print("¡Modelo entrenado con éxito!")
print("-" * 40)
print(modelo.summary().tables[1])