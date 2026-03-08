import pandas as pd
import numpy as np
import statsmodels.api as sm
import requests
import io

print("Iniciando el proceso...")

### Generación de datos simulados
np.random.seed(42)
experiencia_sim = np.random.rand(100) * 10          
salario_sim = 1000 + 150 * experiencia_sim + np.random.randn(100) * 200  

### Guardamos los datos simulados en un DataFrame
df_simulado = pd.DataFrame({'Experiencia': experiencia_sim, 'Salario': salario_sim})

# -------------------------------------------------------------------
# 2. NUEVO PASO: Importar los datos limpios desde Zenodo vía API
# -------------------------------------------------------------------
print("Descargando 'datos_limpios.csv' desde Zenodo vía API...")

# url de Zenodo con los datos simulados
url_zenodo = "https://zenodo.org/api/records/18906308/files/datos_limpios.csv/content"

# Petición HTTP a los servidores de Zenodo
respuesta = requests.get(url_zenodo)

# Sobreescribimos la variable de datos leyendo el texto plano devuelto por la API
df = pd.read_csv(io.StringIO(respuesta.text))
# -------------------------------------------------------------------

# 3. Modelo de Regresión Lineal Simple (Minceriano básico)
# Usamos las columnas exactas de tu archivo CSV (en minúsculas)
print("Entrenando el modelo econométrico...")
X = sm.add_constant(df['experiencia']) 
y = df['salario']

modelo = sm.OLS(y, X).fit()

# 4. Mostrar resultados
print("Modelo entrenado con éxito usando los datos de la nube")
print("-" * 40)
print(modelo.summary().tables[1])