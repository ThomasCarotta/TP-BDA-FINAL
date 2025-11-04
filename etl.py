import pandas as pd
import numpy as np

# Cargar el dataset
df = pd.read_csv('data\exportaciones-sector-actividad-saldocomercial-rengo-empresa-exportadora-mensual.csv')

# 1. EXTRACT - Ya hemos cargado los datos

# 2. TRANSFORM
print("Información inicial del dataset:")
print(f"Dimensiones: {df.shape}")
print(f"Columnas: {len(df.columns)}")
print(f"Rango temporal: {df['indice_tiempo'].min()} a {df['indice_tiempo'].max()}")

# Agregar columna de IDs
df.insert(0, 'id', range(1, len(df) + 1))

# Redondear todos los valores numéricos al 2do decimal
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].round(2)

# Reemplazar valores NaN con 0.0
df = df.fillna(0.0)

# Verificar transformaciones
print(f"\nDespués de las transformaciones:")
print(f"Valores nulos: {df.isnull().sum().sum()}")
print(f"Tipos de datos:\n{df.dtypes.value_counts()}")

# 3. LOAD - Guardar el dataset transformado
output_filename = 'dataset_petroleo_gas_transformado.csv'
df.to_csv(output_filename, index=False)

# Mostrar información del resultado
print(f"\nDataset transformado guardado como: {output_filename}")
print(f"Primeras filas del dataset transformado:")
print(df.head())

print(f"\nEstructura final:")
print(f"Total de registros: {len(df)}")
print(f"Total de columnas: {len(df.columns)}")
print(f"Columna ID agregada: Sí")
print(f"Valores redondeados a 2 decimales: Sí")
print(f"Valores faltantes reemplazados con 0.0: Sí")