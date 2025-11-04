import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 70)
print("ANÁLISIS DE REGRESIÓN: PRODUCCIÓN vs VENTAS DE COMBUSTIBLES")
print("=" * 70)

# Cargar datos transformados
df = pd.read_csv('dataset_petroleo_gas_transformado.csv')

# 1. PREPARAR DATOS PARA REGRESIÓN
print("\n1.  PREPARACIÓN DE DATOS")

# Identificar automáticamente las columnas relevantes
empresas_petroleo = [col for col in df.columns if 'produccion_petroleo_crudo' in col and 'total' not in col]
empresas_gas = [col for col in df.columns if 'produccion_gas_natural' in col and 'total' not in col]
empresas_gasoil = [col for col in df.columns if 'ventas_gasoil' in col and 'total' not in col]
empresas_naftas = [col for col in df.columns if 'ventas_naftas' in col and 'total' not in col]

print(f" Empresas de petróleo identificadas: {len(empresas_petroleo)}")
print(f" Empresas de gas identificadas: {len(empresas_gas)}")
print(f" Empresas de gasoil identificadas: {len(empresas_gasoil)}")
print(f" Empresas de naftas identificadas: {len(empresas_naftas)}")

# Calcular totales
df['produccion_petroleo_total'] = df[empresas_petroleo].sum(axis=1)
df['produccion_gas_total'] = df[empresas_gas].sum(axis=1)
df['ventas_gasoil_total'] = df[empresas_gasoil].sum(axis=1)
df['ventas_naftas_total'] = df[empresas_naftas].sum(axis=1)

# Agregar fecha para análisis temporal
df['fecha'] = pd.to_datetime(df['indice_tiempo'])
df['año'] = df['fecha'].dt.year
df['mes'] = df['fecha'].dt.month

# 2. ESTADÍSTICAS DESCRIPTIVAS
print("\n2.ESTADÍSTICAS DESCRIPTIVAS")

variables_analisis = ['produccion_petroleo_total', 'produccion_gas_total', 
                     'ventas_gasoil_total', 'ventas_naftas_total']

stats_df = df[variables_analisis].describe()
print(stats_df.round(2))

# 3. MATRIZ DE CORRELACIONES
print("\n3.  MATRIZ DE CORRELACIONES")

correlation_matrix = df[variables_analisis].corr()
print("Matriz de correlaciones:")
print(correlation_matrix.round(3))

# Visualizar matriz de correlaciones
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, fmt='.3f')
plt.title('Matriz de Correlaciones - Producción vs Ventas', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('matriz_correlaciones.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. MODELOS DE REGRESIÓN LINEAL
print("\n4.  MODELOS DE REGRESIÓN LINEAL")

# Configurar subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
model_results = []

# Función para entrenar y evaluar modelo
def entrenar_modelo_regresion(X, y, ax, titulo, color_datos='blue', color_linea='red'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Métricas
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Gráfico
    ax.scatter(X_test, y_test, alpha=0.6, color=color_datos, label='Datos reales', s=50)
    
    # Ordenar para línea continua
    sort_idx = np.argsort(X_test.values.flatten())
    X_sorted = X_test.values[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    
    ax.plot(X_sorted, y_pred_sorted, color=color_linea, linewidth=3, label='Línea de regresión')
    ax.set_xlabel(X.columns[0].replace('_', ' ').title(), fontsize=11)
    ax.set_ylabel(y.name.replace('_', ' ').title(), fontsize=11)
    ax.set_title(f'{titulo}\nR² = {r2:.3f}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Agregar ecuación de regresión
    coef = model.coef_[0]
    intercept = model.intercept_
    equation = f'y = {coef:.2f}x + {intercept:.2f}'
    ax.text(0.05, 0.95, equation, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return {
        'modelo': titulo,
        'r2': r2,
        'mse': mse,
        'mae': mae,
        'coef': coef,
        'intercept': intercept,
        'model': model
    }

# Modelo 1: Petróleo vs Gasoil
result1 = entrenar_modelo_regresion(
    df[['produccion_petroleo_total']], 
    df['ventas_gasoil_total'],
    axes[0,0], 
    'Petróleo vs Gasoil',
    'blue', 'red'
)
model_results.append(result1)

# Modelo 2: Petróleo vs Naftas
result2 = entrenar_modelo_regresion(
    df[['produccion_petroleo_total']], 
    df['ventas_naftas_total'],
    axes[0,1], 
    'Petróleo vs Naftas', 
    'green', 'orange'
)
model_results.append(result2)

# Modelo 3: Gas vs Gasoil
result3 = entrenar_modelo_regresion(
    df[['produccion_gas_total']], 
    df['ventas_gasoil_total'],
    axes[1,0], 
    'Gas vs Gasoil',
    'purple', 'brown'
)
model_results.append(result3)

# Modelo 4: Gas vs Naftas
result4 = entrenar_modelo_regresion(
    df[['produccion_gas_total']], 
    df['ventas_naftas_total'],
    axes[1,1], 
    'Gas vs Naftas',
    'brown', 'pink'
)
model_results.append(result4)

plt.tight_layout()
plt.savefig('regresiones_produccion_ventas.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. RESULTADOS DETALLADOS
print("\n5.  RESULTADOS DETALLADOS DE MODELOS")
results_df = pd.DataFrame(model_results)
print(results_df[['modelo', 'r2', 'mse', 'mae', 'coef']].round(4))

# 6. ANÁLISIS DE CORRELACIONES ESTADÍSTICAS
print("\n6.  ANÁLISIS ESTADÍSTICO DE CORRELACIONES")

print("Correlaciones de Pearson (significancia estadística):")
print("-" * 70)

correlaciones_significativas = []

for i, var1 in enumerate(variables_analisis):
    for j, var2 in enumerate(variables_analisis):
        if i < j:  # Evitar duplicados
            corr, p_value = stats.pearsonr(df[var1], df[var2])
            es_significativo = p_value < 0.05
            fuerza = ""
            
            if abs(corr) >= 0.7:
                fuerza = "FUERTE"
            elif abs(corr) >= 0.5:
                fuerza = "MODERADA"
            elif abs(corr) >= 0.3:
                fuerza = "DÉBIL"
            else:
                fuerza = "MUY DÉBIL"
            
            print(f" {var1.replace('_', ' ').title()} vs {var2.replace('_', ' ').title()}")
            print(f"   Correlación: {corr:.3f} ({fuerza})")
            print(f"   p-value: {p_value:.6f}")
            print(f"   Significativo: {' SÍ' if es_significativo else '❌ NO'}")
            
            if es_significativo and abs(corr) >= 0.3:
                correlaciones_significativas.append({
                    'variables': f"{var1} vs {var2}",
                    'correlacion': corr,
                    'fuerza': fuerza
                })
            print()

# 7. EVOLUCIÓN TEMPORAL
print("\n7.  EVOLUCIÓN TEMPORAL DE PRODUCCIÓN Y VENTAS")

# Agrupar por año
df_anual = df.groupby('año').agg({
    'produccion_petroleo_total': 'mean',
    'produccion_gas_total': 'mean',
    'ventas_gasoil_total': 'mean',
    'ventas_naftas_total': 'mean'
}).reset_index()

plt.figure(figsize=(14, 10))

# Producción
plt.subplot(2, 1, 1)
plt.plot(df_anual['año'], df_anual['produccion_petroleo_total'], 
         marker='o', linewidth=2, label='Petróleo', color='blue')
plt.plot(df_anual['año'], df_anual['produccion_gas_total'], 
         marker='s', linewidth=2, label='Gas', color='green')
plt.title('Evolución Anual de la Producción', fontsize=14, fontweight='bold')
plt.xlabel('Año')
plt.ylabel('Producción Total')
plt.legend()
plt.grid(True, alpha=0.3)

# Ventas
plt.subplot(2, 1, 2)
plt.plot(df_anual['año'], df_anual['ventas_gasoil_total'], 
         marker='o', linewidth=2, label='Gasoil', color='red')
plt.plot(df_anual['año'], df_anual['ventas_naftas_total'], 
         marker='s', linewidth=2, label='Naftas', color='orange')
plt.title('Evolución Anual de las Ventas', fontsize=14, fontweight='bold')
plt.xlabel('Año')
plt.ylabel('Ventas Total')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('evolucion_temporal.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. CONCLUSIONES Y RECOMENDACIONES
print("\n8.  CONCLUSIONES Y RECOMENDACIONES")
print("=" * 70)

# Mejor modelo
mejor_modelo = results_df.loc[results_df['r2'].idxmax()]
print(f" MEJOR MODELO: {mejor_modelo['modelo']}")
print(f"   R²: {mejor_modelo['r2']:.3f}")
print(f"   Ecuación: y = {mejor_modelo['coef']:.4f}x + {mejor_modelo['intercept']:.2f}")

print(f"\n📊 INTERPRETACIÓN DE R²:")
print("   • R² > 0.7:  Fuerte relación predictiva")
print("   • R² 0.5-0.7:  Relación moderada") 
print("   • R² 0.3-0.5:  Relación débil")
print("   • R² < 0.3:  Relación muy débil o inexistente")

print(f"\n🔍 CORRELACIONES SIGNIFICATIVAS ENCONTRADAS:")
if correlaciones_significativas:
    for corr in correlaciones_significativas:
        print(f"   • {corr['variables']}: {corr['correlacion']:.3f} ({corr['fuerza']})")
else:
    print("    No se encontraron correlaciones significativas fuertes")

print(f"\n RECOMENDACIONES:")
if mejor_modelo['r2'] > 0.5:
    print("    La producción local SÍ parece impactar en las ventas internas")
    print("    Se recomienda profundizar el análisis con series temporales")
else:
    print("    La relación producción-ventas es más débil de lo esperado")
    print("    Considerar otros factores: precios, importaciones, demanda estacional")

print(f"\n GRÁFICOS GENERADOS:")
print("    matriz_correlaciones.png")
print("    regresiones_produccion_ventas.png") 
print("    evolucion_temporal.png")

print(f"\n ANÁLISIS COMPLETADO EXITOSAMENTE!")
print("=" * 70)

# Guardar resultados en CSV
results_df.to_csv('resultados_regresion.csv', index=False)
print("Resultados guardados en: resultados_regresion.csv")