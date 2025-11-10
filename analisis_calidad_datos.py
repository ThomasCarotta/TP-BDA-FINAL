import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def analizar_calidad_datos():
    """
    Analiza la calidad de datos del archivo CSV de exportaciones petroleras.
    Detecta valores faltantes y outliers.
    """
    
    # Cargar el archivo CSV
    archivo = 'c:/Users/thoma/OneDrive/Escritorio/Trae/TP-FINAL-BDA/data/exportaciones-sector-actividad-saldocomercial-rengo-empresa-exportadora-mensual.csv'
    
    print("=== ANÁLISIS DE CALIDAD DE DATOS ===")
    print("=" * 50)
    
    try:
        # Leer el archivo CSV
        df = pd.read_csv(archivo)
        print(f"Archivo cargado exitosamente: {archivo}")
        print(f"Dimensiones del dataset: {df.shape[0]} filas x {df.shape[1]} columnas")
        print("\n")
        
        # 1. ANÁLISIS DE VALORES FALTANTES
        print("1. ANÁLISIS DE VALORES FALTANTES")
        print("-" * 40)
        
        # Calcular valores faltantes por columna
        valores_faltantes = df.isnull().sum()
        porcentaje_faltantes = (df.isnull().sum() / len(df)) * 100
        
        # Crear resumen de valores faltantes
        resumen_faltantes = pd.DataFrame({
            'Columna': df.columns,
            'Valores_Faltantes': valores_faltantes,
            'Porcentaje_Faltantes': porcentaje_faltantes
        })
        
        # Filtrar solo columnas con valores faltantes
        columnas_con_faltantes = resumen_faltantes[resumen_faltantes['Valores_Faltantes'] > 0]
        
        if len(columnas_con_faltantes) > 0:
            print("Columnas con valores faltantes:")
            print(columnas_con_faltantes.sort_values('Porcentaje_Faltantes', ascending=False))
        else:
            print("No se encontraron valores faltantes en el dataset.")
        
        print(f"\nTotal de valores faltantes en el dataset: {df.isnull().sum().sum()}")
        print(f"Porcentaje total de datos faltantes: {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.2f}%")
        print("\n")
        
        # 2. ANÁLISIS DE OUTLIERS
        print("2. ANÁLISIS DE OUTLIERS")
        print("-" * 40)
        
        # Identificar columnas numéricas
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        # Excluir la columna de fecha si existe
        columnas_numericas = [col for col in columnas_numericas if 'fecha' not in col.lower() and 'indice' not in col.lower()]
        
        print(f"Se analizarán {len(columnas_numericas)} columnas numéricas para detectar outliers.")
        
        # Detectar outliers usando el método IQR
        outliers_por_columna = {}
        
        for columna in columnas_numericas[:10]:  # Limitar a las primeras 10 columnas para el informe
            if df[columna].dtype in ['float64', 'int64']:
                # Calcular IQR
                Q1 = df[columna].quantile(0.25)
                Q3 = df[columna].quantile(0.75)
                IQR = Q3 - Q1
                
                # Definir límites para outliers
                limite_inferior = Q1 - 1.5 * IQR
                limite_superior = Q3 + 1.5 * IQR
                
                # Identificar outliers
                outliers = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]
                outliers_por_columna[columna] = len(outliers)
                
                if len(outliers) > 0:
                    print(f"\n{columna}:")
                    print(f"  - Outliers detectados: {len(outliers)} ({(len(outliers)/len(df))*100:.2f}%)")
                    print(f"  - Rango normal: [{limite_inferior:.2f}, {limite_superior:.2f}]")
                    print(f"  - Valor mínimo: {df[columna].min():.2f}")
                    print(f"  - Valor máximo: {df[columna].max():.2f}")
                    print(f"  - Media: {df[columna].mean():.2f}")
                    print(f"  - Desviación estándar: {df[columna].std():.2f}")
        
        # 3. ESTADÍSTICAS GENERALES
        print("\n3. ESTADÍSTICAS GENERALES DEL DATASET")
        print("-" * 40)
        
        # Estadísticas descriptivas
        print("\nEstadísticas descriptivas (primeras 5 columnas numéricas):")
        print(df[columnas_numericas[:5]].describe())
        
        # 4. VISUALIZACIONES
        print("\n4. GENERANDO VISUALIZACIONES...")
        print("-" * 40)
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análisis de Calidad de Datos - Dataset de Exportaciones Petroleras', fontsize=16)
        
        # Gráfico 1: Heatmap de valores faltantes
        if df.isnull().sum().sum() > 0:
            sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis', ax=axes[0,0])
            axes[0,0].set_title('Mapa de Valores Faltantes')
            axes[0,0].set_xlabel('Columnas')
        else:
            axes[0,0].text(0.5, 0.5, 'No hay valores faltantes', ha='center', va='center', transform=axes[0,0].transAxes)
            axes[0,0].set_title('Mapa de Valores Faltantes')
        
        # Gráfico 2: Distribución de outliers en primera columna numérica
        if len(columnas_numericas) > 0:
            primera_col = columnas_numericas[0]
            df[primera_col].hist(bins=30, ax=axes[0,1], alpha=0.7, color='skyblue', edgecolor='black')
            axes[0,1].set_title(f'Distribución de {primera_col}')
            axes[0,1].set_xlabel('Valor')
            axes[0,1].set_ylabel('Frecuencia')
        
        # Gráfico 3: Boxplot para detectar outliers (segunda columna numérica)
        if len(columnas_numericas) > 1:
            segunda_col = columnas_numericas[1]
            df.boxplot(column=segunda_col, ax=axes[1,0])
            axes[1,0].set_title(f'Boxplot de {segunda_col}')
            axes[1,0].set_ylabel('Valor')
        
        # Gráfico 4: Resumen de calidad por columna
        calidad_resumen = []
        for col in df.columns:
            faltantes = df[col].isnull().sum()
            if col in columnas_numericas:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                limite_inferior = Q1 - 1.5 * IQR
                limite_superior = Q3 + 1.5 * IQR
                outliers = len(df[(df[col] < limite_inferior) | (df[col] > limite_superior)])
            else:
                outliers = 0
            calidad_resumen.append({
                'Columna': col[:20] + '...' if len(col) > 20 else col,
                'Calidad_%': ((len(df) - faltantes - outliers) / len(df)) * 100
            })
        
        calidad_df = pd.DataFrame(calidad_resumen)
        calidad_df.set_index('Columna')['Calidad_%'].plot(kind='bar', ax=axes[1,1], color='lightgreen')
        axes[1,1].set_title('Calidad de Datos por Columna (%)')
        axes[1,1].set_ylabel('Porcentaje de Calidad')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('analisis_calidad_datos.png', dpi=300, bbox_inches='tight')
        print("Visualización guardada como 'analisis_calidad_datos.png'")
        
        # 5. RECOMENDACIONES
        print("\n5. RECOMENDACIONES")
        print("-" * 40)
        
        recomendaciones = []
        
        if df.isnull().sum().sum() > 0:
            recomendaciones.append("• Considerar imputar valores faltantes usando técnicas como media, mediana o interpolación")
            recomendaciones.append("• Evaluar la posibilidad de eliminar filas/columnas con alto porcentaje de valores faltantes")
        
        total_outliers = sum(outliers_por_columna.values())
        if total_outliers > 0:
            recomendaciones.append("• Revisar y validar los outliers detectados antes de eliminarlos")
            recomendaciones.append("• Considerar transformaciones de datos (log, Box-Cox) para reducir la variabilidad")
        
        recomendaciones.append("• Implementar validación de datos en la fuente para prevenir valores faltantes")
        recomendaciones.append("• Establecer límites razonables para detectar valores atípicos en nuevos datos")
        
        for rec in recomendaciones:
            print(rec)
        
        print("\n" + "=" * 50)
        print("ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("=" * 50)
        
        return df
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {archivo}")
        return None
    except Exception as e:
        print(f"Error al procesar el archivo: {str(e)}")
        return None

if __name__ == "__main__":
    # Ejecutar el análisis
    df = analizar_calidad_datos()
    
    if df is not None:
        print("\nResumen ejecutivo:")
        print(f"- Total de filas: {len(df)}")
        print(f"- Total de columnas: {len(df.columns)}")
        print(f"- Valores faltantes totales: {df.isnull().sum().sum()}")
        
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        columnas_numericas = [col for col in columnas_numericas if 'fecha' not in col.lower() and 'indice' not in col.lower()]
        
        total_outliers = 0
        for columna in columnas_numericas:
            Q1 = df[columna].quantile(0.25)
            Q3 = df[columna].quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            outliers = len(df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)])
            total_outliers += outliers
        
        print(f"- Outliers detectados (método IQR): {total_outliers}")
        print("- Se ha generado un gráfico de análisis guardado como 'analisis_calidad_datos.png'")