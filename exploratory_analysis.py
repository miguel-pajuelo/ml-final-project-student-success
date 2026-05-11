import pandas as pd
from exploratory_functions import *

ruta_csv_salida = "rendimiento_estudiantes_procesado.csv"

def cargar_datos(ruta_csv: str) -> pd.DataFrame:
    """
    Carga el dataset desde un archivo CSV usando ';' como separador.
    """
    return pd.read_csv(ruta_csv, sep=";", encoding="utf-8")


def comprobar_valores_faltantes(data: pd.DataFrame) -> None:
    """
    Comprueba si el dataset contiene valores faltantes y muestra un mensaje informativo.
    """
    if data.shape[0] == data.dropna().shape[0]:
        print("No existe ningún dato faltante.")
    else:
        print("Existen datos faltantes, procediendo a la limpieza manual.")


def analizar_distribucion_clases(data: pd.DataFrame, columna_objetivo: str = "objetivo") -> None:
    """
    Muestra la distribución de frecuencias de las clases de la variable objetivo
    y ofrece una interpretación básica del posible desbalanceo.
    """
    datos_totales = data.shape[0]
    distinct_classes = data[columna_objetivo].unique()

    print("\nDistribución de clases:")
    for clase in distinct_classes:
        count = (data[columna_objetivo] == clase).sum()
        porcentaje = round(count / datos_totales, 4) * 100
        print(f"{clase} - {count} - {porcentaje}%")

def separar_variables(data: pd.DataFrame, columna_objetivo: str = "objetivo") -> tuple[pd.Series, pd.DataFrame]:
    """
    Separa la variable objetivo del conjunto de variables predictoras.
    """
    y = data[columna_objetivo].copy()
    X = data.drop(columns=[columna_objetivo]).copy()
    return y, X


def procesar_objetivo(y: pd.Series) -> pd.Series:
    """
    Codifica la variable objetivo en formato numérico.
    """
    return objetive_encoder(y)


def procesar_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el preprocesado completo a las variables predictoras:
    codificación binaria, mapeos ordinales, variables dummy, reducción
    de colinealidad y normalización final.
    """
    X = X.copy()

    print("\n--- INICIO PREPROCESADO FEATURES ---\n")

    # Codificaciones binarias
    print("[Binarias]")
    print(" - asistencia_diurna_vespertina → 0=diurna, 1=vespertina")
    X["asistencia_diurna_vespertina"] = binary_diurna_vespertina(X["asistencia_diurna_vespertina"])

    print(" - becado → 0=no, 1=si")
    X["becado"] = binary_si_no_encoder(X["becado"])

    print(" - desplazado → 0=no, 1=si")
    X["desplazado"] = binary_si_no_encoder(X["desplazado"])

    print(" - deudor → 0=no, 1=si")
    X["deudor"] = binary_si_no_encoder(X["deudor"])

    print(" - genero → 0=hombre, 1=mujer")
    X["genero"] = binary_genre_encoder(X["genero"])

    print(" - internacional → 0=no, 1=si")
    X["internacional"] = binary_si_no_encoder(X["internacional"])

    print(" - matricula_al_dia → 0=no, 1=si")
    X["matricula_al_dia"] = binary_si_no_encoder(X["matricula_al_dia"])

    print(" - necesidades_educativas_especiales → 0=no, 1=si")
    X["necesidades_educativas_especiales"] = binary_si_no_encoder(X["necesidades_educativas_especiales"])

    # Mapeos ordinales
    print("\n[Ordinales]")
    print(" - cualificacion_previa → escala ordinal educativa")
    X["cualificacion_previa"] = mapeo_cualificacion_previa(X["cualificacion_previa"])

    print(" - cualificacion_madre → escala ordinal educativa")
    X["cualificacion_madre"] = mapeo_cualificacion_padres(X["cualificacion_madre"])

    print(" - cualificacion_padre → escala ordinal educativa")
    X["cualificacion_padre"] = mapeo_cualificacion_padres(X["cualificacion_padre"])

    # Transformaciones categóricas
    print("\n[Categóricas → dummies / reducción]")
    print(" - estado_civil → one-hot encoding")
    X = dummy_estado_civil(X)

    print(" - nacionalidad → binaria (portuguesa vs resto)")
    X = binary_nacionalidad(X)

    print(" - modo_solicitud → agrupación + one-hot")
    X = dummy_modo_solicitud(X)

    print(" - curso → agrupación en categorías + one-hot")
    X = dummy_carreras(X)

    print(" - ocupacion_madre/padre → agrupación por sector + one-hot")
    X = dummy_ocupacion(X)

    # Reducción de colinealidad
    print("\n[Reducción de variables]")
    print(" - ratios académicos (presentación y éxito por semestre)")
    print(" - eliminación de variables redundantes (matriculadas, evaluadas, aprobadas, etc.)")
    X = reducir_colinealidad(X)

    # Eliminación de columnas superfluas y limpieza final
    print("\n[Limpieza final]")
    print(" - eliminación de columnas: asignaturas_1sem_convalidadas, asignaturas_2sem_convalidadas")
    X = X.drop(columns=["asignaturas_1sem_convalidadas", "asignaturas_2sem_convalidadas"])

    # Normalización final global
    print("\n[Normalización de variables]")
    print(" - normalización global de todas las variables numéricas a [0,1]")
    X = normalizar_columnas_numericas(X)

    print("\n--- FIN PREPROCESADO FEATURES ---\n")

    return X


def reconstruir_dataset(y: pd.Series, X: pd.DataFrame, columna_objetivo: str = "objetivo") -> pd.DataFrame:
    """
    Reconstruye el dataset final uniendo la variable objetivo procesada
    con las variables predictoras ya transformadas.
    """
    data_procesada = X.copy()
    data_procesada[columna_objetivo] = y
    return data_procesada


def guardar_datos(data: pd.DataFrame, ruta_csv: str) -> None:
    """
    Guarda el dataset procesado en un archivo CSV usando ';' como separador.
    """
    data.to_csv(ruta_csv, sep=";", encoding="utf-8", index=False)


def main() -> None:
    """
    Ejecuta el flujo completo de carga, análisis, preprocesado y guardado del dataset.
    """
    ruta_csv = "rendimiento_estudiantes.csv"

    data = cargar_datos(ruta_csv)
    comprobar_valores_faltantes(data)
    analizar_distribucion_clases(data, columna_objetivo="objetivo")

    y, X = separar_variables(data, columna_objetivo="objetivo")
    y = procesar_objetivo(y)
    X = procesar_features(X)

    data_procesada = reconstruir_dataset(y, X, columna_objetivo="objetivo")
    guardar_datos(data_procesada, ruta_csv_salida)

    print(f"\nDataset procesado y guardado correctamente en: {ruta_csv_salida}")
    print(f"Número de filas: {data_procesada.shape[0]}")
    print(f"Número de columnas: {data_procesada.shape[1]}")


if __name__ == "__main__":
    main()