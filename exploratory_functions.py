"""
SCRIPT AUXILIAR QUE CONTIENE LAS FUNCIONES NECESARIAS PARA EL PROCESADO DE LOS DATOS
"""
import numpy as np
import pandas as pd

mapping_modo_solicitud = {
"1a_fase_contingente_general": "contingente_general",
"2a_fase_contingente_general": "contingente_general",
"3a_fase_contingente_general": "contingente_general",

"1a_fase_contingente_especial_azores": "contingente_especial",
"1a_fase_contingente_especial_madeira": "contingente_especial",

"cambio_de_curso": "cambio_traslado",
"cambio_de_institucion_curso": "cambio_traslado",
"cambio_de_institucion_curso_internacional": "cambio_traslado",
"traslado": "cambio_traslado",

"estudiante_internacional_licenciatura": "internacional",

"mayor_de_23_anos": "mayor_23",

"titular_de_diploma_de_ciclo_corto": "titulacion_previa",
"titular_de_diploma_de_especializacion_tecnologica": "titulacion_previa",
"titular_de_otro_curso_superior": "titulacion_previa",

"ordenanza_533_a_99_apartado_b2_plan_diferente": "via_legal",
"ordenanza_533_a_99_apartado_b3_otra_institucion": "via_legal",
"ordenanza_612_93": "via_legal",
"ordenanza_854_b_99": "via_legal",
}

mapping_cualificacion_previa = {
"educacion_basica_2o_ciclo_o_equivalente": 0,
"educacion_basica_3er_ciclo_o_equivalente": 1,
"10o_ano_no_completado": 2,
"10o_ano": 3,
"11o_ano_no_completado": 4,
"otro_11o_ano": 5,
"12o_ano_no_completado": 6,
"educacion_secundaria": 7,
"curso_de_especializacion_tecnologica": 8,
"curso_tecnico_superior_profesional": 9,
"asistencia_previa_a_educacion_superior": 10,
"educacion_superior_grado_1er_ciclo": 11,
"educacion_superior_grado": 12,
"educacion_superior_licenciatura": 13,
"educacion_superior_master_2o_ciclo": 14,
"educacion_superior_master": 15,
"educacion_superior_doctorado": 16
}

mapping_cualificacion_padres = {
"desconocido": -1,

"no_sabe_leer_ni_escribir": 0,
"sabe_leer_sin_4o_ano": 1,

"7o_ano_antiguo": 2,
"7o_ano": 2,

"8o_ano": 3,
"9o_ano_no_completado": 4,

"educacion_basica_1er_ciclo_o_equivalente": 5,
"educacion_basica_2o_ciclo_o_equivalente": 6,
"educacion_basica_3er_ciclo_o_equivalente": 7,

"10o_ano": 8,
"11o_ano_no_completado": 9,
"otro_11o_ano": 10,

"12o_ano_no_completado": 11,
"curso_complementario_de_secundaria_no_concluido": 11,

"2o_ano_complementario_de_secundaria": 12,
"curso_complementario_de_secundaria": 12,
"2o_ciclo_del_bachillerato_general": 12,
"educacion_secundaria_12o_ano_o_equivalente": 13,

"curso_general_de_comercio": 14,
"curso_general_de_administracion_y_comercio": 14,
"contabilidad_y_administracion_complementaria": 14,

"curso_tecnico_profesional": 15,
"curso_de_especializacion_tecnologica": 16,
"curso_superior_especializado": 17,
"curso_tecnico_superior_profesional": 18,

"frecuencia_de_educacion_superior": 19,

"educacion_superior_grado_1er_ciclo": 20,
"educacion_superior_grado": 21,
"educacion_superior_licenciatura": 22,
"educacion_superior_master_2o_ciclo": 23,
"educacion_superior_master": 24,
"educacion_superior_doctorado_3er_ciclo": 25,
"educacion_superior_doctorado": 26
}

mapping_carreras = {
    "agronomia": "agro_equi",
    "equicultura": "agro_equi",

    "animacion_y_diseno_multimedia": "animacion_diseno_multimedia",

    "diseno_de_comunicacion": "comunicacion_marketing",
    "gestion_de_publicidad_y_marketing": "comunicacion_marketing",
    "periodismo_y_comunicacion": "comunicacion_marketing",

    "educacion_basica": "educacion_basica",
    
    "servicio_social": "servicio_social",
    "servicio_social_nocturno": "servicio_social",

    "enfermeria": "salud",
    "higiene_oral": "salud",
    "enfermeria_veterinaria": "salud",

    "gestion": "gestion",
    "gestion_nocturno": "gestion",

    "ingenieria_informatica": "tecnologia_ingenieria",
    "tecnologias_de_produccion_de_biocombustibles": "tecnologia_ingenieria",

    "turismo": "turismo"
}

mapping_ocupacion = {
    # 1. SIN ACTIVIDAD / NO INFORMADO
    "en_blanco": "sin_actividad_o_no_informado",
    "otra_situacion": "sin_actividad_o_no_informado",
    "estudiante": "sin_actividad_o_no_informado",

    # 2. DIRECCION / GESTION
    "representantes_del_poder_legislativo_y_ejecutivo_directores_y_gerentes_ejecutivos": "direccion_gestion",
    "directores_de_hoteleria_restauracion_comercio_y_otros_servicios": "direccion_gestion",
    "directores_de_servicios_administrativos_y_comerciales": "direccion_gestion",

    # 3. PROFESIONAL CUALIFICADO
    "docentes": "profesional_cualificado",
    "profesionales_de_la_salud": "profesional_cualificado",
    "especialistas_en_actividades_intelectuales_y_cientificas": "profesional_cualificado",
    "especialistas_en_tic": "profesional_cualificado",
    "especialistas_en_ciencias_fisicas_matematicas_ingenieria_y_afines": "profesional_cualificado",
    "especialistas_en_finanzas_contabilidad_organizacion_administrativa_y_relaciones_publicas_y_comerciales": "profesional_cualificado",

    # 4. TECNICO INTERMEDIO
    "tecnicas_y_profesiones_intermedias_de_ciencia_e_ingenieria": "tecnico_intermedio",
    "tecnicos_y_profesiones_intermedias_de_ciencia_e_ingenieria": "tecnico_intermedio",
    "tecnicas_y_profesionales_intermedias_de_salud": "tecnico_intermedio",
    "tecnicos_y_profesionales_intermedios_de_salud": "tecnico_intermedio",
    "tecnicas_intermedias_de_servicios_juridicos_sociales_deportivos_culturales_y_similares": "tecnico_intermedio",
    "tecnicos_intermedios_de_servicios_juridicos_sociales_deportivos_culturales_y_similares": "tecnico_intermedio",
    "tecnicos_de_tecnologias_de_la_informacion_y_la_comunicacion": "tecnico_intermedio",
    "tecnicos_y_profesiones_de_nivel_intermedio": "tecnico_intermedio",

    # 5. ADMINISTRATIVO / OFICINA
    "empleadas_de_oficina_secretarias_y_operadoras_de_procesamiento_de_datos": "administrativo_oficina",
    "empleados_de_oficina_secretarios_y_operadores_de_procesamiento_de_datos": "administrativo_oficina",
    "operadoras_de_datos_contabilidad_estadistica_servicios_financieros_y_registro": "administrativo_oficina",
    "operadores_de_datos_contabilidad_estadistica_servicios_financieros_y_registro": "administrativo_oficina",
    "otro_personal_de_apoyo_administrativo": "administrativo_oficina",
    "personal_administrativo": "administrativo_oficina",

    # 6. SERVICIOS / VENTAS / CUIDADOS
    "ayudantes_de_preparacion_de_comidas": "servicios_ventas_cuidados",
    "cuidadoras_y_trabajadoras_de_atencion_personal": "servicios_ventas_cuidados",
    "cuidadores_y_trabajadores_de_atencion_personal": "servicios_ventas_cuidados",
    "trabajadoras_de_servicios_personales": "servicios_ventas_cuidados",
    "trabajadores_de_servicios_personales": "servicios_ventas_cuidados",
    "trabajadoras_de_servicios_personales_seguridad_y_vendedoras": "servicios_ventas_cuidados",
    "trabajadores_de_servicios_personales_seguridad_y_vendedores": "servicios_ventas_cuidados",
    "vendedoras": "servicios_ventas_cuidados",
    "vendedores": "servicios_ventas_cuidados",
    "vendedores_ambulantes_y_prestadores_de_servicios_callejeros": "servicios_ventas_cuidados",
    "personal_de_limpieza": "servicios_ventas_cuidados",
    "personal_de_proteccion_y_seguridad": "servicios_ventas_cuidados",

    # 7. AGRICULTURA / PESCA
    "agricultoras_y_trabajadoras_cualificadas_de_agricultura_pesca_y_silvicultura": "agricultura_pesca",
    "agricultores_y_trabajadores_cualificados_de_agricultura_pesca_y_silvicultura": "agricultura_pesca",
    "agricultores_orientados_al_mercado_y_trabajadores_agropecuarios_cualificados": "agricultura_pesca",
    "agricultores_ganaderos_pescadores_cazadores_y_recolectores_de_subsistencia": "agricultura_pesca",
    "trabajadoras_no_cualificadas_de_agricultura_ganaderia_pesca_y_silvicultura": "agricultura_pesca",
    "trabajadores_no_cualificados_de_agricultura_ganaderia_pesca_y_silvicultura": "agricultura_pesca",

    # 8. INDUSTRIA / CONSTRUCCION / TRANSPORTE / MAQUINARIA
    "operadoras_de_instalaciones_y_maquinaria_y_montadoras": "industria_construccion_transporte",
    "operadores_de_instalaciones_y_maquinaria_y_montadores": "industria_construccion_transporte",
    "operadores_de_plantas_fijas_y_maquinaria": "industria_construccion_transporte",
    "conductores_de_vehiculos_y_operadores_de_equipos_moviles": "industria_construccion_transporte",
    "montadores": "industria_construccion_transporte",
    "trabajadoras_cualificadas_de_impresion_instrumentos_de_precision_joyeria_y_artesania": "industria_construccion_transporte",
    "trabajadoras_cualificadas_de_industria_construccion_y_artesania": "industria_construccion_transporte",
    "trabajadores_cualificados_de_industria_construccion_y_artesania": "industria_construccion_transporte",
    "trabajadoras_cualificadas_de_la_construccion_excepto_electricistas": "industria_construccion_transporte",
    "trabajadores_cualificados_de_la_construccion_excepto_electricistas": "industria_construccion_transporte",
    "trabajadores_cualificados_de_electricidad_y_electronica": "industria_construccion_transporte",
    "trabajadores_cualificados_de_metalurgia_metalmecanica_y_similares": "industria_construccion_transporte",
    "trabajadoras_de_industrias_alimentarias_madera_confeccion_y_otras_industrias_y_artes": "industria_construccion_transporte",
    "trabajadores_de_industrias_alimentarias_madera_confeccion_y_otras_industrias_y_artes": "industria_construccion_transporte",
    "trabajadoras_no_cualificadas_de_industria_extractiva_construccion_manufactura_y_transporte": "industria_construccion_transporte",
    "trabajadores_no_cualificados_de_industria_extractiva_construccion_manufactura_y_transporte": "industria_construccion_transporte",

    # 9. NO CUALIFICADO GENERAL
    "trabajadoras_no_cualificadas": "no_cualificado",
    "trabajadores_no_cualificados": "no_cualificado",

    # 10. FUERZAS ARMADAS
    "profesiones_de_las_fuerzas_armadas": "fuerzas_armadas",
    "oficiales_de_las_fuerzas_armadas": "fuerzas_armadas",
    "sargentos_de_las_fuerzas_armadas": "fuerzas_armadas",
    "otro_personal_de_las_fuerzas_armadas": "fuerzas_armadas"
}

"""
FUNCIONES AUXILIARES
"""

def comprobar_mapeo_completo(serie: pd.Series, nombre_columna: str, valores_validos: set) -> None:
    """
    Comprueba si todos los valores de una columna están contemplados
    en el mapeo esperado.
    """
    valores_encontrados = set(serie.dropna().unique())
    valores_no_contemplados = valores_encontrados - valores_validos

    if len(valores_no_contemplados) > 0:
        raise ValueError(
            f"La columna '{nombre_columna}' contiene categorías no contempladas: "
            f"{sorted(valores_no_contemplados)}"
        )


def objetive_encoder(y: pd.Series) -> pd.Series:
    """
    Codifica la variable objetivo en formato numérico manteniendo
    un orden fijo y explícito.
    """
    mapping_objetivo = {
        "abandono": 0,
        "graduado": 1,
        "matriculado": 2
    }

    comprobar_mapeo_completo(y, "objetivo", set(mapping_objetivo.keys()))
    return y.map(mapping_objetivo).astype(int)


def binary_si_no_encoder(X: pd.Series) -> pd.Series:
    """
    Codifica variables binarias si/no.
    """
    mapping = {"si": 1, "no": 0}
    comprobar_mapeo_completo(X, X.name, set(mapping.keys()))
    return X.map(mapping).astype(int)


def binary_genre_encoder(X: pd.Series) -> pd.Series:
    """
    Codifica la variable género.
    """
    mapping = {"mujer": 1, "hombre": 0}
    comprobar_mapeo_completo(X, X.name, set(mapping.keys()))
    return X.map(mapping).astype(int)


def binary_diurna_vespertina(X: pd.Series) -> pd.Series:
    """
    Codifica el régimen diurno/vespertino.
    """
    mapping = {"vespertina": 1, "diurna": 0}
    comprobar_mapeo_completo(X, X.name, set(mapping.keys()))
    return X.map(mapping).astype(int)


def binary_nacionalidad(X: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa nacionalidad en portuguesa vs extranjera y la codifica.
    """
    X = X.copy()
    X["nacionalidad"] = X["nacionalidad"].apply(
        lambda x: "portuguesa" if x == "portuguesa" else "extranjero"
    )
    X["nacionalidad"] = X["nacionalidad"].map({
        "portuguesa": 1,
        "extranjero": 0
    }).astype(int)
    return X


def mapeo_cualificacion_previa(X: pd.Series) -> pd.Series:
    """
    Aplica el mapeo ordinal de cualificación previa.
    """
    comprobar_mapeo_completo(X, X.name, set(mapping_cualificacion_previa.keys()))
    return X.map(mapping_cualificacion_previa).astype(int)


def mapeo_cualificacion_padres(X: pd.Series) -> pd.Series:
    """
    Aplica el mapeo ordinal de cualificación de padre/madre.
    """
    comprobar_mapeo_completo(X, X.name, set(mapping_cualificacion_padres.keys()))
    return X.map(mapping_cualificacion_padres).astype(int)


def dummy_estado_civil(X: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica one-hot encoding a estado_civil eliminando una categoría base.
    """
    X = X.copy()

    dummies = pd.get_dummies(
        X["estado_civil"],
        prefix="estado_civil",
        drop_first=True
    ).astype(int)

    X = X.drop(columns=["estado_civil"])
    X = pd.concat([X, dummies], axis=1)

    return X


def dummy_modo_solicitud(X: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa y codifica modo_solicitud con one-hot encoding.
    """
    X = X.copy()

    comprobar_mapeo_completo(X["modo_solicitud"], "modo_solicitud", set(mapping_modo_solicitud.keys()))
    X["modo_solicitud"] = X["modo_solicitud"].map(mapping_modo_solicitud)

    dummies = pd.get_dummies(
        X["modo_solicitud"],
        prefix="modo_solicitud",
        drop_first=True
    ).astype(int)

    X = X.drop(columns=["modo_solicitud"])
    X = pd.concat([X, dummies], axis=1)

    return X


def dummy_carreras(X: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa y codifica la variable curso con one-hot encoding.
    """
    X = X.copy()

    comprobar_mapeo_completo(X["curso"], "curso", set(mapping_carreras.keys()))
    X["curso"] = X["curso"].map(mapping_carreras)

    dummies = pd.get_dummies(
        X["curso"],
        prefix="carrera",
        drop_first=True
    ).astype(int)

    X = X.drop(columns=["curso"])
    X = pd.concat([X, dummies], axis=1)

    return X


def dummy_ocupacion(X: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa y codifica ocupación de madre y padre con one-hot encoding.
    """
    X = X.copy()

    comprobar_mapeo_completo(X["ocupacion_madre"], "ocupacion_madre", set(mapping_ocupacion.keys()))
    comprobar_mapeo_completo(X["ocupacion_padre"], "ocupacion_padre", set(mapping_ocupacion.keys()))

    X["ocupacion_madre"] = X["ocupacion_madre"].map(mapping_ocupacion)
    X["ocupacion_padre"] = X["ocupacion_padre"].map(mapping_ocupacion)

    dummies_madre = pd.get_dummies(
        X["ocupacion_madre"],
        prefix="ocupacion_madre",
        drop_first=True
    ).astype(int)

    X = X.drop(columns=["ocupacion_madre"])
    X = pd.concat([X, dummies_madre], axis=1)

    dummies_padre = pd.get_dummies(
        X["ocupacion_padre"],
        prefix="ocupacion_padre",
        drop_first=True
    ).astype(int)

    X = X.drop(columns=["ocupacion_padre"])
    X = pd.concat([X, dummies_padre], axis=1)

    return X


def normalization(X: pd.Series) -> pd.Series:
    """
    Normaliza una serie al intervalo [0,1].
    Si todos los valores son iguales, devuelve 0 en toda la columna.
    """
    X_max, X_min = X.max(), X.min()

    if X_max == X_min:
        return pd.Series(np.zeros(len(X)), index=X.index)

    return (X - X_min) / (X_max - X_min)


def normalizar_columnas_numericas(X: pd.DataFrame, excluir: list[str] = None) -> pd.DataFrame:
    """
    Normaliza al intervalo [0,1] todas las columnas numéricas salvo las excluidas.
    """
    X = X.copy()

    if excluir is None:
        excluir = []

    columnas_numericas = X.select_dtypes(include=[np.number]).columns.tolist()
    columnas_a_normalizar = [col for col in columnas_numericas if col not in excluir]

    for col in columnas_a_normalizar:
        X[col] = normalization(X[col])

    return X


def comprobar_dataset_numerico(X: pd.DataFrame) -> None:
    """
    Comprueba que todas las columnas del dataset sean numéricas.
    """
    columnas_no_numericas = X.select_dtypes(exclude=[np.number]).columns.tolist()

    if len(columnas_no_numericas) > 0:
        raise ValueError(
            f"Persisten columnas no numéricas tras el preprocesado: {columnas_no_numericas}"
        )


def comprobar_valores_faltantes_post_procesado(X: pd.DataFrame) -> None:
    """
    Comprueba si el preprocesado ha generado valores faltantes.
    """
    total_nan = X.isna().sum().sum()

    if total_nan > 0:
        columnas_con_nan = X.columns[X.isna().any()].tolist()
        raise ValueError(
            f"El preprocesado ha generado valores faltantes en las columnas: {columnas_con_nan}"
        )
    
def reducir_colinealidad(X: pd.DataFrame) -> pd.DataFrame:
    """
    Genera ratios académicos por semestre y elimina variables redundantes
    altamente relacionadas con ellos.
    """
    # 1er semestre
    X["ratio_presentacion_1sem"] = (
        X["asignaturas_1sem_evaluadas"] / X["asignaturas_1sem_matriculadas"]
    ).where(X["asignaturas_1sem_matriculadas"] != 0, 0)

    X["ratio_exito_1sem"] = (
        X["asignaturas_1sem_aprobadas"] / X["asignaturas_1sem_evaluadas"]
    ).where(X["asignaturas_1sem_evaluadas"] != 0, 0)

    # 2º semestre
    X["ratio_presentacion_2sem"] = (
        X["asignaturas_2sem_evaluadas"] / X["asignaturas_2sem_matriculadas"]
    ).where(X["asignaturas_2sem_matriculadas"] != 0, 0)

    X["ratio_exito_2sem"] = (
        X["asignaturas_2sem_aprobadas"] / X["asignaturas_2sem_evaluadas"]
    ).where(X["asignaturas_2sem_evaluadas"] != 0, 0)

    columnas_academicas_a_eliminar = [
        "asignaturas_1sem_matriculadas",
        "asignaturas_1sem_evaluadas",
        "asignaturas_1sem_aprobadas",
        "asignaturas_1sem_sin_evaluacion",
        "asignaturas_2sem_matriculadas",
        "asignaturas_2sem_evaluadas",
        "asignaturas_2sem_aprobadas",
        "asignaturas_2sem_sin_evaluacion",
    ]

    X = X.drop(columns=columnas_academicas_a_eliminar)

    return X


def duplicate_minority_only(X, y):
    """
    Duplica únicamente la clase minoritaria.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    clases, counts = np.unique(y, return_counts=True)

    idx_min = np.argmin(counts)
    clase_minoritaria = clases[idx_min]

    X_min = X[y == clase_minoritaria]
    y_min = y[y == clase_minoritaria]

    X_resto = X[y != clase_minoritaria]
    y_resto = y[y != clase_minoritaria]

    X_min_duplicada = np.vstack([X_min, X_min])
    y_min_duplicada = np.hstack([y_min, y_min])

    X_final = np.vstack([X_resto, X_min_duplicada])
    y_final = np.hstack([y_resto, y_min_duplicada])

    return X_final, y_final