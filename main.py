import os
import re
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np

app = FastAPI(
    title="Anti-Shell Company Analytics Backend",
    description="API para la detección y análisis de riesgo de empresas de cartón / fantasmas en contratación pública.",
    version="1.0.0"
)

# --- REPOSITORIO DE DATOS DE SIMULACIÓN / CARGA ---
# En un entorno real, esto se conectaría a una base de datos (PostgreSQL, BigQuery, etc.)
DATA_DIR = "./data"

def clean_spaced_csv(file_path: str) -> pd.DataFrame:
    """
    Algunos reportes CSV exportados pueden venir con espacios intercalados (ej: 'N I T').
    Esta función remueve espacios adicionales si el formato viene codificado de forma extraña.
    """
    if not os.path.exists(file_path):
        return pd.DataFrame()
    
    # Intenta leer normalmente
    df = pd.read_csv(file_path)
    
    # Si las columnas contienen espacios vacíos intercalados (ej: 'N O M B R E'), normalizamos
    if any(" " in str(col) for col in df.columns if len(str(col)) > 3):
        # Limpieza rápida de strings espaciados
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        # Remover espacios que dividen letras pero mantener estructura
        # Nota: Ajustar según la codificación real del archivo (UTF-16 vs UTF-8)
        pass 
    return df

def load_and_merge_datasets():
    """
    Carga y consolida los archivos registrales (2024-2026) y los montos de adjudicación.
    """
    # Ejemplo de paths (ajusta a la ubicación real de tus archivos)
    reg_2024 = pd.read_csv("operaciones_registrales_2024.xlsx - Operaciones registrales 2024.csv")
    reg_2025 = pd.read_csv("operaciones_registrales_2025.xlsx - data.csv")
    reg_2026 = pd.read_csv("operaciones_registrales_2026.xlsx - data.csv")
    
    # Homologar columnas y años
    reg_2024['ANIO_REGISTRO'] = 2024
    reg_2025['ANIO_REGISTRO'] = 2025
    reg_2026['ANIO_REGISTRO'] = 2026
    
    registros = pd.concat([reg_2024, reg_2025, reg_2026], ignore_index=True)
    # Limpiar NITs eliminando espacios ocultos
    registros['NIT_PROVEEDOR'] = registros['NIT_PROVEEDOR'].astype(str).str.strip().str.upper()
    
    # Cargar montos acumulados (TOP10 u Ofertas)
    # Nota: Los archivos TOP10 contienen columnas: NIT, NOMBRE, CANTIDAD, MONTO
    # Simulamos la unificación de los dataframes de montos adjudicados
    try:
        # Reemplaza con la lógica de limpieza de los archivos 'TOP10...' cargados
        # Si vienen separados por espacios ('N I T'), remover los espacios con regex
        top_financial = pd.read_csv("TOP102026_05_17_20_22_06.csv.csv") 
        # Normalizar nombres de columnas si vienen con espacios: 'N I T' -> 'NIT'
        top_financial.columns = [re.sub(r'\s+', '', col) for col in top_financial.columns]
        top_financial['NIT'] = top_financial['NIT'].astype(str).str.replace(" ", "").str.upper()
        top_financial['MONTO'] = pd.to_numeric(top_financial['MONTO'].astype(str).str.replace(" ", ""), errors='coerce')
        top_financial['CANTIDAD'] = pd.to_numeric(top_financial['CANTIDAD'].astype(str).str.replace(" ", ""), errors='coerce')
    except Exception:
        # Fallback en caso de que varíe el formato del archivo en disco
        top_financial = pd.DataFrame(columns=['NIT', 'NOMBRE', 'CANTIDAD', 'MONTO'])

    return registros, top_financial

# --- MODELOS DE RESPUESTA (Pydantic) ---
class RiskAnalysisResponse(BaseModel):
    nit: str
    nombre: str
    tipo_proveedor: str
    capacidad_economica_registrada: str
    monto_adjudicado_total: float
    cantidad_contratos: int
    score_riesgo: float
    nivel_riesgo: str  # BAJO, MEDIO, ALTO, CRÍTICO
    alertas_activadas: List[str]
    detalles_analisis: Dict[str, Any]

# --- ENDPOINTS ---

@app.get("/api/v1/analizar/{nit}", response_model=RiskAnalysisResponse)
def analizar_empresa(nit: str):
    nit_buscar = nit.strip().upper()
    registros, financieras = load_and_merge_datasets()
    
    # 1. Buscar información registral
    df_prov = registros[registros['NIT_PROVEEDOR'] == nit_buscar]
    # 2. Buscar información financiera/adjudicaciones
    df_fin = financieras[financieras['NIT'] == nit_buscar]
    
    if df_prov.empty and df_fin.empty:
        raise HTTPException(status_code=404, detail="Proveedor no encontrado en las bases de datos analizadas.")
    
    # Extraer datos básicos
    nombre = df_fin['NOMBRE'].iloc[0] if not df_fin.empty else (df_prov['NOMBRE_PROVEEDOR'].iloc[0] if not df_prov.empty else "DESCONOCIDO")
    tipo_prov = df_prov['TIPO_PROVEEDOR'].iloc[0] if not df_prov.empty else "No Registrado"
    capacidad = df_prov['CAPACIDAD_ECONOMICA'].fillna("NO ESPECIFICADA").iloc[0] if not df_prov.empty else "No Registrado"
    
    monto_total = float(df_fin['MONTO'].sum()) if not df_fin.empty else 0.0
    contratos_total = int(df_fin['CANTIDAD'].sum()) if not df_fin.empty else 0
    
    # --- ALGORITMO DE SCORE DE RIESGO ---
    score = 0.0
    alertas = []
    detalles = {}
    
    # Historial de registros por año
    anios_activos = df_prov['ANIO_REGISTRO'].unique().tolist()
    
    # REGLA 1: Desproporción de Capacidad Económica vs Monto Adjudicado
    if "COMPRA DIRECTA" in str(capacidad).upper() and monto_total > 1000000:
        # La Compra Directa suele ser para montos bajos (ej. menores a Q90,000 en Guatemala)
        # Si factura millones, es una anomalía crítica.
        score += 40
        alertas.append("CAPACIDAD_ECONOMICA_EXCEDIDA: Registrado solo para compras directas pero con facturación millonaria.")
        detalles['riesgo_capacidad'] = "Crítico: El volumen adjudicado supera exponencialmente su perfil regulatorio."
    
    # REGLA 2: Empresa "Express" o Sin Historial (Creación Reciente + Alta Facturación)
    if 2024 not in anios_activos and (2025 in anios_activos or 2026 in anios_activos) and monto_total > 5000000:
        score += 30
        alertas.append("EMPRESA_EXPRESS: Sin actividad en 2024, registro reciente y obtención inmediata de contratos masivos.")
        detalles['riesgo_antiguedad'] = "Alto: Comportamiento típico de empresas constituidas únicamente para ganar licitaciones específicas."
        
    # REGLA 3: Simulación de Personal de Staff / Consultores Outsourcing actuando como macro-proveedores
    if "PERSONAL TEMPORAL" in str(df_prov['TIPO_SOLICITUD'].values).upper() and monto_total > 500000:
        score += 15
        alertas.append("PERFIL_INADECUADO: Registros asociados a personal temporal u operaciones técnicas, pero recibe fondos de contratista mayor.")
        detalles['riesgo_perfil'] = "Medio: Uso de NITs de personas individuales para canalizar fondos corporativos masivos."

    # REGLA 4: Concentración de Contratos con Pocos Movimientos / Alta Frecuencia
    if contratos_total > 0 and (monto_total / contratos_total) > 2000000:
        score += 15
        alertas.append("CONCENTRACION_ALTA_VALOR: Promedio de monto por contrato inusualmente elevado.")
        detalles['riesgo_concentracion'] = "Medio: Pocos contratos pero de montos extremadamente altos, típico de adjudicaciones dirigidas."

    # Determinar etiqueta de riesgo
    if score >= 70:
        nivel = "CRÍTICO"
    elif score >= 45:
        nivel = "ALTO"
    elif score >= 20:
        nivel = "MEDIO"
    else:
        nivel = "BAJO"
        
    return RiskAnalysisResponse(
        nit=nit_buscar,
        nombre=nombre,
        tipo_proveedor=tipo_prov,
        capacidad_economica_registrada=capacidad,
        monto_adjudicado_total=monto_total,
        cantidad_contratos=contratos_total,
        score_riesgo=min(score, 100.0),
        nivel_riesgo=nivel,
        alertas_activadas=alertas,
        detalles_analisis=detalles
    )

@app.get("/api/v1/ranking-riesgo", response_model=List[Dict[str, Any]])
def obtener_ranking_riesgo(limite: int = Query(default=10, ge=1, le=100)):
    """
    Devuelve un listado de los proveedores con mayor riesgo detectado en el sistema
    para auditoría preventiva.
    """
    registros, financieras = load_and_merge_datasets()
    nits_unicos = financieras['NIT'].dropna().unique().tolist()
    
    ranking = []
    for nit in nits_unicos[:50]: # Analizar top 50 para no penalizar performance en la demo
        try:
            analisis = analizar_empresa(nit)
            if analisis.score_riesgo > 0:
                ranking.append(analisis.dict())
        except HTTPException:
            continue
            
    # Ordenar por score de riesgo descendente
    ranking = sorted(ranking, key=lambda x: x['score_riesgo'], reverse=True)
    return ranking[:limite]

if __name__ == "__main__":
    import uvicorn
    # Ejecutar servidor localmente
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)