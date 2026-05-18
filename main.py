import os
import re
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse  # <-- IMPORTANTE: Para servir el HTML
from pydantic import BaseModel
import pandas as pd
import numpy as np

app = FastAPI(
    title="Anti-Shell Company Analytics Backend",
    description="API para la detección y análisis de riesgo de empresas de cartón con motor de búsqueda.",
    version="1.2.0"
)

# --- CONFIGURACIÓN DE CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- RUTA PARA SERVIR EL FRONTEND (index.html) ---
@app.get("/")
def obtener_frontend():
    """
    Sirve el archivo index.html en la raíz del sitio.
    """
    # Verificamos si el archivo existe antes de enviarlo
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"error": "Archivo index.html no encontrado en el servidor. Asegúrate de haberlo subido en la raíz junto a main.py"}


# --- FUNCIONES DE LIMPIEZA Y CARGA DE DATOS ---
def normalizar_texto_espaciado(texto: str) -> str:
    if pd.isna(texto):
        return ""
    texto_str = str(texto).strip()
    if len(texto_str) > 3 and texto_str.count(" ") >= (len(texto_str) // 3):
        texto_str = re.sub(r'(?<!\s) (?!\s)', '', texto_str)
        texto_str = re.sub(r'\s+', ' ', texto_str)
    return texto_str.strip().upper()

def load_and_merge_datasets():
    # Rutas relativas para buscar los archivos subidos
    reg_2024 = pd.read_csv("operaciones_registrales_2024.xlsx - Operaciones registrales 2024.csv")
    reg_2025 = pd.read_csv("operaciones_registrales_2025.xlsx - data.csv")
    reg_2026 = pd.read_csv("operaciones_registrales_2026.xlsx - data.csv")
    
    reg_2024['ANIO_REGISTRO'] = 2024
    reg_2025['ANIO_REGISTRO'] = 2025
    reg_2026['ANIO_REGISTRO'] = 2026
    
    registros = pd.concat([reg_2024, reg_2025, reg_2026], ignore_index=True)
    registros['NIT_PROVEEDOR'] = registros['NIT_PROVEEDOR'].astype(str).str.replace(" ", "").str.upper()
    registros['NOMBRE_PROVEEDOR'] = registros['NOMBRE_PROVEEDOR'].astype(str).str.strip().str.upper()

    try:
        top_financial = pd.read_csv("TOP102026_05_17_20_22_06.csv.csv")
        top_financial.columns = [re.sub(r'\s+', '', col) for col in top_financial.columns]
        top_financial['NIT'] = top_financial['NIT'].astype(str).str.replace(" ", "").str.upper()
        top_financial['NOMBRE'] = top_financial['NOMBRE'].apply(normalizar_texto_espaciado)
        top_financial['MONTO'] = pd.to_numeric(top_financial['MONTO'].astype(str).str.replace(" ", ""), errors='coerce').fillna(0.0)
        top_financial['CANTIDAD'] = pd.to_numeric(top_financial['CANTIDAD'].astype(str).str.replace(" ", ""), errors='coerce').fillna(0).astype(int)
    except Exception:
        top_financial = pd.DataFrame(columns=['NIT', 'NOMBRE', 'CANTIDAD', 'MONTO'])

    return registros, top_financial

# --- MODELOS DE DATOS ---
class EmpresaSearchResult(BaseModel):
    nit: str
    nombre: str
    origen_dato: str
    tipo_proveedor: Optional[str] = "No especificado"

class RiskAnalysisResponse(BaseModel):
    nit: str
    nombre: str
    tipo_proveedor: str
    capacidad_economica_registrada: str
    monto_adjudicado_total: float
    cantidad_contratos: int
    score_riesgo: float
    nivel_riesgo: str
    alertas_activadas: List[str]
    detalles_analisis: Dict[str, Any]

# --- ENDPOINTS DE LA API ---
@app.get("/api/v1/buscar", response_model=List[EmpresaSearchResult])
def buscar_empresa(query: str = Query(..., min_length=3)):
    registros, financieras = load_and_merge_datasets()
    q_limpio = query.strip().upper()
    q_sin_espacios = q_limpio.replace(" ", "")
    resultados_dict = {}

    filtro_reg = registros[
        (registros['NIT_PROVEEDOR'].str.contains(q_sin_espacios, na=False)) |
        (registros['NOMBRE_PROVEEDOR'].str.contains(q_limpio, na=False))
    ]
    for _, row in filtro_reg.drop_duplicates(subset=['NIT_PROVEEDOR']).iterrows():
        nit = row['NIT_PROVEEDOR']
        resultados_dict[nit] = {
            "nit": nit,
            "nombre": row['NOMBRE_PROVEEDOR'],
            "origen_dato": "Registro",
            "tipo_proveedor": row['TIPO_PROVEEDOR']
        }

    filtro_fin = financieras[
        (financieras['NIT'].str.contains(q_sin_espacios, na=False)) |
        (financieras['NOMBRE'].str.contains(q_limpio, na=False))
    ]
    for _, row in filtro_fin.drop_duplicates(subset=['NIT']).iterrows():
        nit = row['NIT']
        if nit in resultados_dict:
            resultados_dict[nit]["origen_dato"] = "Ambos"
        else:
            resultados_dict[nit] = {
                "nit": nit,
                "nombre": row['NOMBRE'],
                "origen_dato": "Financiero",
                "tipo_proveedor": "No especificado"
            }
            
    if not resultados_dict:
        raise HTTPException(status_code=404, detail="No se encontraron empresas.")
    return list(resultados_dict.values())

@app.get("/api/v1/analizar/{nit}", response_model=RiskAnalysisResponse)
def analizar_empresa(nit: str):
    nit_buscar = nit.strip().replace(" ", "").upper()
    registros, financieras = load_and_merge_datasets()
    
    df_prov = registros[registros['NIT_PROVEEDOR'] == nit_buscar]
    df_fin = financieras[financieras['NIT'] == nit_buscar]
    
    if df_prov.empty and df_fin.empty:
        raise HTTPException(status_code=404, detail="Proveedor no encontrado.")
    
    nombre = df_fin['NOMBRE'].iloc[0] if not df_fin.empty else (df_prov['NOMBRE_PROVEEDOR'].iloc[0] if not df_prov.empty else "DESCONOCIDO")
    tipo_prov = df_prov['TIPO_PROVEEDOR'].iloc[0] if not df_prov.empty else "No Registrado"
    capacidad = df_prov['CAPACIDAD_ECONOMICA'].fillna("NO ESPECIFICADA").iloc[0] if not df_prov.empty else "No Registrado"
    
    monto_total = float(df_fin['MONTO'].sum()) if not df_fin.empty else 0.0
    contratos_total = int(df_fin['CANTIDAD'].sum()) if not df_fin.empty else 0
    
    score = 0.0
    alertas = []
    detalles = {}
    anios_activos = df_prov['ANIO_REGISTRO'].unique().tolist()
    
    if "COMPRA DIRECTA" in str(capacidad).upper() and monto_total > 1000000:
        score += 40
        alertas.append("CAPACIDAD_ECONOMICA_EXCEDIDA: Perfil de compras directas pero facturación millonaria.")
        detalles['riesgo_capacidad'] = "Crítico: El volumen supera su perfil regulatorio."
    
    if 2024 not in anios_activos and (2025 in anios_activos or 2026 in anios_activos) and monto_total > 5000000:
        score += 30
        alertas.append("EMPRESA_EXPRESS: Alta facturación inmediata tras su inscripción reciente.")
        detalles['riesgo_antiguedad'] = "Alto: Posible empresa creada exclusivamente para adjudicaciones."
        
    if "PERSONAL TEMPORAL" in str(df_prov['TIPO_SOLICITUD'].values).upper() and monto_total > 500000:
        score += 15
        alertas.append("PERFIL_INADECUADO: Registros de recurso técnico con cobros masivos.")
        detalles['riesgo_perfil'] = "Medio: Posible uso de identidades individuales."

    if contratos_total > 0 and (monto_total / contratos_total) > 2000000:
        score += 15
        alertas.append("CONCENTRACION_ALTA_VALOR: Contratos escasos pero montos sobredimensionados.")
        detalles['riesgo_concentracion'] = "Medio: Concentración de fondos en pocas transacciones."

    if score >= 70: nivel = "CRÍTICO"
    elif score >= 45: nivel = "ALTO"
    elif score >= 20: nivel = "MEDIO"
    else: nivel = "BAJO"
        
    return RiskAnalysisResponse(
        nit=nit_buscar, nombre=nombre, tipo_proveedor=tipo_prov,
        capacidad_economica_registrada=capacidad, monto_adjudicado_total=monto_total,
        cantidad_contratos=contratos_total, score_riesgo=min(score, 100.0),
        nivel_riesgo=nivel, alertas_activadas=alertas, detalles_analisis=detalles
    )