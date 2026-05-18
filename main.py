import os
import re
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np



app = FastAPI(
    title="Anti-Shell Company Analytics Backend",
    version="1.1.0"
)

# --- CONFIGURACIÓN DE CORS PARA EL FRONTEND ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite que cualquier frontend local o remoto consulte la API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- FUNCIONES DE LIMPIEZA Y CARGA DE DATOS ---

def normalizar_texto_espaciado(texto: str) -> str:
    """
    Remueve espacios excesivos. Si el texto viene con letras separadas por espacios 
    (ej: 'G U A T E M A L A' o '7 8 4 8 5'), une los caracteres correctamente.
    """
    if pd.isna(texto):
        return ""
    texto_str = str(texto).strip()
    # Si detecta que hay espacios intercalados masivamente (más espacios que letras juntas)
    if len(texto_str) > 3 and texto_str.count(" ") >= (len(texto_str) // 3):
        # Remover espacios individuales pero respetar dobles espacios (que separan palabras)
        # Un truco común para este tipo de exports corruptos:
        texto_str = re.sub(r'(?<!\s) (?!\s)', '', texto_str)
        # Reemplazar múltiples espacios por uno solo
        texto_str = re.sub(r'\s+', ' ', texto_str)
    return texto_str.strip().upper()

def load_and_merge_datasets():
    """
    Carga y consolida los archivos registrales y financieros normalizando los formatos.
    """
    # 1. Cargar registros (Formatos estándar)
    reg_2024 = pd.read_csv("operaciones_registrales_2024.xlsx - Operaciones registrales 2024.csv")
    reg_2025 = pd.read_csv("operaciones_registrales_2025.xlsx - data.csv")
    reg_2026 = pd.read_csv("operaciones_registrales_2026.xlsx - data.csv")
    
    reg_2024['ANIO_REGISTRO'] = 2024
    reg_2025['ANIO_REGISTRO'] = 2025
    reg_2026['ANIO_REGISTRO'] = 2026
    
    registros = pd.concat([reg_2024, reg_2025, reg_2026], ignore_index=True)
    
    # Normalizar columnas registrales
    registros['NIT_PROVEEDOR'] = registros['NIT_PROVEEDOR'].astype(str).str.replace(" ", "").str.upper()
    registros['NOMBRE_PROVEEDOR'] = registros['NOMBRE_PROVEEDOR'].astype(str).str.strip().str.upper()

    # 2. Cargar reportes financieros (Manejo de formato con espacios intercalados 'N I T')
    try:
        top_financial = pd.read_csv("TOP102026_05_17_20_22_06.csv.csv")
        # Limpiar los nombres de las columnas que vienen espaciados ('N I T' -> 'NIT')
        top_financial.columns = [re.sub(r'\s+', '', col) for col in top_financial.columns]
        
        # Aplicar normalización a las celdas
        top_financial['NIT'] = top_financial['NIT'].astype(str).str.replace(" ", "").str.upper()
        top_financial['NOMBRE'] = top_financial['NOMBRE'].apply(normalizar_texto_espaciado)
        top_financial['MONTO'] = pd.to_numeric(top_financial['MONTO'].astype(str).str.replace(" ", ""), errors='coerce').fillna(0.0)
        top_financial['CANTIDAD'] = pd.to_numeric(top_financial['CANTIDAD'].astype(str).str.replace(" ", ""), errors='coerce').fillna(0).astype(int)
    except Exception as e:
        print(f"Error cargando archivo financiero: {e}")
        top_financial = pd.DataFrame(columns=['NIT', 'NOMBRE', 'CANTIDAD', 'MONTO'])

    return registros, top_financial

# --- MODELOS DE DATOS (Pydantic) ---

class EmpresaSearchResult(BaseModel):
    nit: str
    nombre: str
    origen_dato: str  # 'Registro', 'Financiero' o 'Ambos'
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

# --- ENDPOINTS ---

@app.get("/api/v1/buscar", response_model=List[EmpresaSearchResult])
def buscar_empresa(
    query: str = Query(..., min_length=3, description="Escribe el NIT o el Nombre (o parte de él) para buscar la empresa")
):
    """
    Busca coincidencias de empresas en el sistema por NIT o Nombre.
    Soporta búsquedas parciales (Ej: si buscas 'Construcciones', traerá todas las que coincidan).
    """
    registros, financieras = load_and_merge_datasets()
    
    # Limpiar el criterio de búsqueda del usuario
    q_limpio = query.strip().upper()
    q_sin_espacios = q_limpio.replace(" ", "")
    
    resultados_dict = {}

    # 1. Buscar en la base de datos de Operaciones Registrales
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

    # 2. Buscar en la base de datos Financiera (TOP10 / Adjudicaciones)
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
                "tipo_proveedor": "No especificado en Registro"
            }
            
    if not resultados_dict:
        raise HTTPException(status_code=404, detail="No se encontraron empresas que coincidan con el criterio.")

    return list(resultados_dict.values())


@app.get("/api/v1/analizar/{nit}", response_model=RiskAnalysisResponse)
def analizar_empresa(nit: str):
    """
    Analiza las alertas rojas y calcula el score de riesgo de una empresa usando su NIT exacto.
    """
    nit_buscar = nit.strip().replace(" ", "").upper()
    registros, financieras = load_and_merge_datasets()
    
    df_prov = registros[registros['NIT_PROVEEDOR'] == nit_buscar]
    df_fin = financieras[financieras['NIT'] == nit_buscar]
    
    if df_prov.empty and df_fin.empty:
        raise HTTPException(status_code=404, detail="Proveedor no encontrado para análisis.")
    
    nombre = df_fin['NOMBRE'].iloc[0] if not df_fin.empty else (df_prov['NOMBRE_PROVEEDOR'].iloc[0] if not df_prov.empty else "DESCONOCIDO")
    tipo_prov = df_prov['TIPO_PROVEEDOR'].iloc[0] if not df_prov.empty else "No Registrado"
    capacidad = df_prov['CAPACIDAD_ECONOMICA'].fillna("NO ESPECIFICADA").iloc[0] if not df_prov.empty else "No Registrado"
    
    monto_total = float(df_fin['MONTO'].sum()) if not df_fin.empty else 0.0
    contratos_total = int(df_fin['CANTIDAD'].sum()) if not df_fin.empty else 0
    
    # --- ALGORITMO DE SCORE DE RIESGO ---
    score = 0.0
    alertas = []
    detalles = {}
    
    anios_activos = df_prov['ANIO_REGISTRO'].unique().tolist()
    
    # Alerta 1: Desproporción de Capacidad
    if "COMPRA DIRECTA" in str(capacidad).upper() and monto_total > 1000000:
        score += 40
        alertas.append("CAPACIDAD_ECONOMICA_EXCEDIDA: Perfil de compras directas pero facturación millonaria.")
        detalles['riesgo_capacidad'] = "Crítico: El volumen adjudicado supera su perfil regulatorio."
    
    # Alerta 2: Empresa Express
    if 2024 not in anios_activos and (2025 in anios_activos or 2026 in anios_activos) and monto_total > 5000000:
        score += 30
        alertas.append("EMPRESA_EXPRESS: Alta facturación inmediata tras su inscripción reciente.")
        detalles['riesgo_antiguedad'] = "Alto: Comportamiento típico de empresas creadas exclusivamente para fines de adjudicación rápida."
        
    # Alerta 3: Perfil Inadecuado
    if "PERSONAL TEMPORAL" in str(df_prov['TIPO_SOLICITUD'].values).upper() and monto_total > 500000:
        score += 15
        alertas.append("PERFIL_INADECUADO: Registros de recurso técnico/temporal con cobros masivos de contratista general.")
        detalles['riesgo_perfil'] = "Medio: Posible uso de identidades individuales para diluir responsabilidades fiscales."

    # Alerta 4: Concentración Financiera
    if contratos_total > 0 and (monto_total / contratos_total) > 2000000:
        score += 15
        alertas.append("CONCENTRACION_ALTA_VALOR: Contratos escasos pero con montos sobredimensionados.")
        detalles['riesgo_concentracion'] = "Medio: Concentración de fondos en pocas transacciones de alto valor."

    if score >= 70: nivel = "CRÍTICO"
    elif score >= 45: nivel = "ALTO"
    elif score >= 20: nivel = "MEDIO"
    else: nivel = "BAJO"
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)