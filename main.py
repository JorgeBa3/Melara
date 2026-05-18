import os
import re
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
from io import StringIO

app = FastAPI(
    title="Anti-Shell Company Analytics Backend",
    description="API corregida para la detección y análisis de riesgo de empresas de cartón.",
    version="1.4.5"
)

# --- CONFIGURACIÓN DE CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def obtener_frontend():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"error": "Archivo index.html no encontrado."}

def remover_espacios_intercalados(texto: str) -> str:
    """ Remueve los espacios vacíos entre letras causados por codificaciones UTF-16 leídas como UTF-8 """
    if not texto:
        return ""
    # Si detecta espaciado ancho como 'N I T' o 'M O N T O'
    if len(texto) > 1 and texto.count(" ") >= (len(texto) // 2) - 1:
        texto = re.sub(r'\s+', '', texto)
    return texto.strip().upper()

def leer_csv_sucio_sistema(path: str) -> pd.DataFrame:
    """ Abre el archivo eliminando bytes nulos y normalizando texto con espacios intercalados """
    if not os.path.exists(path):
        return pd.DataFrame()
        
    try:
        # Forzar lectura limpia ignorando errores de bytes
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lineas = f.readlines()
            
        lineas_limpias = []
        for linea in lineas:
            # Eliminar bytes nulos comunes en exportaciones de sistemas
            linea_limpia = linea.replace('\x00', '')
            # Si la línea viene segmentada 'F I L A , N I T', juntar las letras de los campos
            if "," in linea_limpia:
                partes = [re.sub(r'(?<!\s) (?!\s)', '', p).strip() for p in linea_limpia.split(",")]
                # Si se detecta el espaciado extremo 'N I T'
                partes = [p.replace(" ", "") if len(p) > 1 and p.count(" ") >= (len(p)//2)-1 else p for p in partes]
                linea_limpia = ",".join(partes) + "\n"
            lineas_limpias.append(linea_limpia)
            
        contenido_final = "".join(lineas_limpias)
        df = pd.read_csv(StringIO(contenido_final), on_bad_lines='skip')
        
        # Limpiar headers de columnas
        df.columns = [re.sub(r'\s+', '', str(col)).upper() for col in df.columns]
        return df
    except Exception as e:
        print(f"Error crítico parseando {path}: {e}")
        return pd.DataFrame()

def load_and_merge_datasets():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    path_2024 = os.path.join(BASE_DIR, "operaciones_registrales_2024.xlsx - Operaciones registrales 2024.csv")
    path_2025 = os.path.join(BASE_DIR, "operaciones_registrales_2025.xlsx - data.csv")
    path_2026 = os.path.join(BASE_DIR, "operaciones_registrales_2026.xlsx - data.csv")
    
    # Se evalúan dinámicamente ambos nombres de reportes financieros por si cambia el timestamp en Git
    path_top = os.path.join(BASE_DIR, "TOP102026_05_17_20_23_19.csv.csv")
    if not os.path.exists(path_top):
        path_top = os.path.join(BASE_DIR, "TOP102026_05_17_20_22_06.csv.csv")

    # 1. Carga adaptativa del bloque de registros anuales
    list_reg = []
    for path_file, anio in [(path_2024, 2024), (path_2025, 2025), (path_2026, 2026)]:
        if os.path.exists(path_file):
            df_temp = leer_csv_sucio_sistema(path_file)
            if not df_temp.empty:
                df_temp['ANIO_REGISTRO'] = anio
                list_reg.append(df_temp)
                
    if list_reg:
        registros = pd.concat(list_reg, ignore_index=True)
        # Asegurar nombres de columnas mapeables
        registros.columns = [str(col).strip().upper() for col in registros.columns]
        c_nit = 'NIT_PROVEEDOR' if 'NIT_PROVEEDOR' in registros.columns else registros.columns[0]
        c_nom = 'NOMBRE_PROVEEDOR' if 'NOMBRE_PROVEEDOR' in registros.columns else registros.columns[2]
        
        registros['NIT_PROVEEDOR_LIMPIO'] = registros[c_nit].astype(str).str.replace(" ", "").str.upper()
        registros['NOMBRE_PROVEEDOR_LIMPIO'] = registros[c_nom].astype(str).str.strip().str.upper()
    else:
        registros = pd.DataFrame(columns=['NIT_PROVEEDOR_LIMPIO', 'NOMBRE_PROVEEDOR_LIMPIO', 'TIPO_PROVEEDOR', 'CAPACIDAD_ECONOMICA', 'ANIO_REGISTRO'])

    # 2. Carga del bloque financiero con normalización de strings espaciados
    top_financial = pd.DataFrame(columns=['NIT', 'NOMBRE', 'CANTIDAD', 'MONTO'])
    df_top = leer_csv_sucio_sistema(path_top)
    
    if not df_top.empty:
        try:
            # Resolver mapeos ante nombres de columnas con problemas de caracteres o vacíos
            col_nit = [c for c in df_top.columns if "NIT" in c][0] if [c for c in df_top.columns if "NIT" in c] else df_top.columns[1]
            col_nom = [c for c in df_top.columns if "NOM" in c][0] if [c for c in df_top.columns if "NOM" in c] else df_top.columns[2]
            col_can = [c for c in df_top.columns if "CAN" in c][0] if [c for c in df_top.columns if "CAN" in c] else df_top.columns[3]
            col_mon = [c for c in df_top.columns if "MON" in c][0] if [c for c in df_top.columns if "MON" in c] else df_top.columns[4]

            top_financial['NIT'] = df_top[col_nit].astype(str).str.replace(" ", "").str.upper()
            top_financial['NOMBRE'] = df_top[col_nom].astype(str).apply(remover_espacios_intercalados)
            top_financial['MONTO'] = pd.to_numeric(df_top[col_mon].astype(str).str.replace(" ", ""), errors='coerce').fillna(0.0)
            top_financial['CANTIDAD'] = pd.to_numeric(df_top[col_can].astype(str).str.replace(" ", ""), errors='coerce').fillna(0).astype(int)
        except Exception as e:
            print(f"Estructura financiera corrupta. Fallo en columnas: {e}")

    return registros, top_financial

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

@app.get("/api/v1/buscar", response_model=List[EmpresaSearchResult])
def buscar_empresa(query: Optional[str] = Query(None)):
    registros, financieras = load_and_merge_datasets()
    resultados_dict = {}

    # Caso A: Búsqueda activa por Query
    if query and query.strip():
        q_limpio = query.strip().upper()
        q_sin_espacios = q_limpio.replace(" ", "")

        if not registros.empty:
            try:
                filtro_reg = registros[
                    (registros['NIT_PROVEEDOR_LIMPIO'].str.contains(q_sin_espacios, na=False, case=False)) |
                    (registros['NOMBRE_PROVEEDOR_LIMPIO'].str.contains(q_limpio, na=False, case=False))
                ]
                for _, row in filtro_reg.drop_duplicates(subset=['NIT_PROVEEDOR_LIMPIO']).iterrows():
                    nit = row['NIT_PROVEEDOR_LIMPIO']
                    if nit and str(nit) != 'NAN' and str(nit).strip() != "":
                        resultados_dict[nit] = {
                            "nit": nit,
                            "nombre": row['NOMBRE_PROVEEDOR_LIMPIO'],
                            "origen_dato": "Registro",
                            "tipo_proveedor": row.get('TIPO_PROVEEDOR', 'No especificado')
                        }
            except Exception:
                pass

        if not financieras.empty:
            try:
                filtro_fin = financieras[
                    (financieras['NIT'].str.contains(q_sin_espacios, na=False, case=False)) |
                    (financieras['NOMBRE'].str.contains(q_limpio, na=False, case=False))
                ]
                for _, row in filtro_fin.drop_duplicates(subset=['NIT']).iterrows():
                    nit = row['NIT']
                    if nit and str(nit) != 'NAN' and str(nit).strip() != "":
                        if nit in resultados_dict:
                            resultados_dict[nit]["origen_dato"] = "Ambos"
                        else:
                            resultados_dict[nit] = {
                                "nit": nit,
                                "nombre": row['NOMBRE'],
                                "origen_dato": "Financiero",
                                "tipo_proveedor": "No especificado"
                            }
            except Exception:
                pass
                
    # Caso B: Catálogo Completo (Query Vacío)
    else:
        if not registros.empty:
            for _, row in registros.drop_duplicates(subset=['NIT_PROVEEDOR_LIMPIO']).iterrows():
                nit = row['NIT_PROVEEDOR_LIMPIO']
                if nit and str(nit) != 'NAN' and str(nit).strip() != "":
                    resultados_dict[nit] = {
                        "nit": nit,
                        "nombre": row['NOMBRE_PROVEEDOR_LIMPIO'],
                        "origen_dato": "Registro",
                        "tipo_proveedor": row.get('TIPO_PROVEEDOR', 'No especificado')
                    }
        if not financieras.empty:
            for _, row in financieras.drop_duplicates(subset=['NIT']).iterrows():
                nit = row['NIT']
                if nit and str(nit) != 'NAN' and str(nit).strip() != "":
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
        # En vez de romper con 404, devolvemos una lista vacía controlada
        return []
        
    return list(resultados_dict.values())


@app.get("/api/v1/analizar/{nit}", response_model=RiskAnalysisResponse)
def analizar_empresa(nit: str):
    nit_buscar = nit.strip().replace(" ", "").upper()
    registros, financieras = load_and_merge_datasets()
    
    df_prov = registros[registros['NIT_PROVEEDOR_LIMPIO'] == nit_buscar] if not registros.empty else pd.DataFrame()
    df_fin = financieras[financieras['NIT'] == nit_buscar] if not financieras.empty else pd.DataFrame()
    
    if df_prov.empty and df_fin.empty:
        raise HTTPException(status_code=404, detail="Proveedor no mapeado.")
    
    nombre = df_fin['NOMBRE'].iloc[0] if not df_fin.empty else (df_prov['NOMBRE_PROVEEDOR_LIMPIO'].iloc[0] if not df_prov.empty else "DESCONOCIDO")
    tipo_prov = df_prov['TIPO_PROVEEDOR'].iloc[0] if not df_prov.empty else "No Registrado"
    capacidad = df_prov['CAPACIDAD_ECONOMICA'].fillna("NO ESPECIFICADA").iloc[0] if not df_prov.empty else "No Registrado"
    
    monto_total = float(df_fin['MONTO'].sum()) if not df_fin.empty else 0.0
    contratos_total = int(df_fin['CANTIDAD'].sum()) if not df_fin.empty else 0
    
    score = 0.0
    alertas = []
    detalles = {}
    anios_activos = df_prov['ANIO_REGISTRO'].unique().tolist() if not df_prov.empty else []
    
    if "COMPRA DIRECTA" in str(capacidad).upper() and monto_total > 1000000:
        score += 40
        alertas.append("CAPACIDAD_ECONOMICA_EXCEDIDA: Perfil de compras directas pero facturación millonaria.")
        detalles['riesgo_capacidad'] = "Crítico: El volumen adjudicado supera su perfil transaccional autorizado."
    
    if 2024 not in anios_activos and (2025 in anios_activos or 2026 in anios_activos) and monto_total > 5000000:
        score += 30
        alertas.append("EMPRESA_EXPRESS: Alta facturación inmediata tras su inscripción reciente.")
        detalles['riesgo_antiguedad'] = "Alto: Alerta de entidad de reciente creación utilizada para adjudicaciones rápidas."

    if contratos_total > 0 and (monto_total / contratos_total) > 2000000:
        score += 15
        alertas.append("CONCENTRACION_ALTA_VALOR: Contratos escasos pero montos sobredimensionados.")
        detalles['riesgo_concentracion'] = "Medio: Concentración inusual de fondos públicos."

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