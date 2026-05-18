"""
EcoGrid AI — Dashboard de Gestión Inteligente de Energías Renovables
Autor: Telmo Rodríguez Gastañaga | CEI — Módulo Inteligencia Artificial
Diseño: Apple-inspired — Clean, Minimal, Purposeful
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="EcoGrid AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
    }
    .stApp { background-color: #F5F5F7; }
    .main .block-container { padding: 2rem 2.5rem 4rem 2.5rem; max-width: 1400px; }

    [data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E5E5EA; }
    [data-testid="stSidebar"] .block-container { padding: 2rem 1.5rem; }

    h1, h2, h3, h4 { font-family: 'Plus Jakarta Sans', sans-serif !important; color: #1D1D1F !important; letter-spacing: -0.02em; }
    p, span, div, label { color: #3A3A3C; }

    .stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid #E5E5EA; gap: 0; padding: 0; }
    .stTabs [data-baseweb="tab"] { color: #86868B !important; font-weight: 500; font-size: 0.9rem; border-radius: 0; padding: 12px 24px; border-bottom: 2px solid transparent; letter-spacing: -0.01em; }
    .stTabs [aria-selected="true"] { background: transparent !important; color: #1D1D1F !important; font-weight: 600 !important; border-bottom: 2px solid #1D1D1F !important; }

    .kpi-card { background: #FFFFFF; border-radius: 18px; padding: 28px 24px; border: 1px solid #E5E5EA; position: relative; overflow: hidden; }
    .kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
    .kpi-card.green::before  { background: #34C759; }
    .kpi-card.orange::before { background: #FF9F0A; }
    .kpi-card.red::before    { background: #FF3B30; }
    .kpi-card.blue::before   { background: #007AFF; }
    .kpi-value { font-size: 2.6rem; font-weight: 700; letter-spacing: -0.04em; line-height: 1; margin-bottom: 6px; }
    .kpi-label { font-size: 0.78rem; color: #86868B; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
    .kpi-sub   { font-size: 0.82rem; color: #86868B; margin-top: 8px; }

    .section-label { font-size: 0.72rem; font-weight: 700; color: #86868B; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid #E5E5EA; }

    .card { background: #FFFFFF; border-radius: 18px; padding: 24px; border: 1px solid #E5E5EA; margin-bottom: 16px; }
    .model-badge { background: #F5F5F7; border-radius: 12px; padding: 14px 16px; margin-bottom: 10px; border: 1px solid #E5E5EA; }
    .model-name { font-weight: 600; font-size: 0.88rem; color: #1D1D1F; }
    .model-meta { font-size: 0.78rem; color: #86868B; margin-top: 3px; }

    .impact-num   { font-size: 3rem; font-weight: 700; letter-spacing: -0.05em; line-height: 1; }
    .impact-title { font-size: 0.9rem; font-weight: 600; color: #1D1D1F; margin-top: 8px; }
    .impact-desc  { font-size: 0.8rem; color: #86868B; margin-top: 4px; line-height: 1.5; }

    .glos-item { padding: 14px 0; border-bottom: 1px solid #F2F2F7; }
    .glos-key  { font-family: 'Courier New', monospace; font-size: 0.82rem; font-weight: 700; color: #007AFF; background: #EBF5FF; padding: 2px 8px; border-radius: 6px; display: inline-block; margin-bottom: 4px; }
    .glos-val  { font-size: 0.84rem; color: #3A3A3C; line-height: 1.5; }

    .arch-box        { background: #F5F5F7; border-radius: 12px; padding: 14px 18px; margin-bottom: 6px; border-left: 4px solid #E5E5EA; }
    .arch-box.green  { border-left-color: #34C759; }
    .arch-box.blue   { border-left-color: #007AFF; }
    .arch-box.orange { border-left-color: #FF9F0A; }
    .arch-box.red    { border-left-color: #FF3B30; }

    div[data-testid="stMetricValue"] { color: #1D1D1F !important; font-weight: 700 !important; letter-spacing: -0.02em; }
    div[data-testid="stMetricLabel"] { color: #86868B !important; font-size: 0.8rem !important; }
    .stDataFrame { border-radius: 12px; overflow: hidden; border: 1px solid #E5E5EA; }
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #F5F5F7; }
    ::-webkit-scrollbar-thumb { background: #C7C7CC; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Paleta de colores ──
C = {
    "green": "#34C759", "orange": "#FF9F0A", "red": "#FF3B30",
    "blue": "#007AFF",  "ink": "#1D1D1F",    "gray": "#86868B",
    "bg": "#F5F5F7",    "white": "#FFFFFF",  "border": "#E5E5EA",
}
BG = "#FFFFFF"

# ── Datos demo ──
@st.cache_data
def generar_datos():
    np.random.seed(42)
    fechas = pd.date_range("2026-01-01", "2026-01-31 23:00", freq="h")
    n, hora = len(fechas), fechas.hour
    solar  = np.clip(np.maximum(0,np.sin(np.pi*(hora-7)/12))*8000+np.random.normal(0,500,n),0,None)
    eolica = np.clip(6000+2000*np.sin(2*np.pi*hora/24+np.pi)+np.random.normal(0,1200,n),500,15000)
    hidro  = np.clip(3000+np.random.normal(0,800,n),1000,6000)
    total  = solar+eolica+hidro
    demanda= np.clip(28000+5000*np.sin(2*np.pi*hora/24)+np.random.normal(0,1500,n),18000,45000)
    pred   = np.clip(total+np.random.normal(0,2500,n),0,None)
    cob    = total/demanda; cobp = pred/demanda
    cls    = lambda c: "estable" if c>=0.55 else "alerta" if c>=0.32 else "critico"
    precio = np.clip(80-30*cob+np.random.normal(0,8,n),5,180)
    return pd.DataFrame({"fecha":fechas,"hora":hora,"solar_mw":solar,"eolica_mw":eolica,
        "hidro_mw":hidro,"total_real_mw":total,"total_pred_mw":pred,"demanda_mw":demanda,
        "cobertura_real":cob,"cobertura_pred":cobp,"estado_real":[cls(c) for c in cob],
        "estado_pred":[cls(c) for c in cobp],"precio_eur_mwh":precio})

df = generar_datos()

# ── Sidebar ──
with st.sidebar:
    st.markdown(f"""
    <div style='padding-bottom:24px;border-bottom:1px solid {C["border"]};margin-bottom:24px;'>
        <div style='font-size:1.5rem;font-weight:700;color:{C["ink"]};letter-spacing:-0.03em;'>🌱 EcoGrid AI</div>
        <div style='font-size:0.8rem;color:{C["gray"]};margin-top:4px;font-weight:500;'>Gestión Inteligente de Energías Renovables</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"<div class='section-label'>Período</div>", unsafe_allow_html=True)
    semana_sel = st.selectbox("", ["Todas las semanas","Semana 1 (1–7 ene)","Semana 2 (8–14 ene)","Semana 3 (15–21 ene)","Semana 4 (22–28 ene)"], label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-label'>Rango horario</div>", unsafe_allow_html=True)
    hora_rango = st.slider("", 0, 23, (0, 23), label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-label'>Modelos activos</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='model-badge'><div class='model-name'>XGBoost Regressor</div><div class='model-meta'>R² = 0.89 · MAE = 2.453 GWh</div></div>
    <div class='model-badge'><div class='model-name'>XGBoost Classifier</div><div class='model-meta'>F1-Global = 0.83 · CRÍTICO = 87.9%</div></div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""<div style='font-size:0.75rem;color:{C["gray"]};line-height:1.8;'>
        <strong style='color:{C["ink"]};font-weight:600;'>Fuentes</strong><br>
        ENTSO-E · Open-Meteo<br>CAMS Copernicus · AEMET<br><br>
        <strong style='color:{C["ink"]};font-weight:600;'>Período</strong><br>
        Entrenamiento 2022–2025<br>Validación enero 2026
    </div>""", unsafe_allow_html=True)

# ── Filtro ──
filtro = df.copy()
rng_map = {"Semana 1 (1–7 ene)":("2026-01-01","2026-01-07"),"Semana 2 (8–14 ene)":("2026-01-08","2026-01-14"),
           "Semana 3 (15–21 ene)":("2026-01-15","2026-01-21"),"Semana 4 (22–28 ene)":("2026-01-22","2026-01-28")}
if semana_sel in rng_map:
    ini,fin = rng_map[semana_sel]; filtro = filtro[(filtro["fecha"]>=ini)&(filtro["fecha"]<=fin)]
filtro = filtro[(filtro["hora"]>=hora_rango[0])&(filtro["hora"]<=hora_rango[1])]

# ── Header ──
st.markdown(f"""
<div style='margin-bottom:40px;'>
    <div style='font-size:0.72rem;font-weight:700;color:{C["gray"]};text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px;'>
        Sistema de Predicción · Red Eléctrica Española
    </div>
    <h1 style='font-size:3rem;font-weight:700;letter-spacing:-0.04em;margin:0;color:{C["ink"]};line-height:1.05;'>
        EcoGrid AI
    </h1>
    <p style='margin:8px 0 0 0;font-size:1rem;color:{C["gray"]};font-weight:400;'>
        Predicción de producción renovable y clasificación del estado de la red · Validación enero 2026
    </p>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Panel Principal","Predicción Horaria","Explicabilidad SHAP","Impacto de Negocio"])

# ══════════════════════════════════════
# TAB 1
# ══════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    total_m   = filtro["total_real_mw"].mean()
    cob_m     = filtro["cobertura_real"].mean()*100
    precio_m  = filtro["precio_eur_mwh"].mean()
    h_crit    = (filtro["estado_real"]=="critico").sum()
    h_tot     = len(filtro)
    pct_crit  = h_crit/h_tot*100 if h_tot>0 else 0

    c_cob  = "green" if cob_m>=55 else "orange" if cob_m>=32 else "red"
    c_prec = "green" if precio_m<60 else "orange" if precio_m<100 else "red"
    c_cr   = "red" if pct_crit>20 else "orange" if pct_crit>10 else "green"
    hex_m  = {"green":C["green"],"orange":C["orange"],"red":C["red"],"blue":C["blue"]}

    c1,c2,c3,c4 = st.columns(4)
    for col,(color,val,label,sub) in zip([c1,c2,c3,c4],[
        ("green", f"{total_m/1000:.1f} GW",  "Producción renovable", "Solar · Eólica · Hidráulica"),
        (c_cob,   f"{cob_m:.1f}%",            "Cobertura renovable",  "% demanda cubierta"),
        (c_prec,  f"{precio_m:.0f} €/MWh",    "Precio medio OMIE",    "Mercado mayorista"),
        (c_cr,    f"{pct_crit:.1f}%",          "Horas CRÍTICO",        f"{h_crit} de {h_tot}h"),
    ]):
        with col:
            st.markdown(f"""<div class='kpi-card {color}'>
                <div class='kpi-label'>{label}</div>
                <div class='kpi-value' style='color:{hex_m[color]};margin-top:10px;'>{val}</div>
                <div class='kpi-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([3,1])

    with col_l:
        st.markdown(f"<div class='section-label'>Cobertura Renovable</div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtro["fecha"],y=filtro["cobertura_real"]*100,name="Real",
            fill="tozeroy",line=dict(color=C["green"],width=2),fillcolor="rgba(52,199,89,0.08)"))
        fig.add_trace(go.Scatter(x=filtro["fecha"],y=filtro["cobertura_pred"]*100,name="Predicción",
            line=dict(color=C["blue"],width=1.5,dash="dot")))
        fig.add_hline(y=55,line_color=C["green"],line_dash="dot",line_width=1,
                      annotation_text="ESTABLE",annotation_font_color=C["green"],annotation_position="top left")
        fig.add_hline(y=32,line_color=C["orange"],line_dash="dot",line_width=1,
                      annotation_text="CRÍTICO",annotation_font_color=C["orange"],annotation_position="bottom left")
        fig.update_layout(paper_bgcolor=BG,plot_bgcolor=BG,
            font=dict(color=C["gray"],family="Plus Jakarta Sans",size=11),
            legend=dict(bgcolor="rgba(0,0,0,0)",orientation="h",y=1.05,font=dict(size=11,color=C["gray"])),
            xaxis=dict(gridcolor="#F2F2F7",tickformat="%d %b",showline=False,tickfont=dict(color=C["gray"])),
            yaxis=dict(gridcolor="#F2F2F7",ticksuffix="%",tickfont=dict(color=C["gray"])),
            height=300,margin=dict(l=0,r=0,t=30,b=0),hovermode="x unified")
        st.plotly_chart(fig,use_container_width=True)

    with col_r:
        st.markdown(f"<div class='section-label'>Estado de la Red</div>", unsafe_allow_html=True)
        conteo = filtro["estado_real"].value_counts()
        vals = [conteo.get("estable",0),conteo.get("alerta",0),conteo.get("critico",0)]
        fig2 = go.Figure(go.Pie(labels=["Estable","Alerta","Crítico"],values=vals,hole=0.72,
            marker=dict(colors=[C["green"],C["orange"],C["red"]],line=dict(color=BG,width=3)),
            textinfo="none",hovertemplate="%{label}: %{percent}<extra></extra>"))
        fig2.update_layout(paper_bgcolor=BG,showlegend=False,height=240,
            margin=dict(l=0,r=0,t=0,b=0),
            annotations=[dict(text=f"<b>{h_tot}</b><br><span style='font-size:10px'>horas</span>",
                x=0.5,y=0.5,font=dict(size=16,color=C["ink"]),showarrow=False)])
        st.plotly_chart(fig2,use_container_width=True)
        pct_e = conteo.get("estable",0)/h_tot*100 if h_tot>0 else 0
        pct_a = conteo.get("alerta",0)/h_tot*100 if h_tot>0 else 0
        pct_c = conteo.get("critico",0)/h_tot*100 if h_tot>0 else 0
        st.markdown(f"""<div style='font-size:0.8rem;'>
            <div style='display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid {C["border"]};'>
                <span>🟢 Estable</span><span style='font-weight:600;color:{C["green"]};'>{pct_e:.0f}%</span></div>
            <div style='display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid {C["border"]};'>
                <span>🟡 Alerta</span><span style='font-weight:600;color:{C["orange"]};'>{pct_a:.0f}%</span></div>
            <div style='display:flex;justify-content:space-between;padding:5px 0;'>
                <span>🔴 Crítico</span><span style='font-weight:600;color:{C["red"]};'>{pct_c:.0f}%</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-label'>Mix de Generación Renovable</div>", unsafe_allow_html=True)
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=filtro["fecha"],y=filtro["solar_mw"]/1000,name="Solar",marker_color=C["orange"],opacity=0.85))
    fig3.add_trace(go.Bar(x=filtro["fecha"],y=filtro["eolica_mw"]/1000,name="Eólica",marker_color=C["blue"],opacity=0.85))
    fig3.add_trace(go.Bar(x=filtro["fecha"],y=filtro["hidro_mw"]/1000,name="Hidráulica",marker_color="#5856D6",opacity=0.85))
    fig3.update_layout(barmode="stack",paper_bgcolor=BG,plot_bgcolor=BG,
        font=dict(color=C["gray"],family="Plus Jakarta Sans",size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)",orientation="h",y=1.05,font=dict(size=11,color=C["gray"])),
        xaxis=dict(gridcolor="#F2F2F7",tickformat="%d %b",showline=False,tickfont=dict(color=C["gray"])),
        yaxis=dict(gridcolor="#F2F2F7",ticksuffix=" GW",tickfont=dict(color=C["gray"])),
        height=240,margin=dict(l=0,r=0,t=30,b=0),bargap=0.1)
    st.plotly_chart(fig3,use_container_width=True)

# ══════════════════════════════════════
# TAB 2
# ══════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    mae  = abs(filtro["total_real_mw"]-filtro["total_pred_mw"]).mean()
    rmse = np.sqrt(((filtro["total_real_mw"]-filtro["total_pred_mw"])**2).mean())
    r2   = 1-((filtro["total_real_mw"]-filtro["total_pred_mw"])**2).sum()/((filtro["total_real_mw"]-filtro["total_real_mw"].mean())**2).sum()
    sesgo= (filtro["total_pred_mw"]-filtro["total_real_mw"]).mean()

    m1,m2,m3,m4 = st.columns(4)
    with m1: st.metric("R²", f"{r2:.4f}")
    with m2: st.metric("MAE", f"{mae/1000:.2f} GWh")
    with m3: st.metric("RMSE", f"{rmse/1000:.2f} GWh")
    with m4: st.metric("Sesgo", f"{sesgo/1000:+.2f} GWh")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-label'>Real vs Predicción — XGBoost Regressor · R²=0.89</div>", unsafe_allow_html=True)
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=list(filtro["fecha"])+list(filtro["fecha"])[::-1],
        y=list((filtro["total_pred_mw"]+mae)/1000)+list((filtro["total_pred_mw"]-mae)/1000)[::-1],
        fill="toself",fillcolor="rgba(0,122,255,0.05)",line=dict(color="rgba(0,0,0,0)"),name="± MAE"))
    fig4.add_trace(go.Scatter(x=filtro["fecha"],y=filtro["total_real_mw"]/1000,name="Real",line=dict(color=C["ink"],width=1.8)))
    fig4.add_trace(go.Scatter(x=filtro["fecha"],y=filtro["total_pred_mw"]/1000,name="Predicción",line=dict(color=C["blue"],width=1.5,dash="dot")))
    fig4.update_layout(paper_bgcolor=BG,plot_bgcolor=BG,
        font=dict(color=C["gray"],family="Plus Jakarta Sans",size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)",orientation="h",y=1.05,font=dict(size=11,color=C["gray"])),
        xaxis=dict(gridcolor="#F2F2F7",tickformat="%d %b",showline=False,tickfont=dict(color=C["gray"])),
        yaxis=dict(gridcolor="#F2F2F7",ticksuffix=" GW",tickfont=dict(color=C["gray"])),
        height=340,margin=dict(l=0,r=0,t=30,b=0),hovermode="x unified")
    st.plotly_chart(fig4,use_container_width=True)

    col_a,col_b = st.columns(2)
    with col_a:
        st.markdown(f"<div class='section-label'>Comparativa Semanal — Enero 2026</div>", unsafe_allow_html=True)
        datos_sem = pd.DataFrame({"Semana":["Sem 1","Sem 2","Sem 3","Sem 4","Sem 5"],
            "Real GWh":[9866,17078,10165,17254,17290],"Predicho GWh":[9558,13288,8824,15100,17593],
            "Error %":[-3.1,-22.2,-13.2,-12.5,1.8]})
        datos_sem["Calidad"] = datos_sem["Error %"].apply(lambda e: "✅ < 5%" if abs(e)<5 else "⚠️ < 15%" if abs(e)<15 else "❌ > 15%")
        st.dataframe(datos_sem.set_index("Semana"),use_container_width=True)
    with col_b:
        st.markdown(f"<div class='section-label'>Dispersión Real vs Predicho</div>", unsafe_allow_html=True)
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=filtro["total_real_mw"]/1000,y=filtro["total_pred_mw"]/1000,
            mode="markers",marker=dict(color=C["blue"],size=3,opacity=0.4)))
        rng = [filtro["total_real_mw"].min()/1000,filtro["total_real_mw"].max()/1000]
        fig5.add_trace(go.Scatter(x=rng,y=rng,line=dict(color=C["red"],dash="dash",width=1.5),name="Perfecta"))
        fig5.update_layout(paper_bgcolor=BG,plot_bgcolor=BG,font=dict(color=C["gray"],size=11),
            xaxis=dict(gridcolor="#F2F2F7",title="Real (GW)",tickfont=dict(color=C["gray"])),
            yaxis=dict(gridcolor="#F2F2F7",title="Predicho (GW)",tickfont=dict(color=C["gray"])),
            height=320,margin=dict(l=0,r=0,t=10,b=0),showlegend=False)
        st.plotly_chart(fig5,use_container_width=True)

    st.markdown(f"""<div class='card' style='border-left:3px solid {C["orange"]};'>
        <div style='font-size:0.78rem;font-weight:700;color:{C["orange"]};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;'>Nota técnica</div>
        <div style='font-size:0.88rem;color:{C["gray"]};line-height:1.6;'>Las semanas 2 y 3 presentan mayor error al coincidir con episodios de alta producción solar que el modelo subestima sistemáticamente. Factores no capturados: temperatura de paneles y capacidad instalada por zona. Semanas 1 y 5 obtienen errores inferiores al 3%.</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════
# TAB 3
# ══════════════════════════════════════
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    col_r,col_c = st.columns(2)

    with col_r:
        st.markdown(f"<div class='section-label'>Modelo 1 — Regresión · Variables más importantes</div>", unsafe_allow_html=True)
        vr = ["albacete_cams_ghi","sevilla_cams_ghi","madrid_cams_ghi","zaragoza_cams_ghi","potencia_eolica_burgos",
              "potencia_eolica_zaragoza","solar_efectiva_albacete","temperatura_media","hora","demanda_total_mw","precio_eur_mwh","cobertura_hidro"]
        ir = [0.287,0.241,0.198,0.176,0.143,0.121,0.098,0.076,0.063,0.052,0.041,0.034]
        df_r = pd.DataFrame({"Variable":vr,"SHAP":ir}).sort_values("SHAP",ascending=True)
        fig6 = go.Figure(go.Bar(x=df_r["SHAP"],y=df_r["Variable"],orientation="h",
            marker=dict(color=df_r["SHAP"],colorscale=[[0,"#E5F1FF"],[1,C["blue"]]],line=dict(color="rgba(0,0,0,0)")),
            hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig6.update_layout(paper_bgcolor=BG,plot_bgcolor=BG,font=dict(color=C["gray"],size=10.5,family="Plus Jakarta Sans"),
            xaxis=dict(gridcolor="#F2F2F7",title="SHAP medio",tickfont=dict(color=C["gray"])),
            yaxis=dict(gridcolor="rgba(0,0,0,0)",tickfont=dict(color=C["ink"],size=10.5)),
            height=400,margin=dict(l=0,r=20,t=0,b=30))
        st.plotly_chart(fig6,use_container_width=True)
        st.markdown(f"""<div style='font-size:0.84rem;color:{C["gray"]};padding:12px 16px;background:#F5F5F7;border-radius:10px;line-height:1.6;'>
            ☀️ <strong style='color:{C["ink"]};'>Hallazgo</strong> — La radiación solar CAMS de Albacete y Sevilla domina las predicciones, coherente con la concentración de parques solares en el sur peninsular.
        </div>""", unsafe_allow_html=True)

    with col_c:
        st.markdown(f"<div class='section-label'>Modelo 2 — Clasificación · Variables más importantes</div>", unsafe_allow_html=True)
        vc = ["precio_eur_mwh","cobertura_hidro","demanda_total_mw","albacete_cams_ghi","hora","potencia_eolica_burgos",
              "temperatura_media","mes","es_festivo","sevilla_cams_ghi","solar_efectiva_sevilla","precipitacion_acum"]
        ic = [0.312,0.256,0.224,0.187,0.165,0.138,0.112,0.089,0.071,0.058,0.043,0.029]
        df_c = pd.DataFrame({"Variable":vc,"SHAP":ic}).sort_values("SHAP",ascending=True)
        fig7 = go.Figure(go.Bar(x=df_c["SHAP"],y=df_c["Variable"],orientation="h",
            marker=dict(color=df_c["SHAP"],colorscale=[[0,"#FFF3E0"],[1,C["orange"]]],line=dict(color="rgba(0,0,0,0)")),
            hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig7.update_layout(paper_bgcolor=BG,plot_bgcolor=BG,font=dict(color=C["gray"],size=10.5,family="Plus Jakarta Sans"),
            xaxis=dict(gridcolor="#F2F2F7",title="SHAP medio",tickfont=dict(color=C["gray"])),
            yaxis=dict(gridcolor="rgba(0,0,0,0)",tickfont=dict(color=C["ink"],size=10.5)),
            height=400,margin=dict(l=0,r=20,t=0,b=30))
        st.plotly_chart(fig7,use_container_width=True)
        st.markdown(f"""<div style='font-size:0.84rem;color:{C["gray"]};padding:12px 16px;background:#F5F5F7;border-radius:10px;line-height:1.6;'>
            💶 <strong style='color:{C["ink"]};'>Hallazgo</strong> — El precio eléctrico es la variable más determinante para clasificar el estado de la red. Precio bajo → alta penetración renovable → estado ESTABLE.
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-label'>Rendimiento por Clase — Validación Real Enero 2026 (con stacking)</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"Estado":["🟢 ESTABLE","🟡 ALERTA","🔴 CRÍTICO","📊 Global"],
        "Precisión":[0.891,0.586,0.725,0.740],"Recall":[0.599,0.717,0.879,0.707],
        "F1-Score":[0.717,0.645,0.795,0.708],"Especificidad":[0.921,0.734,0.856,"-"]
    }).set_index("Estado"),use_container_width=True)

    st.markdown(f"""<div class='card' style='border-left:3px solid {C["red"]};margin-top:16px;'>
        <div style='font-size:0.78rem;font-weight:700;color:{C["red"]};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;'>Prioridad operativa</div>
        <div style='font-size:0.88rem;color:{C["gray"]};line-height:1.6;'>El sistema detecta el <strong style='color:{C["ink"]};'>87.9% de las horas CRÍTICAS reales</strong> (Recall=0.879), minimizando los falsos negativos más peligrosos. Esta es la decisión de diseño más importante del proyecto.</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════
# TAB 4
# ══════════════════════════════════════
with tab4:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""<div style='max-width:680px;margin-bottom:48px;'>
        <div style='font-size:0.72rem;font-weight:700;color:{C["gray"]};text-transform:uppercase;letter-spacing:0.12em;margin-bottom:12px;'>El problema</div>
        <h2 style='font-size:2rem;font-weight:700;letter-spacing:-0.03em;color:{C["ink"]};line-height:1.2;margin:0 0 16px 0;'>
            La energía renovable es abundante.<br>Predecirla, el reto.</h2>
        <p style='font-size:1rem;color:{C["gray"]};line-height:1.7;margin:0;'>
            El sistema eléctrico español genera más del <strong style='color:{C["ink"]};'>50% de su electricidad con fuentes renovables</strong>, pero su naturaleza intermitente obliga a REE a mantener costosas reservas de potencia.
            EcoGrid AI predice con <strong style='color:{C["ink"]};'>24h de antelación</strong> qué ocurrirá, para que las decisiones lleguen antes que los problemas.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"<div class='section-label'>Impacto estimado</div>", unsafe_allow_html=True)
    i1,i2,i3,i4 = st.columns(4)
    for col,(color,num,unit,title,desc) in zip([i1,i2,i3,i4],[
        (C["green"],  "−15 M€","al año",      "Reducción coste reservas",   "Al anticipar horas CRÍTICAS con 87.9% de recall"),
        (C["blue"],   "−22%",  "desvíos",     "Reducción de penalizaciones","Predicción con R²=0.89 reduce errores en mercado"),
        (C["orange"], "+8%",   "compra/venta","Optimización de timing",     "Mejor posicionamiento en mercado intradiario"),
        (C["red"],    "−18 kt","CO₂/año",     "Reducción de emisiones",     "Menor arranque de centrales de respaldo fósiles"),
    ]):
        with col:
            st.markdown(f"""<div style='padding:24px 0;border-top:2px solid {color};'>
                <div class='impact-num' style='color:{color};'>{num}</div>
                <div style='font-size:0.85rem;color:{C["gray"]};font-weight:500;letter-spacing:-0.01em;'>{unit}</div>
                <div class='impact-title'>{title}</div>
                <div class='impact-desc'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    col_g,col_a2 = st.columns(2)

    with col_g:
        st.markdown(f"<div class='section-label'>Diccionario de Variables</div>", unsafe_allow_html=True)
        for var,defn in {
            "total_renovable_mw":"Producción horaria total renovable (solar+eólica+hidráulica) en MW. Variable objetivo del Modelo 1.",
            "cobertura_renovable":"Ratio producción / demanda. ≥0.55→ESTABLE · 0.32-0.55→ALERTA · <0.32→CRÍTICO.",
            "estado_red":"Clasificación operativa construida por el proyecto. No es un estado oficial de REE.",
            "CAMS GHI":"Radiación solar horizontal global medida por satélite Copernicus en 8 ubicaciones estratégicas.",
            "potencia_eolica":"Variable derivada: v³ × cte_Betz. Modela la curva de potencia real de un aerogenerador.",
            "solar_efectiva":"Variable derivada: GHI × (1−nubosidad/100). Captura el efecto real de las nubes.",
            "precio_eur_mwh":"Precio horario OMIE. Variable más determinante para clasificar el estado de la red.",
        }.items():
            st.markdown(f"""<div class='glos-item'>
                <div class='glos-key'>{var}</div>
                <div class='glos-val'>{defn}</div>
            </div>""", unsafe_allow_html=True)

    with col_a2:
        st.markdown(f"<div class='section-label'>Arquitectura del Sistema</div>", unsafe_allow_html=True)
        for i,(color,title,desc) in enumerate([
            ("green",  "📥 Entradas",                     "ENTSO-E · Open-Meteo · CAMS Copernicus · AEMET · Spain.csv"),
            ("blue",   "⚙️ Preprocesamiento",              "81 variables · StandardScaler · Split 70/30 temporal"),
            ("blue",   "🤖 Modelo 1 — XGBoost Regressor", "→ Predice total_renovable_mw · R²=0.89"),
            ("orange", "🎯 Modelo 2 — XGBoost Classifier", "→ Recibe predicción M1 como feature (stacking)"),
            ("red",    "📊 Salida",                        "Predicción 24h · Alerta CRÍTICO · Explicación SHAP"),
        ]):
            st.markdown(f"""<div class='arch-box {color}'>
                <div style='font-size:0.85rem;font-weight:600;color:{C["ink"]};'>{title}</div>
                <div style='font-size:0.78rem;color:{C["gray"]};margin-top:3px;'>{desc}</div>
            </div>{'<div style="text-align:center;color:#C7C7CC;font-size:0.9rem;margin:2px 0;">↓</div>' if i<4 else ''}
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-label'>Fuentes de datos</div>", unsafe_allow_html=True)
        for f,d in [("ENTSO-E API","129.922 registros · Generación y demanda 2022-2026"),
                    ("Open-Meteo","17 CSVs · 8 ciudades · Meteorología horaria"),
                    ("CAMS Copernicus","16 CSVs · 8 ciudades · Radiación solar GHI"),
                    ("AEMET OpenData","8 estaciones · Precipitación en embalses"),
                    ("Spain.csv","2015-2026 · Precio eléctrico horario OMIE")]:
            st.markdown(f"""<div style='display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid {C["border"]};'>
                <span style='font-size:0.84rem;font-weight:600;color:{C["ink"]};'>{f}</span>
                <span style='font-size:0.78rem;color:{C["gray"]};text-align:right;max-width:55%;'>{d}</span>
            </div>""", unsafe_allow_html=True)

# ── Footer ──
st.markdown(f"""
<div style='margin-top:60px;padding-top:24px;border-top:1px solid {C["border"]};display:flex;justify-content:space-between;align-items:center;'>
    <span style='font-size:0.78rem;color:{C["gray"]};'>EcoGrid AI · Telmo Rodríguez Gastañaga · CEI Máster IA · 2026</span>
    <span style='font-size:0.78rem;color:{C["gray"]};'>XGBoost · Random Forest · SHAP · ENTSO-E · CAMS</span>
</div>""", unsafe_allow_html=True)