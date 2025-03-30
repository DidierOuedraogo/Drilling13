import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import io

# Configuration de la page
st.set_page_config(
    page_title="Mining Geology Data Application",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E4053;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #3498DB;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2E4053;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .author {
        font-size: 1rem;
        color: #566573;
        font-style: italic;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F8F9F9;
        padding: 10px 15px;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498DB !important;
        color: white !important;
    }
    .stButton>button {
        background-color: #3498DB;
        color: white;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #2874A6;
        color: white;
    }
    .uploadedFile {
        border: 1px solid #3498DB;
        border-radius: 5px;
        padding: 10px;
    }
    .success-message {
        background-color: #D4EFDF;
        border-left: 5px solid #2ECC71;
        padding: 10px;
        border-radius: 0px 5px 5px 0px;
    }
    .warning-message {
        background-color: #FCF3CF;
        border-left: 5px solid #F1C40F;
        padding: 10px;
        border-radius: 0px 5px 5px 0px;
    }
    .error-message {
        background-color: #FADBD8;
        border-left: 5px solid #E74C3C;
        padding: 10px;
        border-radius: 0px 5px 5px 0px;
    }
    .info-card {
        background-color: #EBF5FB;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #F8F9F9;
        border-left: 4px solid #3498DB;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    .sidebar .sidebar-content {
        background-color: #F8F9F9;
    }
</style>
""", unsafe_allow_html=True)

# Titre de l'application et auteur
st.markdown('<h1 class="main-header">Mining Geology Data Application</h1>', unsafe_allow_html=True)
st.markdown('<p class="author">Développé par: Didier Ouedraogo, P.Geo.</p>', unsafe_allow_html=True)

# Fonction pour convertir les chaînes en nombres flottants avec gestion d'erreurs
def safe_float(value):
    if pd.isna(value):
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

# Fonction pour télécharger les données en CSV
def get_csv_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="color: #3498DB; text-decoration: none;"><button style="background-color: #3498DB; color: white; padding: 8px 15px; border: none; border-radius: 5px; cursor: pointer;">Télécharger {text}</button></a>'
    return href

# Fonction pour vérifier si une colonne existe dans un DataFrame
def column_exists(df, column_name):
    return df is not None and column_name and column_name in df.columns

# Fonction pour créer des composites d'analyses avec coordonnées
def create_composites(assays_df, hole_id_col, from_col, to_col, value_col, composite_length=1.0, 
                     collars_df=None, survey_df=None, x_col=None, y_col=None, z_col=None, 
                     azimuth_col=None, dip_col=None, depth_col=None):
    if assays_df is None or assays_df.empty:
        return None
    
    # Vérifier que toutes les colonnes nécessaires existent
    if not all(col in assays_df.columns for col in [hole_id_col, from_col, to_col, value_col]):
        st.markdown('<div class="error-message">Colonnes manquantes dans le DataFrame des analyses</div>', unsafe_allow_html=True)
        return None
    
    # Créer une copie des données pour éviter de modifier l'original
    df = assays_df.copy()
    
    # Convertir les colonnes numériques en flottants
    for col in [from_col, to_col, value_col]:
        df[col] = df[col].apply(safe_float)
    
    # Initialiser le DataFrame des composites
    composites = []
    
    # Pour chaque trou de forage
    for hole_id in df[hole_id_col].unique():
        hole_data = df[df[hole_id_col] == hole_id].sort_values(by=from_col)
        
        if hole_data.empty:
            continue

        # Récupérer les données de collars et survey pour les coordonnées si disponibles
        collar_info = None
        surveys = None
        
        if collars_df is not None and survey_df is not None and all(col is not None for col in [x_col, y_col, z_col, depth_col, azimuth_col, dip_col]):
            if hole_id in collars_df[hole_id_col].values:
                collar_info = collars_df[collars_df[hole_id_col] == hole_id].iloc[0]
                surveys = survey_df[survey_df[hole_id_col] == hole_id].sort_values(by=depth_col)
        
        # Pour chaque intervalle de composite
        composite_start = float(hole_data[from_col].min())
        while composite_start < float(hole_data[to_col].max()):
            composite_end = composite_start + composite_length
            
            # Trouver tous les intervalles qui chevauchent le composite actuel
            overlapping = hole_data[
                ((hole_data[from_col] >= composite_start) & (hole_data[from_col] < composite_end)) |
                ((hole_data[to_col] > composite_start) & (hole_data[to_col] <= composite_end)) |
                ((hole_data[from_col] <= composite_start) & (hole_data[to_col] >= composite_end))
            ]
            
            if not overlapping.empty:
                # Calculer le poids pondéré pour chaque intervalle chevauchant
                weighted_values = []
                total_length = 0
                
                for _, row in overlapping.iterrows():
                    overlap_start = max(composite_start, row[from_col])
                    overlap_end = min(composite_end, row[to_col])
                    overlap_length = overlap_end - overlap_start
                    
                    if overlap_length > 0:
                        weighted_values.append(row[value_col] * overlap_length)
                        total_length += overlap_length
                
                # Calculer la valeur pondérée du composite
                if total_length > 0:
                    composite_value = sum(weighted_values) / total_length
                    
                    # Créer une entrée de composite de base
                    composite_entry = {
                        hole_id_col: hole_id,
                        'From': composite_start,
                        'To': composite_end,
                        'Length': total_length,
                        value_col: composite_value
                    }
                    
                    # Ajouter les coordonnées si les données nécessaires sont disponibles
                    if collar_info is not None and not surveys.empty:
                        # Calculer la position moyenne (milieu de l'intervalle)
                        mid_depth = (composite_start + composite_end) / 2
                        
                        # Chercher les données de survey les plus proches
                        closest_idx = surveys[depth_col].apply(lambda d: abs(d - mid_depth)).idxmin()
                        closest_survey = surveys.loc[closest_idx]
                        
                        # Récupérer les données du collar
                        x_start = safe_float(collar_info[x_col])
                        y_start = safe_float(collar_info[y_col])
                        z_start = safe_float(collar_info[z_col])
                        
                        # Calculer les coordonnées 3D approximatives pour le composite
                        # (Méthode simplifiée - pour une précision parfaite, une interpolation plus complexe serait nécessaire)
                        depth = safe_float(closest_survey[depth_col])
                        azimuth = safe_float(closest_survey[azimuth_col])
                        dip = safe_float(closest_survey[dip_col])
                        
                        # Convertir l'azimuth et le dip en direction 3D
                        azimuth_rad = np.radians(azimuth)
                        dip_rad = np.radians(dip)
                        
                        # Calculer la position approximative
                        dx = depth * np.sin(dip_rad) * np.sin(azimuth_rad)
                        dy = depth * np.sin(dip_rad) * np.cos(azimuth_rad)
                        dz = -depth * np.cos(dip_rad)  # Z est négatif pour la profondeur
                        
                        # Ajouter les coordonnées au composite
                        composite_entry['X'] = x_start + dx
                        composite_entry['Y'] = y_start + dy
                        composite_entry['Z'] = z_start + dz
                    
                    # Ajouter le composite au résultat
                    composites.append(composite_entry)
            
            composite_start = composite_end
    
    # Créer un DataFrame à partir des composites
    if composites:
        return pd.DataFrame(composites)
    else:
        return pd.DataFrame()

# Fonction pour créer un strip log pour un forage spécifique
def create_strip_log(hole_id, collars_df, survey_df, lithology_df, assays_df, 
                    hole_id_col, depth_col, 
                    lith_from_col, lith_to_col, lith_col,
                    assay_from_col, assay_to_col, assay_value_col):
    
    # Vérifier si les données nécessaires sont disponibles
    if collars_df is None or survey_df is None:
        return None
    
    # Récupérer les informations du forage
    hole_surveys = survey_df[survey_df[hole_id_col] == hole_id].sort_values(by=depth_col)
    
    if hole_surveys.empty:
        return None
    
    # Convertir les valeurs de profondeur en flottants
    hole_surveys[depth_col] = hole_surveys[depth_col].apply(safe_float)
    
    # Profondeur maximale du forage
    max_depth = hole_surveys[depth_col].max()
    
    # Créer la figure
    fig, axes = plt.subplots(1, 3, figsize=(12, max_depth/10 + 2), 
                            gridspec_kw={'width_ratios': [2, 1, 3]})
    
    # Titre du graphique
    fig.suptitle(f'Strip Log - Forage {hole_id}', fontsize=16)
    
    # 1. Colonne de lithologie
    if lithology_df is not None and all(col and col in lithology_df.columns for col in [hole_id_col, lith_from_col, lith_to_col, lith_col]):
        hole_litho = lithology_df[lithology_df[hole_id_col] == hole_id].sort_values(by=lith_from_col)
        
        if not hole_litho.empty:
            # Convertir les colonnes de profondeur en flottants
            hole_litho[lith_from_col] = hole_litho[lith_from_col].apply(safe_float)
            hole_litho[lith_to_col] = hole_litho[lith_to_col].apply(safe_float)
            
            # Définir une palette de couleurs pour les différentes lithologies
            unique_liths = hole_litho[lith_col].unique()
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_liths)))
            lith_color_map = {lith: color for lith, color in zip(unique_liths, colors)}
            
            # Dessiner des rectangles pour chaque intervalle de lithologie
            for _, row in hole_litho.iterrows():
                lith_from = row[lith_from_col]
                lith_to = row[lith_to_col]
                lith_type = row[lith_col]
                
                axes[0].add_patch(plt.Rectangle((0, lith_from), 1, lith_to - lith_from, 
                                                color=lith_color_map[lith_type]))
                
                # Ajouter le texte de la lithologie au milieu de l'intervalle
                interval_height = lith_to - lith_from
                font_size = min(10, max(6, interval_height * 0.8))  # Taille de police adaptative
                
                axes[0].text(0.5, (lith_from + lith_to) / 2, lith_type,
                            ha='center', va='center', fontsize=font_size,
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Configurer l'axe de la lithologie
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(max_depth, 0)  # Inverser l'axe y pour que la profondeur augmente vers le bas
    axes[0].set_xlabel('Lithologie')
    axes[0].set_ylabel('Profondeur (m)')
    axes[0].set_xticks([])
    
    # 2. Colonne de profondeur
    depth_ticks = np.arange(0, max_depth + 10, 10)
    axes[1].set_yticks(depth_ticks)
    axes[1].set_ylim(max_depth, 0)
    axes[1].set_xlim(0, 1)
    axes[1].set_xticks([])
    axes[1].set_xlabel('Profondeur')
    axes[1].grid(axis='y')
    
    # 3. Colonne d'analyses
    if assays_df is not None and all(col and col in assays_df.columns for col in [hole_id_col, assay_from_col, assay_to_col, assay_value_col]):
        hole_assays = assays_df[assays_df[hole_id_col] == hole_id].sort_values(by=assay_from_col)
        
        if not hole_assays.empty:
            # Convertir les colonnes numériques en flottants
            hole_assays[assay_from_col] = hole_assays[assay_from_col].apply(safe_float)
            hole_assays[assay_to_col] = hole_assays[assay_to_col].apply(safe_float)
            hole_assays[assay_value_col] = hole_assays[assay_value_col].apply(safe_float)
            
            # Trouver la valeur maximale pour normaliser
            max_value = hole_assays[assay_value_col].max()
            
            # Dessiner des barres horizontales pour chaque intervalle d'analyse
            for _, row in hole_assays.iterrows():
                assay_from = row[assay_from_col]
                assay_to = row[assay_to_col]
                assay_value = row[assay_value_col]
                
                # Dessiner une barre horizontale pour la valeur
                bar_width = (assay_value / max_value) * 0.9 if max_value > 0 else 0  # Normaliser la largeur
                axes[2].add_patch(plt.Rectangle((0, assay_from), bar_width, assay_to - assay_from, 
                                                color='red', alpha=0.7))
                
                # Ajouter la valeur comme texte avec taille de police adaptative
                interval_height = assay_to - assay_from
                font_size = min(12, max(7, interval_height * 1))  # Taille de police ajustée pour les teneurs
                
                # Afficher seulement si l'intervalle est assez grand
                if interval_height > 1 or assay_value > max_value * 0.5:  # Afficher les valeurs importantes même dans de petits intervalles
                    axes[2].text(bar_width + 0.05, (assay_from + assay_to) / 2, f"{assay_value:.2f}",
                                va='center', fontsize=font_size, fontweight='bold')
    
    # Configurer l'axe des analyses
    axes[2].set_xlim(0, 1.2)
    axes[2].set_ylim(max_depth, 0)
    axes[2].set_xlabel(f'Analyses ({assay_value_col if assay_value_col else ""})')
    axes[2].grid(axis='y')
    
    plt.tight_layout()
    
    # Convertir le graphique en image pour Streamlit
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    
    return buf

# Fonction pour créer une représentation 3D des forages
def create_drillhole_3d_plot(collars_df, survey_df, lithology_df=None, assays_df=None, 
                            hole_id_col=None, x_col=None, y_col=None, z_col=None,
                            azimuth_col=None, dip_col=None, depth_col=None,
                            lith_from_col=None, lith_to_col=None, lith_col=None,
                            assay_from_col=None, assay_to_col=None, assay_value_col=None):
    
    if collars_df is None or survey_df is None:
        return None
    
    # Vérifier que les colonnes nécessaires existent
    if not all(col and col in collars_df.columns for col in [hole_id_col, x_col, y_col, z_col]) or \
       not all(col and col in survey_df.columns for col in [hole_id_col, azimuth_col, dip_col, depth_col]):
        st.markdown('<div class="error-message">Colonnes nécessaires manquantes dans les DataFrames collars ou survey</div>', unsafe_allow_html=True)
        return None
    
    # Convertir les colonnes numériques en flottants
    for col in [x_col, y_col, z_col]:
        collars_df[col] = collars_df[col].apply(safe_float)
    
    for col in [azimuth_col, dip_col, depth_col]:
        survey_df[col] = survey_df[col].apply(safe_float)
    
    # Convertir les colonnes lithologiques en flottants si nécessaires
    if lithology_df is not None and lith_from_col in lithology_df.columns and lith_to_col in lithology_df.columns:
        lithology_df[lith_from_col] = lithology_df[lith_from_col].apply(safe_float)
        lithology_df[lith_to_col] = lithology_df[lith_to_col].apply(safe_float)
    
    # Convertir les colonnes d'analyses en flottants si nécessaires
    if assays_df is not None and assay_from_col in assays_df.columns and assay_to_col in assays_df.columns and assay_value_col in assays_df.columns:
        assays_df[assay_from_col] = assays_df[assay_from_col].apply(safe_float)
        assays_df[assay_to_col] = assays_df[assay_to_col].apply(safe_float)
        assays_df[assay_value_col] = assays_df[assay_value_col].apply(safe_float)
    
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
    
    # Pour chaque trou de forage
    for hole_id in collars_df[hole_id_col].unique():
        # Récupérer les données de collar pour ce trou
        collar = collars_df[collars_df[hole_id_col] == hole_id]
        if collar.empty:
            continue
            
        # Point de départ du trou (collar)
        x_start = collar[x_col].values[0]
        y_start = collar[y_col].values[0]
        z_start = collar[z_col].values[0]
        
        # Récupérer les données de survey pour ce trou
        hole_surveys = survey_df[survey_df[hole_id_col] == hole_id].sort_values(by=depth_col)
        
        if hole_surveys.empty:
            continue
            
        # Calculer les points 3D pour le tracé du trou
        x_points = [x_start]
        y_points = [y_start]
        z_points = [z_start]
        
        current_x, current_y, current_z = x_start, y_start, z_start
        prev_depth = 0
        
        for _, survey in hole_surveys.iterrows():
            depth = survey[depth_col]
            azimuth = survey[azimuth_col]
            dip = survey[dip_col]
            
            segment_length = depth - prev_depth
            
            # Convertir l'azimuth et le dip en direction 3D
            azimuth_rad = np.radians(azimuth)
            dip_rad = np.radians(dip)
            
            # Calculer la nouvelle position
            dx = segment_length * np.sin(dip_rad) * np.sin(azimuth_rad)
            dy = segment_length * np.sin(dip_rad) * np.cos(azimuth_rad)
            dz = -segment_length * np.cos(dip_rad)  # Z est négatif pour la profondeur
            
            current_x += dx
            current_y += dy
            current_z += dz
            
            x_points.append(current_x)
            y_points.append(current_y)
            z_points.append(current_z)
            
            prev_depth = depth
        
        # Ajouter la trace du trou de forage
        fig.add_trace(
            go.Scatter3d(
                x=x_points,
                y=y_points,
                z=z_points,
                mode='lines',
                name=f'Forage {hole_id}',
                line=dict(width=4, color='blue'),
                hoverinfo='text',
                hovertext=[f'ID: {hole_id}<br>X: {x:.2f}<br>Y: {y:.2f}<br>Z: {z:.2f}' 
                          for x, y, z in zip(x_points, y_points, z_points)]
            )
        )
        
        # Ajouter les intersections lithologiques si disponibles
        litho_cols_exist = lithology_df is not None and all(col and col in lithology_df.columns for col in [hole_id_col, lith_from_col, lith_to_col, lith_col])
        if litho_cols_exist:
            hole_litho = lithology_df[lithology_df[hole_id_col] == hole_id]
            
            if not hole_litho.empty:
                for _, litho in hole_litho.iterrows():
                    from_depth = litho[lith_from_col]
                    to_depth = litho[lith_to_col]
                    lith_type = litho[lith_col]
                    
                    # Simplification: placer des marqueurs aux points médians des intervalles lithologiques
                    midpoint_depth = (from_depth + to_depth) / 2
                    
                    # Trouver les coordonnées 3D pour ce point
                    idx = np.interp(midpoint_depth, hole_surveys[depth_col], np.arange(len(hole_surveys)))
                    idx = int(min(idx, len(hole_surveys)-1))
                    
                    if idx < len(x_points) - 1:
                        # Calculer la fraction entre les deux points de survey
                        depths = hole_surveys[depth_col].values
                        if idx + 1 < len(depths):
                            fraction = (midpoint_depth - depths[idx]) / (depths[idx+1] - depths[idx]) if depths[idx+1] > depths[idx] else 0
                            
                            # Interpoler les coordonnées 3D
                            x_lith = x_points[idx] + fraction * (x_points[idx+1] - x_points[idx])
                            y_lith = y_points[idx] + fraction * (y_points[idx+1] - y_points[idx])
                            z_lith = z_points[idx] + fraction * (z_points[idx+1] - z_points[idx])
                            
                            # Ajouter un marqueur pour cette lithologie
                            color_idx = abs(hash(str(lith_type))) % len(px.colors.qualitative.Plotly)
                            
                            fig.add_trace(
                                go.Scatter3d(
                                    x=[x_lith],
                                    y=[y_lith],
                                    z=[z_lith],
                                    mode='markers',
                                    name=f'{hole_id}: {lith_type}',
                                    marker=dict(
                                        size=8,
                                        color=px.colors.qualitative.Plotly[color_idx]
                                    ),
                                    hoverinfo='text',
                                    hovertext=f'ID: {hole_id}<br>Lithologie: {lith_type}<br>Profondeur: {from_depth:.2f}-{to_depth:.2f}m'
                                )
                            )
        
        # Ajouter les valeurs d'analyses si disponibles
        assay_cols_exist = assays_df is not None and all(col and col in assays_df.columns for col in [hole_id_col, assay_from_col, assay_to_col, assay_value_col])
        if assay_cols_exist:
            hole_assays = assays_df[assays_df[hole_id_col] == hole_id]
            
            if not hole_assays.empty:
                max_value = hole_assays[assay_value_col].max()
                
                for _, assay in hole_assays.iterrows():
                    from_depth = assay[assay_from_col]
                    to_depth = assay[assay_to_col]
                    value = assay[assay_value_col]
                    
                    # Simplification: placer des marqueurs aux points médians des intervalles d'analyse
                    midpoint_depth = (from_depth + to_depth) / 2
                    
                    # Trouver les coordonnées 3D pour ce point
                    idx = np.interp(midpoint_depth, hole_surveys[depth_col], np.arange(len(hole_surveys)))
                    idx = int(min(idx, len(hole_surveys)-1))
                    
                    if idx < len(x_points) - 1:
                        # Calculer la fraction entre les deux points de survey
                        depths = hole_surveys[depth_col].values
                        if idx + 1 < len(depths):
                            fraction = (midpoint_depth - depths[idx]) / (depths[idx+1] - depths[idx]) if depths[idx+1] > depths[idx] else 0
                            
                            # Interpoler les coordonnées 3D
                            x_assay = x_points[idx] + fraction * (x_points[idx+1] - x_points[idx])
                            y_assay = y_points[idx] + fraction * (y_points[idx+1] - y_points[idx])
                            z_assay = z_points[idx] + fraction * (z_points[idx+1] - z_points[idx])
                            
                            # Normaliser la valeur pour déterminer la taille du marqueur (avec gestion des erreurs)
                            try:
                                # S'assurer que marker_size est un nombre valide et dans la plage acceptée
                                marker_size = 5 + (value / max_value) * 15 if max_value > 0 else 5
                                # S'assurer que marker_size est dans les limites raisonnables
                                marker_size = max(3, min(marker_size, 20))
                            except (ZeroDivisionError, TypeError, ValueError):
                                marker_size = 5  # Valeur par défaut en cas d'erreur
                            
                            # Vérifier si la valeur est un nombre valide
                            if np.isnan(value) or np.isinf(value):
                                value_for_color = 0
                            else:
                                value_for_color = value
                            
                            # Ajouter un marqueur pour cette analyse
                            fig.add_trace(
                                go.Scatter3d(
                                    x=[x_assay],
                                    y=[y_assay],
                                    z=[z_assay],
                                    mode='markers',
                                    name=f'{hole_id}: {value:.2f}',
                                    marker=dict(
                                        size=marker_size,
                                        color=value_for_color,
                                        colorscale='Reds',
                                        colorbar=dict(title=assay_value_col)
                                    ),
                                    hoverinfo='text',
                                    hovertext=f'ID: {hole_id}<br>Teneur: {value:.2f}<br>Profondeur: {from_depth:.2f}-{to_depth:.2f}m'
                                )
                            )
    
    # Ajuster la mise en page
    fig.update_layout(
        title={
            'text': "Visualisation 3D des forages",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24, 'color': '#2E4053'}
        },
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z (Élévation)",
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
            xaxis=dict(showbackground=True, backgroundcolor='rgb(240, 240, 240)'),
            yaxis=dict(showbackground=True, backgroundcolor='rgb(240, 240, 240)'),
            zaxis=dict(showbackground=True, backgroundcolor='rgb(240, 240, 240)')
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='#3498DB',
            borderwidth=1
        ),
        template='plotly_white'
    )
    
    return fig

# Barre latérale pour la navigation
with st.sidebar:
    st.markdown('<h2 style="color: #3498DB;">Navigation</h2>', unsafe_allow_html=True)
    page = st.radio('', [
        'Chargement des données', 
        'Aperçu des données', 
        'Composites', 
        'Strip Logs',
        'Visualisation 3D'
    ], index=0, key='navigation')
    
    st.markdown('---')
    st.markdown('<div style="padding: 20px; background-color: #EBF5FB; border-radius: 5px;">', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-weight: bold;">Mining Geology Data Application</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 0.9rem;">Cet outil permet de visualiser, analyser et interpréter les données de forages miniers.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Initialisation des variables de session
if 'collars_df' not in st.session_state:
    st.session_state.collars_df = None
    
if 'survey_df' not in st.session_state:
    st.session_state.survey_df = None
    
if 'lithology_df' not in st.session_state:
    st.session_state.lithology_df = None
    
if 'assays_df' not in st.session_state:
    st.session_state.assays_df = None
    
if 'composites_df' not in st.session_state:
    st.session_state.composites_df = None

if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {
        'hole_id': None,
        'x': None,
        'y': None,
        'z': None,
        'azimuth': None,
        'dip': None,
        'depth': None,
        'lith_from': None,
        'lith_to': None,
        'lithology': None,
        'assay_from': None,
        'assay_to': None,
        'assay_value': None
    }

# Page de chargement des données
if page == 'Chargement des données':
    st.markdown('<h2 class="sub-header">Chargement des données</h2>', unsafe_allow_html=True)
    
    # Créer des onglets pour les différents types de données
    tabs = st.tabs(["Collars", "Survey", "Lithologie", "Analyses"])
    
    # Onglet Collars
    with tabs[0]:
        st.markdown('<h3 style="color: #3498DB;">Chargement des données de collars</h3>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-card">Les données de collars contiennent les informations sur la position de départ des forages (coordonnées X, Y, Z).</div>', unsafe_allow_html=True)
        
        collars_file = st.file_uploader("Télécharger le fichier CSV des collars", type=['csv'])
        if collars_file is not None:
            try:
                st.session_state.collars_df = pd.read_csv(collars_file)
                st.markdown(f'<div class="success-message">✅ Fichier chargé avec succès. {len(st.session_state.collars_df)} enregistrements trouvés.</div>', unsafe_allow_html=True)
                
                # Sélection des colonnes importantes
                st.markdown('<h4 style="margin-top: 20px;">Sélection des colonnes</h4>', unsafe_allow_html=True)
                cols = st.session_state.collars_df.columns.tolist()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state.column_mapping['hole_id'] = st.selectbox("Colonne ID du trou", 
                                                                             [''] + cols, 
                                                                             index=0 if len(cols) == 0 else 1)
                with col2:
                    st.session_state.column_mapping['x'] = st.selectbox("Colonne X", 
                                                                        [''] + cols,
                                                                        index=0 if len(cols) == 0 else 1)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state.column_mapping['y'] = st.selectbox("Colonne Y", 
                                                                        [''] + cols,
                                                                        index=0 if len(cols) == 0 else 1)
                with col2:
                    st.session_state.column_mapping['z'] = st.selectbox("Colonne Z", 
                                                                        [''] + cols,
                                                                        index=0 if len(cols) == 0 else 1)
                
                # Aperçu des données
                if st.checkbox("Afficher l'aperçu des données"):
                    st.markdown('<h4 style="margin-top: 20px;">Aperçu des données</h4>', unsafe_allow_html=True)
                    st.dataframe(st.session_state.collars_df.head(), use_container_width=True)
            except Exception as e:
                st.markdown(f'<div class="error-message">❌ Erreur lors du chargement du fichier: {str(e)}</div>', unsafe_allow_html=True)
    
    # Onglet Survey
    with tabs[1]:
        st.markdown('<h3 style="color: #3498DB;">Chargement des données de survey</h3>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-card">Les données de survey contiennent les mesures d\'orientation (azimut, pendage) prises à différentes profondeurs le long du forage.</div>', unsafe_allow_html=True)
        
        survey_file = st.file_uploader("Télécharger le fichier CSV des surveys", type=['csv'])
        if survey_file is not None:
            try:
                st.session_state.survey_df = pd.read_csv(survey_file)
                st.markdown(f'<div class="success-message">✅ Fichier chargé avec succès. {len(st.session_state.survey_df)} enregistrements trouvés.</div>', unsafe_allow_html=True)
                
                # Sélection des colonnes importantes
                st.markdown('<h4 style="margin-top: 20px;">Sélection des colonnes</h4>', unsafe_allow_html=True)
                cols = st.session_state.survey_df.columns.tolist()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state.column_mapping['hole_id'] = st.selectbox("Colonne ID du trou (Survey)", 
                                                                             [''] + cols, 
                                                                             index=0 if len(cols) == 0 else 1)
                with col2:
                    st.session_state.column_mapping['depth'] = st.selectbox("Colonne profondeur", 
                                                                            [''] + cols,
                                                                            index=0 if len(cols) == 0 else 1)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state.column_mapping['azimuth'] = st.selectbox("Colonne azimut", 
                                                                              [''] + cols,
                                                                              index=0 if len(cols) == 0 else 1)
                with col2:
                    st.session_state.column_mapping['dip'] = st.selectbox("Colonne pendage", 
                                                                          [''] + cols,
                                                                          index=0 if len(cols) == 0 else 1)
                
                # Aperçu des données
                if st.checkbox("Afficher l'aperçu des données (Survey)"):
                    st.markdown('<h4 style="margin-top: 20px;">Aperçu des données</h4>', unsafe_allow_html=True)
                    st.dataframe(st.session_state.survey_df.head(), use_container_width=True)
            except Exception as e:
                st.markdown(f'<div class="error-message">❌ Erreur lors du chargement du fichier: {str(e)}</div>', unsafe_allow_html=True)
    
    # Onglet Lithologie
    with tabs[2]:
        st.markdown('<h3 style="color: #3498DB;">Chargement des données de lithologie</h3>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-card">Les données de lithologie contiennent des informations sur les types de roches rencontrés à différentes profondeurs lors du forage.</div>', unsafe_allow_html=True)
        
        lithology_file = st.file_uploader("Télécharger le fichier CSV des lithologies", type=['csv'])
        if lithology_file is not None:
            try:
                st.session_state.lithology_df = pd.read_csv(lithology_file)
                st.markdown(f'<div class="success-message">✅ Fichier chargé avec succès. {len(st.session_state.lithology_df)} enregistrements trouvés.</div>', unsafe_allow_html=True)
                
                # Sélection des colonnes importantes
                st.markdown('<h4 style="margin-top: 20px;">Sélection des colonnes</h4>', unsafe_allow_html=True)
                cols = st.session_state.lithology_df.columns.tolist()
                
                col1, col2 = st.columns(2)
                with col1:
                    hole_id_index = 1 if cols and len(cols) > 0 else 0
                    st.session_state.column_mapping['hole_id'] = st.selectbox("Colonne ID du trou (Lithologie)", 
                                                                             [''] + cols, 
                                                                             index=hole_id_index)
                with col2:
                    lith_index = 2 if cols and len(cols) > 1 else (1 if cols and len(cols) > 0 else 0)
                    st.session_state.column_mapping['lithology'] = st.selectbox("Colonne de lithologie", 
                                                                               [''] + cols,
                                                                               index=lith_index)
                
                col1, col2 = st.columns(2)
                with col1:
                    lith_from_index = 3 if cols and len(cols) > 2 else (1 if cols and len(cols) > 0 else 0)
                    st.session_state.column_mapping['lith_from'] = st.selectbox("Colonne de profondeur début", 
                                                                               [''] + cols,
                                                                               index=lith_from_index)
                with col2:
                    lith_to_index = 4 if cols and len(cols) > 3 else (2 if cols and len(cols) > 1 else 0)
                    st.session_state.column_mapping['lith_to'] = st.selectbox("Colonne de profondeur fin", 
                                                                             [''] + cols,
                                                                             index=lith_to_index)
                
                # Aperçu des données
                if st.checkbox("Afficher l'aperçu des données (Lithologie)"):
                    st.markdown('<h4 style="margin-top: 20px;">Aperçu des données</h4>', unsafe_allow_html=True)
                    st.dataframe(st.session_state.lithology_df.head(), use_container_width=True)
            except Exception as e:
                st.markdown(f'<div class="error-message">❌ Erreur lors du chargement du fichier: {str(e)}</div>', unsafe_allow_html=True)
    
    # Onglet Analyses
    with tabs[3]:
        st.markdown('<h3 style="color: #3498DB;">Chargement des données d\'analyses</h3>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-card">Les données d\'analyses contiennent les résultats d\'analyses géochimiques réalisées sur les échantillons de carottes à différentes profondeurs.</div>', unsafe_allow_html=True)
        
        assays_file = st.file_uploader("Télécharger le fichier CSV des analyses", type=['csv'])
        if assays_file is not None:
            try:
                st.session_state.assays_df = pd.read_csv(assays_file)
                st.markdown(f'<div class="success-message">✅ Fichier chargé avec succès. {len(st.session_state.assays_df)} enregistrements trouvés.</div>', unsafe_allow_html=True)
                
                # Sélection des colonnes importantes
                st.markdown('<h4 style="margin-top: 20px;">Sélection des colonnes</h4>', unsafe_allow_html=True)
                cols = st.session_state.assays_df.columns.tolist()
                
                col1, col2 = st.columns(2)
                with col1:
                    hole_id_index = 1 if cols and len(cols) > 0 else 0
                    st.session_state.column_mapping['hole_id'] = st.selectbox("Colonne ID du trou (Analyses)", 
                                                                             [''] + cols, 
                                                                             index=hole_id_index)
                with col2:
                    value_index = 4 if cols and len(cols) > 3 else (1 if cols and len(cols) > 0 else 0)
                    st.session_state.column_mapping['assay_value'] = st.selectbox("Colonne de valeur (par ex. Au g/t)", 
                                                                                 [''] + cols,
                                                                                 index=value_index)
                
                col1, col2 = st.columns(2)
                with col1:
                    from_index = 2 if cols and len(cols) > 1 else (1 if cols and len(cols) > 0 else 0)
                    st.session_state.column_mapping['assay_from'] = st.selectbox("Colonne de profondeur début (Analyses)", 
                                                                                [''] + cols,
                                                                                index=from_index)
                with col2:
                    to_index = 3 if cols and len(cols) > 2 else (2 if cols and len(cols) > 1 else 0)
                    st.session_state.column_mapping['assay_to'] = st.selectbox("Colonne de profondeur fin (Analyses)", 
                                                                              [''] + cols,
                                                                              index=to_index)
                
                # Aperçu des données
                if st.checkbox("Afficher l'aperçu des données (Analyses)"):
                    st.markdown('<h4 style="margin-top: 20px;">Aperçu des données</h4>', unsafe_allow_html=True)
                    st.dataframe(st.session_state.assays_df.head(), use_container_width=True)
            except Exception as e:
                st.markdown(f'<div class="error-message">❌ Erreur lors du chargement du fichier: {str(e)}</div>', unsafe_allow_html=True)

# Page d'aperçu des données
elif page == 'Aperçu des données':
    st.markdown('<h2 class="sub-header">Aperçu des données</h2>', unsafe_allow_html=True)
    
    # Vérifier si des données ont été chargées
    if st.session_state.collars_df is None and st.session_state.survey_df is None and st.session_state.lithology_df is None and st.session_state.assays_df is None:
        st.markdown('<div class="warning-message">⚠️ Aucune donnée n\'a été chargée. Veuillez d\'abord charger des données.</div>', unsafe_allow_html=True)
    else:
        # Créer des onglets pour les différents types de données
        data_tabs = st.tabs(["Collars", "Survey", "Lithologie", "Analyses"])
        
        # Onglet Collars
        with data_tabs[0]:
            if st.session_state.collars_df is not None:
                st.markdown('<h3 style="color: #3498DB;">Données de collars</h3>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'<div class="metric-card"><span style="font-size: 1.5rem; font-weight: bold;">{len(st.session_state.collars_df)}</span><br>Enregistrements</div>', unsafe_allow_html=True)
                with col2:
                    if 'hole_id' in st.session_state.column_mapping and st.session_state.column_mapping['hole_id'] in st.session_state.collars_df.columns:
                        unique_holes = st.session_state.collars_df[st.session_state.column_mapping['hole_id']].nunique()
                        st.markdown(f'<div class="metric-card"><span style="font-size: 1.5rem; font-weight: bold;">{unique_holes}</span><br>Forages uniques</div>', unsafe_allow_html=True)
                
                st.dataframe(st.session_state.collars_df, use_container_width=True)
                
                st.markdown(get_csv_download_link(st.session_state.collars_df, "collars_data.csv", "les données de collars"), unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-card">Aucune donnée de collars n\'a été chargée.</div>', unsafe_allow_html=True)
        
        # Onglet Survey
        with data_tabs[1]:
            if st.session_state.survey_df is not None:
                st.markdown('<h3 style="color: #3498DB;">Données de survey</h3>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'<div class="metric-card"><span style="font-size: 1.5rem; font-weight: bold;">{len(st.session_state.survey_df)}</span><br>Enregistrements</div>', unsafe_allow_html=True)
                with col2:
                    if 'hole_id' in st.session_state.column_mapping and st.session_state.column_mapping['hole_id'] in st.session_state.survey_df.columns:
                        unique_holes = st.session_state.survey_df[st.session_state.column_mapping['hole_id']].nunique()
                        st.markdown(f'<div class="metric-card"><span style="font-size: 1.5rem; font-weight: bold;">{unique_holes}</span><br>Forages uniques</div>', unsafe_allow_html=True)
                
                st.dataframe(st.session_state.survey_df, use_container_width=True)
                
                st.markdown(get_csv_download_link(st.session_state.survey_df, "survey_data.csv", "les données de survey"), unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-card">Aucune donnée de survey n\'a été chargée.</div>', unsafe_allow_html=True)
        
        # Onglet Lithologie
        with data_tabs[2]:
            if st.session_state.lithology_df is not None:
                st.markdown('<h3 style="color: #3498DB;">Données de lithologie</h3>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'<div class="metric-card"><span style="font-size: 1.5rem; font-weight: bold;">{len(st.session_state.lithology_df)}</span><br>Enregistrements</div>', unsafe_allow_html=True)
                with col2:
                    if 'hole_id' in st.session_state.column_mapping and st.session_state.column_mapping['hole_id'] in st.session_state.lithology_df.columns:
                        unique_holes = st.session_state.lithology_df[st.session_state.column_mapping['hole_id']].nunique()
                        st.markdown(f'<div class="metric-card"><span style="font-size: 1.5rem; font-weight: bold;">{unique_holes}</span><br>Forages uniques</div>', unsafe_allow_html=True)
                with col3:
                    if 'lithology' in st.session_state.column_mapping and st.session_state.column_mapping['lithology'] in st.session_state.lithology_df.columns:
                        unique_liths = st.session_state.lithology_df[st.session_state.column_mapping['lithology']].nunique()
                        st.markdown(f'<div class="metric-card"><span style="font-size: 1.5rem; font-weight: bold;">{unique_liths}</span><br>Types de lithologies</div>', unsafe_allow_html=True)
                
                st.dataframe(st.session_state.lithology_df, use_container_width=True)
                
                st.markdown(get_csv_download_link(st.session_state.lithology_df, "lithology_data.csv", "les données de lithologie"), unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-card">Aucune donnée de lithologie n\'a été chargée.</div>', unsafe_allow_html=True)
        
        # Onglet Analyses
        with data_tabs[3]:
            if st.session_state.assays_df is not None:
                st.markdown('<h3 style="color: #3498DB;">Données d\'analyses</h3>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'<div class="metric-card"><span style="font-size: 1.5rem; font-weight: bold;">{len(st.session_state.assays_df)}</span><br>Enregistrements</div>', unsafe_allow_html=True)
                with col2:
                    if 'hole_id' in st.session_state.column_mapping and st.session_state.column_mapping['hole_id'] in st.session_state.assays_df.columns:
                        unique_holes = st.session_state.assays_df[st.session_state.column_mapping['hole_id']].nunique()
                        st.markdown(f'<div class="metric-card"><span style="font-size: 1.5rem; font-weight: bold;">{unique_holes}</span><br>Forages uniques</div>', unsafe_allow_html=True)
                with col3:
                    if 'assay_value' in st.session_state.column_mapping and st.session_state.column_mapping['assay_value'] in st.session_state.assays_df.columns:
                        # Convertir en nombre avant de calculer la moyenne
                        values = pd.to_numeric(st.session_state.assays_df[st.session_state.column_mapping['assay_value']], errors='coerce')
                        avg_value = values.mean()
                        st.markdown(f'<div class="metric-card"><span style="font-size: 1.5rem; font-weight: bold;">{avg_value:.2f}</span><br>Valeur moyenne</div>', unsafe_allow_html=True)
                
                st.dataframe(st.session_state.assays_df, use_container_width=True)
                
                st.markdown(get_csv_download_link(st.session_state.assays_df, "assays_data.csv", "les données d'analyses"), unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-card">Aucune donnée d\'analyses n\'a été chargée.</div>', unsafe_allow_html=True)

# Page de calcul des composites
elif page == 'Composites':
    st.markdown('<h2 class="sub-header">Calcul des composites</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-card">Les composites permettent de regrouper les données d\'analyses en intervalles réguliers pour faciliter l\'interprétation et la modélisation.</div>', unsafe_allow_html=True)
    
    if st.session_state.assays_df is None:
        st.markdown('<div class="warning-message">⚠️ Aucune donnée d\'analyses n\'a été chargée. Veuillez d\'abord charger des données d\'analyses.</div>', unsafe_allow_html=True)
    else:
        hole_id_col = st.session_state.column_mapping['hole_id']
        assay_from_col = st.session_state.column_mapping['assay_from']
        assay_to_col = st.session_state.column_mapping['assay_to']
        assay_value_col = st.session_state.column_mapping['assay_value']
        
        # Vérifier que les colonnes nécessaires existent dans le DataFrame des analyses
        if not all(col and col in st.session_state.assays_df.columns for col in [hole_id_col, assay_from_col, assay_to_col, assay_value_col]):
            st.markdown('<div class="warning-message">⚠️ Certaines colonnes nécessaires n\'existent pas dans les données d\'analyses. Veuillez vérifier la sélection des colonnes.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<h3 style="color: #3498DB;">Options de composites</h3>', unsafe_allow_html=True)
            
            # Sélectionner la longueur des composites
            composite_length = st.slider("Longueur des composites (m)", 
                                        min_value=0.5, 
                                        max_value=5.0, 
                                        value=1.0, 
                                        step=0.5)
            
            # Option pour ajouter les coordonnées
            add_coordinates = st.checkbox("Ajouter les coordonnées aux composites", value=True)
            
            # Informations sur l'ajout de coordonnées
            if add_coordinates:
                st.markdown('<div class="info-card" style="margin-top: 10px;">Les coordonnées X, Y, Z seront calculées et ajoutées à chaque composite en utilisant les données de collar et survey.</div>', unsafe_allow_html=True)
            
            # Calculer les composites
            if st.button("Calculer les composites"):
                with st.spinner("Calcul des composites en cours..."):
                    try:
                        # Si on demande d'ajouter les coordonnées, vérifier que les données nécessaires sont disponibles
                        include_coordinates = (add_coordinates and 
                                              st.session_state.collars_df is not None and 
                                              st.session_state.survey_df is not None and
                                              all(col and col in st.session_state.collars_df.columns for col in 
                                                  [hole_id_col, st.session_state.column_mapping['x'], 
                                                   st.session_state.column_mapping['y'], st.session_state.column_mapping['z']]) and
                                              all(col and col in st.session_state.survey_df.columns for col in 
                                                  [hole_id_col, st.session_state.column_mapping['depth'], 
                                                   st.session_state.column_mapping['azimuth'], st.session_state.column_mapping['dip']]))
                        
                        if add_coordinates and not include_coordinates:
                            st.markdown('<div class="warning-message">⚠️ Impossible d\'ajouter les coordonnées: données de collar ou survey insuffisantes.</div>', unsafe_allow_html=True)
                        
                        st.session_state.composites_df = create_composites(
                            st.session_state.assays_df,
                            hole_id_col,
                            assay_from_col,
                            assay_to_col,
                            assay_value_col,
                            composite_length,
                            st.session_state.collars_df if include_coordinates else None,
                            st.session_state.survey_df if include_coordinates else None,
                            st.session_state.column_mapping['x'] if include_coordinates else None,
                            st.session_state.column_mapping['y'] if include_coordinates else None,
                            st.session_state.column_mapping['z'] if include_coordinates else None,
                            st.session_state.column_mapping['azimuth'] if include_coordinates else None,
                            st.session_state.column_mapping['dip'] if include_coordinates else None,
                            st.session_state.column_mapping['depth'] if include_coordinates else None
                        )
                        
                        if st.session_state.composites_df is not None and not st.session_state.composites_df.empty:
                            st.markdown(f'<div class="success-message">✅ Composites calculés avec succès. {len(st.session_state.composites_df)} enregistrements générés.</div>', unsafe_allow_html=True)
                            
                            # Afficher les composites
                            st.markdown('<h3 style="color: #3498DB; margin-top: 20px;">Résultats des composites</h3>', unsafe_allow_html=True)
                            st.dataframe(st.session_state.composites_df, use_container_width=True)
                            
                            # Lien de téléchargement
                            st.markdown(get_csv_download_link(st.session_state.composites_df, "composites.csv", "les composites"), unsafe_allow_html=True)
                            
                            # Résumé statistique
                            st.markdown('<h3 style="color: #3498DB; margin-top: 20px;">Résumé statistique</h3>', unsafe_allow_html=True)
                            
                            # Convertir en flottant pour les calculs statistiques
                            st.session_state.composites_df[assay_value_col] = st.session_state.composites_df[assay_value_col].apply(safe_float)
                            st.write(st.session_state.composites_df[assay_value_col].describe())
                            
                            # Histogramme des valeurs de composites
                            st.markdown('<h3 style="color: #3498DB; margin-top: 20px;">Distribution des valeurs</h3>', unsafe_allow_html=True)
                            
                            fig = px.histogram(
                                st.session_state.composites_df, 
                                x=assay_value_col,
                                title=f"Distribution des valeurs de composites ({assay_value_col})",
                                labels={assay_value_col: f'Teneur'},
                                color_discrete_sequence=['#3498DB'],
                                template='plotly_white'
                            )
                            
                            fig.update_layout(
                                xaxis_title=f"Teneur ({assay_value_col})",
                                yaxis_title="Fréquence",
                                title={
                                    'y':0.95,
                                    'x':0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top',
                                    'font': {'size': 16, 'color': '#2E4053'}
                                },
                                bargap=0.1
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Comparaison avec les données originales
                            st.markdown('<h3 style="color: #3498DB; margin-top: 20px;">Comparaison avec les données originales</h3>', unsafe_allow_html=True)
                            
                            # Convertir les valeurs originales en flottants également
                            st.session_state.assays_df[assay_value_col] = st.session_state.assays_df[assay_value_col].apply(safe_float)
                            
                            comparison_data = pd.DataFrame({
                                'Statistique': ['Moyenne', 'Médiane', 'Écart-type', 'Minimum', 'Maximum'],
                                'Données originales': [
                                    st.session_state.assays_df[assay_value_col].mean(),
                                    st.session_state.assays_df[assay_value_col].median(),
                                    st.session_state.assays_df[assay_value_col].std(),
                                    st.session_state.assays_df[assay_value_col].min(),
                                    st.session_state.assays_df[assay_value_col].max()
                                ],
                                'Composites': [
                                    st.session_state.composites_df[assay_value_col].mean(),
                                    st.session_state.composites_df[assay_value_col].median(),
                                    st.session_state.composites_df[assay_value_col].std(),
                                    st.session_state.composites_df[assay_value_col].min(),
                                    st.session_state.composites_df[assay_value_col].max()
                                ]
                            })
                            
                            # Créer une figure pour le comparatif
                            fig_comp = go.Figure()
                            
                            # Ajouter une trace pour les données originales
                            fig_comp.add_trace(go.Bar(
                                name='Données originales',
                                x=comparison_data['Statistique'],
                                y=comparison_data['Données originales'],
                                marker_color='#3498DB'
                            ))
                            
                            # Ajouter une trace pour les composites
                            fig_comp.add_trace(go.Bar(
                                name='Composites',
                                x=comparison_data['Statistique'],
                                y=comparison_data['Composites'],
                                marker_color='#2ECC71'
                            ))
                            
                            # Mettre à jour la mise en page
                            fig_comp.update_layout(
                                title={
                                    'text': 'Comparaison des statistiques: Données originales vs Composites',
                                    'y':0.95,
                                    'x':0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top',
                                    'font': {'size': 16, 'color': '#2E4053'}
                                },
                                barmode='group',
                                xaxis={'title': 'Statistique'},
                                yaxis={'title': 'Valeur'},
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig_comp, use_container_width=True)
                            
                            # Afficher aussi le tableau des comparaisons
                            st.table(comparison_data.set_index('Statistique').round(3))
                            
                            # Si les coordonnées ont été ajoutées, afficher une visualisation 3D
                            if 'X' in st.session_state.composites_df.columns and 'Y' in st.session_state.composites_df.columns and 'Z' in st.session_state.composites_df.columns:
                                st.markdown('<h3 style="color: #3498DB; margin-top: 20px;">Visualisation 3D des composites</h3>', unsafe_allow_html=True)
                                
                                fig_3d = px.scatter_3d(
                                    st.session_state.composites_df,
                                    x='X',
                                    y='Y',
                                    z='Z',
                                    color=assay_value_col,
                                    color_continuous_scale='Viridis',
                                    size=assay_value_col,
                                    size_max=10,
                                    opacity=0.7,
                                    hover_data={
                                        hole_id_col: True,
                                        'From': True,
                                        'To': True,
                                        assay_value_col: ':.2f'
                                    },
                                    title='Distribution spatiale des composites'
                                )
                                
                                fig_3d.update_layout(
                                    scene={
                                        'aspectmode': 'data',
                                        'camera': {'eye': {'x': 1.5, 'y': 1.5, 'z': 1.2}},
                                        'xaxis_title': 'X',
                                        'yaxis_title': 'Y',
                                        'zaxis_title': 'Z (Élévation)'
                                    },
                                    margin=dict(l=0, r=0, b=0, t=40),
                                    coloraxis_colorbar=dict(
                                        title=assay_value_col
                                    ),
                                    template='plotly_white'
                                )
                                
                                st.plotly_chart(fig_3d, use_container_width=True)
                        else:
                            st.markdown('<div class="error-message">❌ Impossible de calculer les composites avec les données fournies.</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<div class="error-message">❌ Erreur lors du calcul des composites: {str(e)}</div>', unsafe_allow_html=True)

# Page de Strip Logs
elif page == 'Strip Logs':
    st.markdown('<h2 class="sub-header">Strip Logs des forages</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-card">Les strip logs permettent de visualiser graphiquement les données de lithologie et d\'analyses le long d\'un forage.</div>', unsafe_allow_html=True)
    
    if st.session_state.collars_df is None or st.session_state.survey_df is None:
        st.markdown('<div class="warning-message">⚠️ Les données de collars et de survey sont nécessaires pour les strip logs. Veuillez les charger d\'abord.</div>', unsafe_allow_html=True)
    else:
        hole_id_col = st.session_state.column_mapping['hole_id']
        depth_col = st.session_state.column_mapping['depth']
        
        # Vérifier que les colonnes nécessaires existent
        if not hole_id_col or not depth_col or hole_id_col not in st.session_state.collars_df.columns or hole_id_col not in st.session_state.survey_df.columns or depth_col not in st.session_state.survey_df.columns:
            st.markdown('<div class="warning-message">⚠️ Veuillez d\'abord spécifier correctement les colonnes d\'ID de trou et de profondeur dans la page de chargement des données.</div>', unsafe_allow_html=True)
        else:
            # Sélection du forage à afficher
            all_holes = sorted(st.session_state.collars_df[hole_id_col].unique())
            if not all_holes:
                st.markdown('<div class="warning-message">⚠️ Aucun forage trouvé dans les données.</div>', unsafe_allow_html=True)
            else:
                selected_hole = st.selectbox("Sélectionner un forage", all_holes)
                
                if selected_hole:
                    # Récupérer les informations du forage sélectionné
                    selected_collar = st.session_state.collars_df[st.session_state.collars_df[hole_id_col] == selected_hole]
                    selected_survey = st.session_state.survey_df[st.session_state.survey_df[hole_id_col] == selected_hole]
                    
                    if not selected_survey.empty:
                        # Afficher les informations du forage
                        collar_info = selected_collar.iloc[0]
                        
                        st.markdown(f'<h3 style="color: #3498DB;">Informations sur le forage {selected_hole}</h3>', unsafe_allow_html=True)
                        
                        info_cols = st.columns(3)
                        with info_cols[0]:
                            st.markdown('<div class="info-card" style="height: 100%;">', unsafe_allow_html=True)
                            x_col = st.session_state.column_mapping['x']
                            y_col = st.session_state.column_mapping['y']
                            z_col = st.session_state.column_mapping['z']
                            
                            st.markdown('<p style="font-weight: bold; margin-bottom: 5px;">Coordonnées</p>', unsafe_allow_html=True)
                            
                            if x_col and x_col in selected_collar.columns:
                                st.write(f"X: {safe_float(collar_info[x_col]):.2f}")
                            if y_col and y_col in selected_collar.columns:
                                st.write(f"Y: {safe_float(collar_info[y_col]):.2f}")
                            if z_col and z_col in selected_collar.columns:
                                st.write(f"Z: {safe_float(collar_info[z_col]):.2f}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with info_cols[1]:
                            st.markdown('<div class="info-card" style="height: 100%;">', unsafe_allow_html=True)
                            max_depth = safe_float(selected_survey[depth_col].max())
                            st.markdown('<p style="font-weight: bold; margin-bottom: 5px;">Profondeur</p>', unsafe_allow_html=True)
                            st.write(f"Profondeur maximale: {max_depth:.2f} m")
                            
                            # Infos supplémentaires si lithologie disponible
                            if st.session_state.lithology_df is not None:
                                lith_col = st.session_state.column_mapping['lithology']
                                if hole_id_col in st.session_state.lithology_df.columns and lith_col and lith_col in st.session_state.lithology_df.columns:
                                    selected_litho = st.session_state.lithology_df[st.session_state.lithology_df[hole_id_col] == selected_hole]
                                    if not selected_litho.empty:
                                        unique_liths = selected_litho[lith_col].nunique()
                                        st.write(f"Nombre de lithologies: {unique_liths}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with info_cols[2]:
                            st.markdown('<div class="info-card" style="height: 100%;">', unsafe_allow_html=True)
                            # Infos supplémentaires si analyses disponibles
                            st.markdown('<p style="font-weight: bold; margin-bottom: 5px;">Analyses</p>', unsafe_allow_html=True)
                            if st.session_state.assays_df is not None:
                                assay_value_col = st.session_state.column_mapping['assay_value']
                                if hole_id_col in st.session_state.assays_df.columns and assay_value_col and assay_value_col in st.session_state.assays_df.columns:
                                    selected_assays = st.session_state.assays_df[st.session_state.assays_df[hole_id_col] == selected_hole]
                                    if not selected_assays.empty:
                                        # Convertir en nombres
                                        selected_assays[assay_value_col] = selected_assays[assay_value_col].apply(safe_float)
                                        avg_value = selected_assays[assay_value_col].mean()
                                        max_value = selected_assays[assay_value_col].max()
                                        st.write(f"Valeur moyenne: {avg_value:.2f}")
                                        st.write(f"Valeur maximale: {max_value:.2f}")
                                else:
                                    st.write("Aucune donnée d'analyse disponible")
                            else:
                                st.write("Aucune donnée d'analyse disponible")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Créer et afficher le strip log
                        try:
                            with st.spinner("Création du strip log en cours..."):
                                strip_log_image = create_strip_log(
                                    selected_hole,
                                    st.session_state.collars_df,
                                    st.session_state.survey_df,
                                    st.session_state.lithology_df,
                                    st.session_state.assays_df,
                                    hole_id_col,
                                    depth_col,
                                    st.session_state.column_mapping['lith_from'],
                                    st.session_state.column_mapping['lith_to'],
                                    st.session_state.column_mapping['lithology'],
                                    st.session_state.column_mapping['assay_from'],
                                    st.session_state.column_mapping['assay_to'],
                                    st.session_state.column_mapping['assay_value']
                                )
                            
                            if strip_log_image:
                                st.image(strip_log_image, caption=f"Strip Log du forage {selected_hole}", use_column_width=True)
                                
                                # Téléchargement de l'image
                                download_col1, download_col2 = st.columns([1, 3])
                                with download_col1:
                                    btn = st.download_button(
                                        label="Télécharger le strip log",
                                        data=strip_log_image,
                                        file_name=f"strip_log_{selected_hole}.png",
                                        mime="image/png"
                                    )
                            else:
                                st.markdown('<div class="error-message">❌ Impossible de créer le strip log avec les données fournies.</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.markdown(f'<div class="error-message">❌ Erreur lors de la création du strip log: {str(e)}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="error-message">❌ Aucune donnée de survey trouvée pour le forage {selected_hole}.</div>', unsafe_allow_html=True)

# Page de visualisation 3D
elif page == 'Visualisation 3D':
    st.markdown('<h2 class="sub-header">Visualisation 3D des forages</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-card">La visualisation 3D permet de visualiser les trajectoires des forages dans l\'espace et les données associées.</div>', unsafe_allow_html=True)
    
    # Vérifier si les données nécessaires ont été chargées
    if st.session_state.collars_df is None or st.session_state.survey_df is None:
        st.markdown('<div class="warning-message">⚠️ Les données de collars et de survey sont nécessaires pour la visualisation 3D. Veuillez les charger d\'abord.</div>', unsafe_allow_html=True)
    else:
        # Vérifier si les colonnes nécessaires ont été spécifiées
        hole_id_col = st.session_state.column_mapping['hole_id']
        x_col = st.session_state.column_mapping['x']
        y_col = st.session_state.column_mapping['y']
        z_col = st.session_state.column_mapping['z']
        azimuth_col = st.session_state.column_mapping['azimuth']
        dip_col = st.session_state.column_mapping['dip']
        depth_col = st.session_state.column_mapping['depth']
        
        required_cols_exist = (
            hole_id_col and hole_id_col in st.session_state.collars_df.columns and hole_id_col in st.session_state.survey_df.columns and
            x_col and x_col in st.session_state.collars_df.columns and
            y_col and y_col in st.session_state.collars_df.columns and
            z_col and z_col in st.session_state.collars_df.columns and
            azimuth_col and azimuth_col in st.session_state.survey_df.columns and
            dip_col and dip_col in st.session_state.survey_df.columns and
            depth_col and depth_col in st.session_state.survey_df.columns
        )
        
        if not required_cols_exist:
            st.markdown('<div class="warning-message">⚠️ Certaines colonnes requises n\'ont pas été spécifiées ou n\'existent pas dans les données. Veuillez vérifier la sélection des colonnes dans l\'onglet \'Chargement des données\'.</div>', unsafe_allow_html=True)
        else:
            # Options pour la visualisation
            st.markdown('<h3 style="color: #3498DB;">Options de visualisation</h3>', unsafe_allow_html=True)
            
            # Sélection des forages à afficher
            all_holes = sorted(st.session_state.collars_df[hole_id_col].unique())
            
            if not all_holes:
                st.markdown('<div class="warning-message">⚠️ Aucun forage trouvé dans les données.</div>', unsafe_allow_html=True)
            else:
                selected_holes = st.multiselect("Sélectionner les forages à afficher", all_holes, default=all_holes[:min(5, len(all_holes))])
                
                # Options additionnelles
                option_cols = st.columns(2)
                with option_cols[0]:
                    show_lithology = st.checkbox("Afficher la lithologie", value=True if st.session_state.lithology_df is not None else False)
                with option_cols[1]:
                    show_assays = st.checkbox("Afficher les teneurs", value=True if st.session_state.assays_df is not None else False)
                
                # Filtrer les données selon les forages sélectionnés
                if selected_holes:
                    filtered_collars = st.session_state.collars_df[st.session_state.collars_df[hole_id_col].isin(selected_holes)]
                    filtered_survey = st.session_state.survey_df[st.session_state.survey_df[hole_id_col].isin(selected_holes)]
                    
                    # Filtrer lithology et assays si nécessaire
                    filtered_lithology = None
                    if show_lithology and st.session_state.lithology_df is not None:
                        # Vérifier que hole_id_col existe dans lithology_df avant de filtrer
                        if hole_id_col in st.session_state.lithology_df.columns:
                            filtered_lithology = st.session_state.lithology_df[st.session_state.lithology_df[hole_id_col].isin(selected_holes)]
                        else:
                            st.markdown(f'<div class="warning-message">⚠️ La colonne {hole_id_col} n\'existe pas dans les données de lithologie.</div>', unsafe_allow_html=True)
                    
                    filtered_assays = None
                    if show_assays and st.session_state.assays_df is not None:
                        # Vérifier que hole_id_col existe dans assays_df avant de filtrer
                        if hole_id_col in st.session_state.assays_df.columns:
                            filtered_assays = st.session_state.assays_df[st.session_state.assays_df[hole_id_col].isin(selected_holes)]
                        else:
                            st.markdown(f'<div class="warning-message">⚠️ La colonne {hole_id_col} n\'existe pas dans les données d\'analyses.</div>', unsafe_allow_html=True)
                    
                    # Créer la visualisation 3D
                    try:
                        with st.spinner("Création de la visualisation 3D en cours..."):
                            fig_3d = create_drillhole_3d_plot(
                                filtered_collars, 
                                filtered_survey, 
                                filtered_lithology, 
                                filtered_assays,
                                hole_id_col=hole_id_col,
                                x_col=x_col,
                                y_col=y_col,
                                z_col=z_col,
                                azimuth_col=azimuth_col,
                                dip_col=dip_col,
                                depth_col=depth_col,
                                lith_from_col=st.session_state.column_mapping['lith_from'],
                                lith_to_col=st.session_state.column_mapping['lith_to'],
                                lith_col=st.session_state.column_mapping['lithology'],
                                assay_from_col=st.session_state.column_mapping['assay_from'],
                                assay_to_col=st.session_state.column_mapping['assay_to'],
                                assay_value_col=st.session_state.column_mapping['assay_value']
                            )
                        
                        if fig_3d:
                            st.plotly_chart(fig_3d, use_container_width=True, height=800)
                        else:
                            st.markdown('<div class="error-message">❌ Impossible de créer la visualisation 3D avec les données fournies.</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<div class="error-message">❌ Erreur lors de la création de la visualisation 3D: {str(e)}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="info-card">Veuillez sélectionner au moins un forage à afficher.</div>', unsafe_allow_html=True)

# Pied de page
st.markdown("""
<div style="position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; background-color: #F8F9F9; border-top: 1px solid #E5E7E9;">
    <p style="font-size: 0.8rem; color: #7F8C8D;">© 2025 Didier Ouedraogo, P.Geo. | Mining Geology Data Application</p>
</div>
""", unsafe_allow_html=True)