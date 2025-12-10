
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd
import time
import os
import requests
from io import BytesIO
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="Recommandation Mode par Morphologie",
    page_icon="👗",
    layout="wide"
)

# CSS PROFESSIONNEL
st.markdown("""
<style>
    /* ===== VARIABLES DE COULEUR ===== */
    :root {
        --primary: #667eea;
        --secondary: #764ba2;
        --accent: #f56565;
        --success: #48bb78;
        --warning: #ed8936;
        --light: #f7fafc;
        --dark: #2d3748;
        --gray-100: #f7fafc;
        --gray-200: #edf2f7;
        --gray-300: #e2e8f0;
        --gray-400: #cbd5e0;
        --gray-500: #a0aec0;
        --gray-600: #718096;
        --gray-700: #4a5568;
        --gray-800: #2d3748;
        --gray-900: #1a202c;
    }

    /* ===== RÉINITIALISATION ET BASE ===== */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
    }

    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* ===== HEADER ET TITRES ===== */
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }

    h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
        margin-top: 0.5rem;
    }

    h2 {
        font-size: 2rem !important;
        border-bottom: 3px solid var(--primary);
        padding-bottom: 10px;
        display: inline-block;
    }

    h3 {
        font-size: 1.5rem !important;
        color: var(--gray-800) !important;
        background: none !important;
        -webkit-text-fill-color: var(--gray-800) !important;
    }

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid var(--gray-200);
        box-shadow: 4px 0 15px rgba(0, 0, 0, 0.05);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }

    /* ===== BOUTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08);
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 7px 14px rgba(50, 50, 93, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08);
        background: linear-gradient(135deg, #5a67d8, #6b46c1);
    }

    .stButton > button:focus {
        outline: none;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3);
    }

    /* Bouton secondaire */
    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #edf2f7, #e2e8f0);
        color: var(--gray-700);
        border: 1px solid var(--gray-300);
    }

    .stButton > button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #e2e8f0, #cbd5e0);
        color: var(--gray-800);
    }

    /* ===== CARTES PRODUITS ===== */
    .product-card {
        border-radius: 16px;
        background: white;
        border: 1px solid var(--gray-200);
        padding: 15px;
        margin: 10px 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        height: 100%;
    }

    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        border-color: var(--primary);
    }

    /* ===== IMAGES ===== */
    .stImage {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        border: 1px solid var(--gray-200);
    }

    .stImage img {
        transition: transform 0.5s ease;
    }

    .stImage:hover img {
        transform: scale(1.05);
    }

    /* ===== INPUTS ET SELECTEURS ===== */
    .stSelectbox > div > div {
        border-radius: 8px !important;
        border: 2px solid var(--gray-300) !important;
        transition: all 0.3s ease;
    }

    .stSelectbox > div > div:hover {
        border-color: var(--primary) !important;
    }

    /* ===== PROGRESS BAR ===== */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        border-radius: 10px;
    }

    /* ===== METRICS ===== */
    .stMetric {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid var(--gray-200);
        text-align: center;
    }

    /* ===== ALERTES ET MESSAGES ===== */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }

    .stAlert > div {
        padding: 15px 20px;
    }

    /* ===== SÉPARATEURS ===== */
    hr {
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--primary), transparent);
        border: none;
        margin: 30px 0;
    }

    /* ===== BADGES OCCASIONS ===== */
    .occasion-badge {
        display: inline-block;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
        box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3);
    }

    .occasion-badge-secondary {
        background: linear-gradient(135deg, #48bb78, #38a169);
    }

    /* ===== GRID PRODUITS ===== */
    .product-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }

    /* ===== FOOTER ===== */
    .footer {
        text-align: center;
        padding: 30px 20px;
        margin-top: 50px;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 20px;
        border-top: 1px solid var(--gray-200);
    }

    /* ===== ANIMATIONS ===== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    .pulse {
        animation: pulse 2s infinite;
    }

    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        .main > div {
            padding: 10px !important;
        }
        
        h1 {
            font-size: 2rem !important;
        }
        
        h2 {
            font-size: 1.5rem !important;
        }
        
        .product-grid {
            grid-template-columns: 1fr;
        }
    }

    /* ===== SCROLLBAR PERSONNALISÉE ===== */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--gray-100);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(var(--primary), var(--secondary));
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(#5a67d8, #6b46c1);
    }

    /* ===== CARTE DE MORPHOLOGIE ===== */
    .morphology-card {
        background: white;
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        border: 1px solid var(--gray-200);
        transition: all 0.3s ease;
        height: 100%;
    }

    .morphology-card:hover {
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.12);
        transform: translateY(-5px);
    }

    /* ===== ICÔNES ===== */
    .icon-container {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 50px;
        height: 50px;
        border-radius: 12px;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        font-size: 24px;
        margin-right: 15px;
    }

    /* ===== MENU DÉROULANT OCCASIONS ===== */
    .occasion-selector {
        background: white;
        border-radius: 12px;
        padding: 15px;
        border: 2px solid var(--gray-200);
        transition: all 0.3s ease;
    }

    .occasion-selector:hover {
        border-color: var(--primary);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.1);
    }

    /* ===== STATISTIQUES ===== */
    .stat-card {
        background: linear-gradient(135deg, #ffffff, #f8fafc);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        border: 1px solid var(--gray-200);
        transition: all 0.3s ease;
    }

    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
    }

    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: block;
        line-height: 1;
    }

    .stat-label {
        font-size: 0.9rem;
        color: var(--gray-600);
        margin-top: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    /* ===== AMÉLIORATIONS SPÉCIFIQUES ===== */
    .camera-container {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        border: 2px solid var(--gray-200);
    }

    .upload-container {
        border: 2px dashed var(--primary);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        transition: all 0.3s ease;
    }

    .upload-container:hover {
        background: rgba(102, 126, 234, 0.1);
        border-color: var(--secondary);
    }

    .confidence-meter {
        position: relative;
        height: 10px;
        background: var(--gray-200);
        border-radius: 5px;
        overflow: hidden;
        margin: 10px 0;
    }

    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        border-radius: 5px;
        transition: width 1s ease-in-out;
    }

    .step-indicator {
        display: flex;
        justify-content: space-between;
        margin: 30px 0;
        position: relative;
    }

    .step {
        display: flex;
        flex-direction: column;
        align-items: center;
        position: relative;
        z-index: 2;
    }

    .step-number {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: white;
        border: 3px solid var(--gray-300);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: var(--gray-500);
        margin-bottom: 10px;
        transition: all 0.3s ease;
    }

    .step.active .step-number {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        border-color: var(--primary);
        color: white;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }

    .step-line {
        position: absolute;
        top: 20px;
        left: 10%;
        right: 10%;
        height: 3px;
        background: var(--gray-300);
        z-index: 1;
    }

    .step.active ~ .step .step-number {
        background: var(--gray-200);
        border-color: var(--gray-400);
    }

    .price-tag {
        display: inline-block;
        background: linear-gradient(135deg, #48bb78, #38a169);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
    }

    /* ===== OVERRIDES STREAMLIT ===== */
    div[data-testid="stHorizontalBlock"] {
        gap: 20px;
    }

    .stCameraInput > div {
        border-radius: 16px !important;
        overflow: hidden !important;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1) !important;
        border: 2px solid var(--gray-200) !important;
    }
</style>
""", unsafe_allow_html=True)

# Titre et description
st.markdown("""
<div style="text-align: center; padding: 20px 0 40px 0;">
    <h1>👗 Détecteur de Morphologie & Recommandations Mode</h1>
    <p style="font-size: 1.2rem; color: var(--gray-600); max-width: 800px; margin: 0 auto;">
        <strong>📸 Capturez votre photo</strong> → <strong>🎯 Détectez votre morphologie</strong> → <strong>🛍️ Recevez des recommandations personnalisées</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# Indicateur d'étapes
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

st.markdown("""
<div class="step-indicator">
    <div class="step-line"></div>
    <div class="step active">
        <div class="step-number">1</div>
        <div style="font-weight: 600; color: var(--gray-700);">Capture</div>
    </div>
    <div class="step" id="step2">
        <div class="step-number">2</div>
        <div style="font-weight: 600; color: var(--gray-700);">Analyse</div>
    </div>
    <div class="step" id="step3">
        <div class="step-number">3</div>
        <div style="font-weight: 600; color: var(--gray-700);">Recommandations</div>
    </div>
</div>

<script>
    if (window.location.hash.includes("analysis_done")) {
        document.getElementById("step2").classList.add("active");
        document.getElementById("step3").classList.add("active");
    }
</script>
""", unsafe_allow_html=True)

# Charger le modèle de classification de morphologie
@st.cache_resource
def load_classifier_model():
    """Charger le modèle de classification de morphologie"""
    try:
        model = tf.keras.models.load_model('/content/body_type_classifier_model.keras')
        return model
    except Exception as e:
        st.error(f"❌ Erreur de chargement du modèle: {e}")
        return None

# Charger les données des produits avec occasions
@st.cache_data
def load_product_data():
    """Charger les données des produits depuis le CSV"""
    try:
        csv_path = '/content/zen_products_with_morphology_and_occasion.csv'
        df = pd.read_csv(csv_path)
        
        # Nettoyage des données
        if 'morphology' in df.columns:
            df['morphology'] = df['morphology'].str.upper().str.strip()
        
        # Liste RESTREINTE des occasions disponibles (seulement celles que vous voulez)
        all_occasions = [
            "Toutes occasions",  # Option par défaut
            "professionnel",     # Tenue professionnelle
            "casual",           # Décontracté
            "sport",            # Sport
            "soiree",           # Soirée
            "chic",             # Chic/élégant
            "plage",            # Plage/vacances
        ]
        
        # Gestion des occasions - seulement extraire les occasions qui sont dans notre liste
        if 'occasion' in df.columns:
            # Remplir les valeurs manquantes
            df['occasion'] = df['occasion'].fillna('').str.strip()
            
            # Créer une liste d'occasions pour chaque produit, mais uniquement celles qui sont dans notre liste
            df['occasion_list'] = df['occasion'].apply(
                lambda x: [
                    o.strip().lower() for o in str(x).split(',') 
                    if o.strip().lower() in [occ.lower() for occ in all_occasions[1:]]  # Exclure "Toutes occasions"
                ] if x and str(x).strip() else []
            )
        
        else:
            df['occasion'] = ''
            df['occasion_list'] = [[] for _ in range(len(df))]
        
        st.sidebar.success(f"✅ {len(df)} produits chargés")
        
        # Afficher les occasions disponibles dans la sidebar
        st.sidebar.markdown("""
        <div style="background: white; padding: 20px; border-radius: 12px; border: 1px solid var(--gray-200);">
            <h3 style="color: var(--gray-800); margin-bottom: 15px;">🎪 Occasions disponibles</h3>
        """, unsafe_allow_html=True)
        
        for occ in all_occasions:
            if occ == "Toutes occasions":
                st.sidebar.markdown(f'<div style="padding: 8px 12px; background: var(--gray-100); border-radius: 8px; margin: 5px 0;"><strong>• {occ}</strong> (par défaut)</div>', unsafe_allow_html=True)
            else:
                st.sidebar.markdown(f'<div style="padding: 8px 12px; background: white; border-radius: 8px; margin: 5px 0; border-left: 4px solid var(--primary);">• {occ}</div>', unsafe_allow_html=True)
        
        st.sidebar.markdown("</div>", unsafe_allow_html=True)
        
        return df, all_occasions
        
    except Exception as e:
        st.error(f"❌ Erreur de chargement des données: {e}")
        return pd.DataFrame(), []

# Labels des morphologies
BODY_TYPES = ['APPLE', 'PEAR', 'RECTANGLE', 'HOURGLASS', 'INVERTED TRIANGLE']
BODY_TYPES_DISPLAY = {
    'APPLE': "APPLE (Pomme) 🍎",
    'PEAR': "PEAR (Poire) 🍐", 
    'RECTANGLE': "RECTANGLE (Rectangle) ⬜",
    'HOURGLASS': "HOURGLASS (Sablier) ⏳",
    'INVERTED TRIANGLE': "INVERTED TRIANGLE (Triangle inversé) 🔺"
}

# Fonction pour charger les images
def load_product_image(image_path):
    """Charger une image de produit"""
    try:
        if pd.isna(image_path) or not image_path:
            return None
        
        # Si c'est une URL
        if isinstance(image_path, str) and image_path.startswith('http'):
            try:
                response = requests.get(image_path, timeout=5)
                if response.status_code == 200:
                    return Image.open(BytesIO(response.content))
            except:
                return None
        
        # Si c'est un chemin local
        if isinstance(image_path, str):
            clean_path = image_path.strip()
            
            # Prendre la première image si plusieurs
            if ',' in clean_path:
                clean_path = clean_path.split(',')[0].strip()
            
            # Essayer différents chemins
            possible_paths = [
                clean_path,
                f'/content/{clean_path}',
                f'/content/product_images/{os.path.basename(clean_path)}',
                f'product_images/{os.path.basename(clean_path)}'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    return Image.open(path)
        
        return None
    except:
        return None

# Fonction pour obtenir les produits recommandés - MODIFIÉE
def get_recommended_products(morphology_type, selected_occasion, product_df, num_recommendations=9):
    """Obtenir les produits recommandés filtrés par morphologie et occasion"""
    if product_df.empty:
        return []
    
    # Filtrer par morphologie
    filtered = product_df[product_df['morphology'] == morphology_type.upper()].copy()
    
    if filtered.empty:
        # Si aucun produit pour cette morphologie, prendre tous les produits
        filtered = product_df.copy()
    
    # IMPORTANT: Toujours retourner des produits même si l'occasion n'est pas trouvée
    if selected_occasion and selected_occasion != "Toutes occasions":
        # Normaliser l'occasion pour la comparaison
        selected_occasion_lower = selected_occasion.lower()
        
        # Filtrer les produits qui ont cette occasion dans leur liste
        filtered_by_occasion = filtered[
            filtered['occasion_list'].apply(
                lambda occasions: any(
                    occ.lower() == selected_occasion_lower 
                    for occ in occasions
                ) if isinstance(occasions, list) else False
            )
        ]
        
        # Si on trouve des produits avec cette occasion, les utiliser
        if not filtered_by_occasion.empty:
            filtered = filtered_by_occasion
    
    # Si toujours vide (normalement pas possible maintenant), prendre tous les produits
    if filtered.empty:
        filtered = product_df.copy()
    
    # Prendre un échantillon
    sample = filtered.sample(min(num_recommendations, len(filtered)))
    
    # Préparer les produits
    products = []
    for _, row in sample.iterrows():
        # Chercher une image
        image_path = None
        for col in ['main_image', 'color_images', 'image_url']:
            if col in row and pd.notna(row[col]):
                image_path = row[col]
                break
        
        # Obtenir les occasions du produit (seulement celles dans notre liste restreinte)
        product_occasions = []
        if 'occasion_list' in row and isinstance(row['occasion_list'], list):
            product_occasions = [occ.capitalize() for occ in row['occasion_list']]
        
        product = {
            'name': str(row.get('name', 'Produit')),
            'category': str(row.get('category', 'Non spécifié')),
            'price': str(row.get('price', 'N/A')),
            'url': str(row.get('url', '#')),
            'image_path': image_path,
            'occasions': product_occasions,
            'image': load_product_image(image_path)
        }
        products.append(product)
    
    return products

# Charger les modèles et données
classifier_model = load_classifier_model()
product_df, all_occasions = load_product_data()

# S'assurer que "Toutes occasions" est la première option
if "Toutes occasions" not in all_occasions:
    all_occasions = ["Toutes occasions"] + all_occasions
else:
    # Déplacer "Toutes occasions" en première position
    all_occasions.remove("Toutes occasions")
    all_occasions = ["Toutes occasions"] + all_occasions

# Initialisation de session state
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'selected_occasion' not in st.session_state:
    st.session_state.selected_occasion = 'Toutes occasions'
if 'recommended_products' not in st.session_state:
    st.session_state.recommended_products = []
if 'all_occasions' not in st.session_state:
    st.session_state.all_occasions = all_occasions

# Interface principale
st.sidebar.header("⚙️ Paramètres")

# Section 1: Capture d'image
st.markdown("""
<div class="fade-in">
    <h2>📸 Étape 1: Capturez votre photo</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Choix de la méthode
    capture_option = st.radio(
        "Méthode de capture:",
        ["📷 Utiliser la caméra", "📤 Télécharger une image"],
        horizontal=True
    )
    
    captured_image = None
    
    if capture_option == "📷 Utiliser la caméra":
        st.markdown('<div class="camera-container">', unsafe_allow_html=True)
        captured_image = st.camera_input("Prenez une photo", key="camera")
        st.markdown('</div>', unsafe_allow_html=True)
        if captured_image:
            st.success("✅ Photo capturée! Cliquez sur 'Analyser' pour continuer")
    else:
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        captured_image = st.file_uploader(
            "Choisissez une image",
            type=['jpg', 'jpeg', 'png'],
            key="upload"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        if captured_image:
            st.success("✅ Image téléchargée! Cliquez sur 'Analyser' pour continuer")

with col2:
    # Affichage de l'image capturée
    if captured_image:
        try:
            image = Image.open(captured_image).convert('RGB')
            st.markdown('<div class="morphology-card">', unsafe_allow_html=True)
            st.image(image, caption="Votre image", use_column_width=True)
            
            # Bouton d'analyse
            if st.button("🔍 Analyser la morphologie", type="primary", use_container_width=True):
                if classifier_model:
                    with st.spinner("Analyse en cours..."):
                        # Prétraitement
                        img_resized = image.resize((224, 224))
                        img_array = np.array(img_resized) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        # Prédiction
                        predictions = classifier_model.predict(img_array, verbose=0)
                        predicted_idx = np.argmax(predictions[0])
                        confidence = predictions[0][predicted_idx] * 100
                        
                        # Morphologie détectée
                        if predicted_idx < len(BODY_TYPES):
                            morphology = BODY_TYPES[predicted_idx]
                            morphology_display = BODY_TYPES_DISPLAY.get(morphology, morphology)
                            
                            # Stocker les résultats
                            st.session_state.morphology = morphology
                            st.session_state.morphology_display = morphology_display
                            st.session_state.confidence = confidence
                            st.session_state.predictions = predictions[0]
                            st.session_state.analysis_done = True
                            st.session_state.original_image = image
                            
                            # Obtenir les produits initiaux
                            products = get_recommended_products(
                                morphology, 
                                'Toutes occasions', 
                                product_df, 
                                9
                            )
                            st.session_state.recommended_products = products
                            
                            # Mettre à jour l'étape 2
                            st.markdown("""
                            <script>
                                document.getElementById("step2").classList.add("active");
                            </script>
                            """, unsafe_allow_html=True)
                            
                            st.rerun()
                            
                        else:
                            st.error("❌ Erreur dans la prédiction")
                else:
                    st.error("❌ Modèle non chargé")
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Erreur: {e}")

# Section 2: Résultats et sélection d'occasion (si analyse faite)
if st.session_state.analysis_done:
    st.markdown("""
    <script>
        document.getElementById("step3").classList.add("active");
    </script>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="fade-in">
        <h2>🎯 Étape 2: Résultats d'analyse & Sélection d'occasion</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Affichage des résultats
    col_res1, col_res2, col_res3 = st.columns(3)
    
    with col_res1:
        st.markdown('<div class="morphology-card">', unsafe_allow_html=True)
        st.subheader("📊 Votre morphologie")
        st.markdown(f"### {st.session_state.morphology_display}")
        
        # Barre de progression personnalisée
        st.markdown(f"""
        <div style="margin: 20px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span>Confiance</span>
                <span><strong>{st.session_state.confidence:.1f}%</strong></span>
            </div>
            <div class="confidence-meter">
                <div class="confidence-fill" style="width: {st.session_state.confidence}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_res2:
        st.markdown('<div class="morphology-card">', unsafe_allow_html=True)
        st.subheader("📈 Répartition")
        # Graphique des probabilités
        prob_data = {}
        for i, morph in enumerate(BODY_TYPES):
            prob = st.session_state.predictions[i] * 100 if i < len(st.session_state.predictions) else 0
            prob_data[morph] = prob
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(prob_data.keys()),
                y=list(prob_data.values()),
                marker_color=['#FF6B6B' if k == st.session_state.morphology else '#4ECDC4' for k in prob_data.keys()],
                text=[f'{v:.1f}%' for v in prob_data.values()],
                textposition='auto',
            )
        ])
        fig.update_layout(
            height=200, 
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor='white',
            paper_bgcolor='white',
            yaxis=dict(title='Probabilité (%)', gridcolor='var(--gray-200)'),
            xaxis=dict(title='Morphologie')
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_res3:
        st.markdown('<div class="morphology-card">', unsafe_allow_html=True)
        st.subheader("💡 Conseils de style")
        conseils = {
            'APPLE': "Privilégiez les V-neck et robes trapèze pour équilibrer votre silhouette",
            'PEAR': "Mettez l'accent sur le haut du corps avec des couleurs vives et des motifs",
            'RECTANGLE': "Créez des courbes avec des ceintures et des vêtements structurés",
            'HOURGLASS': "Soulignez votre taille fine avec des ceintures et des coupes cintrées",
            'INVERTED TRIANGLE': "Équilibrez avec des bas sombres et des hauts plus simples"
        }
        tips = conseils.get(st.session_state.morphology, "Choisissez des vêtements adaptés à votre silhouette pour mettre en valeur vos atouts.")
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); 
                    padding: 20px; border-radius: 12px; border-left: 4px solid var(--primary);">
            <p style="color: var(--gray-700); font-size: 14px; line-height: 1.6; margin: 0;">{tips}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sélecteur d'occasion
    st.markdown("""
    <div style="margin: 40px 0 20px 0;">
        <h3>🎪 Choisissez une occasion pour vos recommandations</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Utiliser un selectbox avec seulement les occasions spécifiées
    selected_occasion = st.selectbox(
        "Sélectionnez une occasion:",
        all_occasions,
        index=0,  # "Toutes occasions" par défaut
        key="occasion_selector",
        help="Choisissez une occasion pour filtrer les recommandations"
    )
    
    # Si l'occasion a changé, recalculer les produits
    if selected_occasion != st.session_state.selected_occasion:
        st.session_state.selected_occasion = selected_occasion
        
        # Recalculer les produits
        products = get_recommended_products(
            st.session_state.morphology,
            selected_occasion if selected_occasion != "Toutes occasions" else None,
            product_df,
            9
        )
        st.session_state.recommended_products = products
        st.rerun()
    
    # Afficher l'occasion sélectionnée avec le nombre de produits disponibles
    if selected_occasion != "Toutes occasions":
        # Compter les produits pour cette morphologie et cette occasion
        filtered_count = len(product_df[
            (product_df['morphology'] == st.session_state.morphology.upper()) &
            (product_df['occasion_list'].apply(
                lambda occasions: selected_occasion.lower() in [occ.lower() for occ in occasions]
                if isinstance(occasions, list) else False
            ))
        ])
        
        # TOTAL des produits pour cette morphologie
        total_for_morph = len(product_df[product_df['morphology'] == st.session_state.morphology.upper()])
        
        # Afficher le message approprié
        if filtered_count > 0:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #48bb78, #38a169); color: white; 
                        padding: 15px 20px; border-radius: 12px; margin: 10px 0;">
                <strong>Occasion sélectionnée:</strong> {selected_attention} 
                <span style="float: right; background: white; color: #38a169; padding: 2px 10px; border-radius: 20px;">
                    {filtered_count} produits trouvés
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ed8936, #dd6b20); color: white; 
                        padding: 15px 20px; border-radius: 12px; margin: 10px 0;">
                <strong>⚠️ Aucun produit trouvé pour '{selected_occasion}'</strong>
                <p style="margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;">
                    Affichage de {min(9, total_for_morph)} produits adaptés à votre morphologie
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        total_for_morph = len(product_df[product_df['morphology'] == st.session_state.morphology.upper()])
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, var(--primary), var(--secondary)); color: white; 
                    padding: 15px 20px; border-radius: 12px; margin: 10px 0;">
            <strong>Toutes occasions sélectionnées</strong>
            <span style="float: right; background: white; color: var(--primary); padding: 2px 10px; border-radius: 20px;">
                {total_for_morph} produits disponibles
            </span>
        </div>
        """, unsafe_allow_html=True)

# Section 3: Produits recommandés
if st.session_state.analysis_done and st.session_state.recommended_products:
    st.markdown("""
    <div class="fade-in">
        <h2>🛍️ Étape 3: Produits Recommandés</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistiques
    if st.session_state.selected_occasion != "Toutes occasions":
        # Compter les produits qui ont réellement cette occasion
        exact_match_count = len(product_df[
            (product_df['morphology'] == st.session_state.morphology.upper()) &
            (product_df['occasion_list'].apply(
                lambda occasions: st.session_state.selected_occasion.lower() in [occ.lower() for occ in occasions]
                if isinstance(occasions, list) else False
            ))
        ])
        
        total_for_morph = len(product_df[product_df['morphology'] == st.session_state.morphology.upper()])
        
        if exact_match_count > 0:
            st.info(f"📊 {len(st.session_state.recommended_products)} produits affichés sur {exact_match_count} disponibles pour '{st.session_state.selected_occasion}'")
        else:
            st.info(f"📊 {len(st.session_state.recommended_products)} produits affichés (tous adaptés à votre morphologie)")
    else:
        total_for_morph = len(product_df[product_df['morphology'] == st.session_state.morphology.upper()])
        st.info(f"📊 {len(st.session_state.recommended_products)} produits affichés sur {total_for_morph} disponibles")
    
    # Affichage des produits en grille CSS
    st.markdown('<div class="product-grid">', unsafe_allow_html=True)
    
    for product in st.session_state.recommended_products:
        st.markdown('<div class="product-card fade-in">', unsafe_allow_html=True)
        
        # Image du produit
        if product['image']:
            img = product['image'].copy()
            img.thumbnail((250, 250))
            st.image(img, use_column_width=True)
        else:
            # Placeholder avec icône
            placeholder = Image.new('RGB', (250, 250), color='#f0f0f0')
            st.image(placeholder, use_column_width=True)
            st.markdown('<div style="text-align: center; color: var(--gray-500); font-size: 14px; margin: 10px 0;">Image non disponible</div>', unsafe_allow_html=True)
        
        # Informations
        st.markdown(f"**{product['name']}**")
        st.markdown(f'<div style="color: var(--gray-600); font-size: 14px; margin: 5px 0;">📂 {product["category"]}</div>', unsafe_allow_html=True)
        
        # Afficher les occasions du produit
        if product['occasions']:
            st.markdown('<div style="margin: 10px 0;">', unsafe_allow_html=True)
            for occ in product['occasions'][:3]:  # Max 3 badges
                st.markdown(f'<span class="occasion-badge">{occ}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Prix
        if product['price'] != 'N/A':
            st.markdown(f'<div style="margin: 10px 0;"><span class="price-tag">{product["price"]}</span></div>', unsafe_allow_html=True)
        
        # Bouton pour voir le produit
        if product['url'] and product['url'] != '#':
            st.markdown(f"""
            <a href="{product['url']}" target="_blank" style="
                display: inline-block;
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                color: white;
                padding: 8px 16px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: 600;
                font-size: 14px;
                text-align: center;
                width: 100%;
                transition: all 0.3s ease;
                margin-top: 10px;
            ">
                🔗 Voir le produit
            </a>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Bouton pour réinitialiser
    col_reset1, col_reset2, col_reset3 = st.columns([1, 2, 1])
    with col_reset2:
        if st.button("🔄 Nouvelle analyse", type="secondary", use_container_width=True):
            for key in ['analysis_done', 'morphology', 'morphology_display', 'confidence', 
                       'predictions', 'selected_occasion', 'recommended_products', 'original_image']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# Section d'instructions si aucune analyse
elif not st.session_state.analysis_done:
    st.markdown("""
    <div class="fade-in">
        <h2>📋 Comment utiliser cette application</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col_inst1, col_inst2, col_inst3 = st.columns(3)
    
    with col_inst1:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size: 40px; margin-bottom: 10px;">📸</div>
        <h3>1. Capture</h3>
        <ul style="text-align: left; color: var(--gray-600); padding-left: 20px;">
            <li>Utilisez la caméra ou téléchargez une photo</li>
            <li>Photo de préférence en tenue près du corps</li>
            <li>Bonne luminosité pour une meilleure analyse</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_inst2:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size: 40px; margin-bottom: 10px;">🎯</div>
        <h3>2. Analyse</h3>
        <ul style="text-align: left; color: var(--gray-600); padding-left: 20px;">
            <li>Cliquez sur "Analyser la morphologie"</li>
            <li>L'IA détecte votre type de silhouette</li>
            <li>Visualisez les résultats avec graphiques</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_inst3:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size: 40px; margin-bottom: 10px;">🛍️</div>
        <h3>3. Recommandations</h3>
        <ul style="text-align: left; color: var(--gray-600); padding-left: 20px;">
            <li>Choisissez une occasion spécifique</li>
            <li>Recevez des produits adaptés</li>
            <li>Cliquez sur les liens pour voir les produits</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <div style="max-width: 800px; margin: 0 auto;">
        <h3 style="color: var(--gray-800); margin-bottom: 20px;">👗 Recommandation Mode Intelligente</h3>
        <p style="color: var(--gray-600); font-size: 16px; line-height: 1.6; margin-bottom: 20px;">
            <strong>Morphologie + Occasion = Personnalisation parfaite</strong><br>
            Utilise l'IA pour analyser votre silhouette et recommander des vêtements adaptés à votre morphologie et à l'occasion
        </p>
        <div style="display: flex; justify-content: center; gap: 20px; margin-top: 20px;">
            <div style="text-align: center;">
                <div style="font-size: 24px; color: var(--primary);">🎯</div>
                <div style="font-size: 12px; color: var(--gray-600);">Précision IA</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 24px; color: var(--primary);">👗</div>
                <div style="font-size: 12px; color: var(--gray-600);">Morphologie</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 24px; color: var(--primary);">🎪</div>
                <div style="font-size: 12px; color: var(--gray-600);">Occasion</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 24px; color: var(--primary);">🛍️</div>
                <div style="font-size: 12px; color: var(--gray-600);">Recommandations</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
