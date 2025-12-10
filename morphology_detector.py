
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import pandas as pd
import time
import os
import requests
from io import BytesIO

# Configuration de la page
st.set_page_config(
    page_title="Détecteur de Morphologie + Recommandations",
    page_icon="👗",
    layout="wide"
)

# Titre et description
st.title("👗 Détecteur de Morphologie & Recommandations Intelligentes")
st.markdown("""
### Analysez votre morphologie et obtenez des recommandations personnalisées selon l'occasion
**Instructions:**
1. Capturez votre photo
2. Analysez votre morphologie
3. Choisissez une occasion
4. Recevez des recommandations personnalisées
""")

# Charger le modèle de classification de morphologie
@st.cache_resource
def load_classifier_model():
    """Charger le modèle de classification de morphologie"""
    try:
        model = tf.keras.models.load_model('/content/body_type_classifier_model.keras')
        st.success("✅ Modèle de classification chargé avec succès!")
        return model
    except Exception as e:
        st.error(f"❌ Erreur de chargement du modèle de classification: {e}")
        return None

# Charger les données des produits avec occasions
@st.cache_data
def load_product_data():
    """Charger les données des produits depuis le CSV"""
    try:
        # Chemin vers votre fichier CSV
        csv_path = '/content/zen_products_with_morphology_and_occasion.csv'
        df = pd.read_csv(csv_path)
        
        # Vérifier et nettoyer les colonnes
        if 'morphology' in df.columns:
            df['morphology'] = df['morphology'].str.upper().str.strip()
        
        # Nettoyer la colonne occasion
        if 'occasion' in df.columns:
            df['occasion'] = df['occasion'].str.strip()
            # Supprimer les lignes où occasion est NaN
            df = df.dropna(subset=['occasion'])
            # Diviser les occasions multiples séparées par des virgules
            df['occasion_list'] = df['occasion'].apply(lambda x: [o.strip() for o in str(x).split(',')])
        else:
            df['occasion_list'] = [[] for _ in range(len(df))]
            st.warning("⚠️ Colonne 'occasion' non trouvée dans le CSV")
        
        st.success(f"✅ {len(df)} produits chargés avec succès!")
        
        # Debug info dans la sidebar
        st.sidebar.info(f"Produits chargés: {len(df)}")
        if 'occasion' in df.columns:
            occasions_count = df['occasion'].str.split(',').explode().str.strip().value_counts()
            st.sidebar.info("Occasions disponibles:")
            for occ, count in occasions_count.head(10).items():
                st.sidebar.caption(f"• {occ}: {count}")
        
        return df
    except Exception as e:
        st.error(f"❌ Erreur de chargement des données produits: {e}")
        return pd.DataFrame()

# Labels des morphologies
BODY_TYPES = {
    0: "APPLE",
    1: "PEAR",
    2: "RECTANGLE",
    3: "HOURGLASS",
    4: "INVERTED TRIANGLE"
}

BODY_TYPES_DISPLAY = {
    0: "APPLE (Pomme) 🍎",
    1: "PEAR (Poire) 🍐",
    2: "RECTANGLE (Rectangle) ⬜",
    3: "HOURGLASS (Sablier) ⏳",
    4: "INVERTED TRIANGLE (Triangle inversé) 🔺"
}

DESCRIPTIONS = {
    "APPLE": "Épaules et buste larges, taille peu marquée",
    "PEAR": "Hanches plus larges que les épaules",
    "RECTANGLE": "Silhouette droite, peu de courbes",
    "HOURGLASS": "Épaules et hanches alignées, taille fine",
    "INVERTED TRIANGLE": "Épaules larges, hanches étroites"
}

STYLE_TIPS = {
    "APPLE": "Privilégiez les V-neck, robes trapèze, et hauts fluides",
    "PEAR": "Mettez l'accent sur le haut avec des tops structurés",
    "RECTANGLE": "Créez des courbes avec des ceintures et des coupes cintrées",
    "HOURGLASS": "Soulignez votre taille avec des vêtements ajustés",
    "INVERTED TRIANGLE": "Équilibrez avec des jupes évasées et des bas sombres"
}

# Fonction pour charger les images des produits
def load_product_images(image_path):
    """Charger une image de produit depuis un chemin ou une URL"""
    try:
        # Si le chemin est vide
        if pd.isna(image_path) or not image_path or str(image_path).strip() == '':
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
            # Nettoyer le chemin
            clean_path = image_path.strip()
            
            # Si c'est une liste d'images séparées par des virgules, prendre la première
            if ',' in clean_path:
                clean_path = clean_path.split(',')[0].strip()
            
            # Essayer différents chemins possibles
            possible_paths = [
                clean_path,
                f'/content/{clean_path}',
                f'/content/product_images/{os.path.basename(clean_path)}',
                f'product_images/{os.path.basename(clean_path)}',
                os.path.join('/content/product_images', os.path.basename(clean_path))
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    return Image.open(path)
        
        return None
    except Exception as e:
        print(f"Erreur de chargement d'image: {e}, chemin: {image_path}")
        return None

# Fonction pour obtenir les produits recommandés AVEC FILTRE OCCASION
def get_recommended_products(morphology_type, selected_occasion, product_df, num_recommendations=6):
    """Obtenir les produits recommandés pour une morphologie et occasion données"""
    if product_df.empty:
        return []
    
    # Filtrer par morphologie
    morphology_upper = morphology_type.upper()
    filtered_products = product_df[product_df['morphology'] == morphology_upper].copy()
    
    # Filtrer par occasion si une occasion est sélectionnée
    if selected_occasion and selected_occasion != "Toutes les occasions":
        # Filtrer les produits qui contiennent l'occasion sélectionnée dans leur liste
        filtered_products = filtered_products[
            filtered_products['occasion_list'].apply(
                lambda occasions: selected_occasion in [str(o).strip() for o in occasions]
            )
        ]
    
    if filtered_products.empty:
        # Si aucun produit trouvé, retourner un échantillon aléatoire de produits de la même morphologie
        filtered_products = product_df[product_df['morphology'] == morphology_upper]
        if filtered_products.empty:
            filtered_products = product_df
    
    # Prendre un échantillon
    sample_products = filtered_products.sample(min(num_recommendations, len(filtered_products)))
    
    # Préparer les données des produits
    recommended_products = []
    for _, product in sample_products.iterrows():
        # Obtenir la première image disponible
        image_path = None
        for col in ['main_image', 'color_images', 'image_url']:
            if col in product and pd.notna(product[col]):
                image_path = product[col]
                break
        
        product_dict = {
            'name': str(product.get('name', f'Produit {len(recommended_products)+1}')),
            'category': str(product.get('category', 'Non spécifié')),
            'price': str(product.get('price', 'N/A')),
            'url': str(product.get('url', '#')),
            'image_path': image_path,
            'occasion': str(product.get('occasion', 'Non spécifié')) if 'occasion' in product else 'Non spécifié'
        }
        
        # Charger l'image
        product_dict['image'] = load_product_images(product_dict['image_path'])
        
        recommended_products.append(product_dict)
    
    return recommended_products

# Fonction pour obtenir toutes les occasions disponibles
def get_all_occasions(product_df):
    """Obtenir toutes les occasions uniques disponibles"""
    if product_df.empty or 'occasion_list' not in product_df.columns:
        return ["Toutes les occasions"]
    
    all_occasions = set()
    for occasions_list in product_df['occasion_list']:
        for occasion in occasions_list:
            if occasion and str(occasion).strip():
                all_occasions.add(str(occasion).strip())
    
    occasions_list = ["Toutes les occasions"] + sorted(list(all_occasions))
    return occasions_list

# Charger les modèles et données
classifier_model = load_classifier_model()
product_df = load_product_data()

# Obtenir toutes les occasions disponibles
all_occasions = get_all_occasions(product_df)

# Interface principale
st.sidebar.header("⚙️ Paramètres")

# Section capture d'image
st.header("📸 Étape 1: Capturez votre image")

# Option de capture
capture_method = st.radio(
    "Choisissez votre méthode:",
    ["Utiliser la caméra", "Télécharger une image"],
    horizontal=True
)

captured_image = None

if capture_method == "Utiliser la caméra":
    st.info("Positionnez-vous bien dans le cadre")
    captured_image = st.camera_input("Prenez une photo", key="camera")
else:
    st.info("Téléchargez une photo de vous en tenue près du corps")
    captured_image = st.file_uploader(
        "Choisissez une image",
        type=['jpg', 'jpeg', 'png'],
        key="upload"
    )

# Section pour la sélection d'occasion (affichée après l'analyse)
if 'prediction' in st.session_state:
    st.header("🎯 Étape 2: Choisissez une occasion")
    
    # Sélecteur d'occasion
    selected_occasion = st.selectbox(
        "Pour quelle occasion cherchez-vous des vêtements?",
        all_occasions,
        key="occasion_selector"
    )
    
    # Stocker l'occasion sélectionnée
    st.session_state['selected_occasion'] = selected_occasion
    
    # Bouton pour mettre à jour les recommandations
    if st.button("🔄 Mettre à jour les recommandations", type="secondary"):
        # Recalculer les recommandations avec la nouvelle occasion
        pred = st.session_state['prediction']
        recommended_products = get_recommended_products(
            pred['simple_type'], 
            selected_occasion, 
            product_df, 
            num_recommendations=6
        )
        st.session_state['recommended_products'] = recommended_products
        st.success(f"✅ Recommandations mises à jour pour: {selected_occasion}")

# Traitement de l'image
if captured_image is not None:
    try:
        # Ouvrir l'image
        original_image = Image.open(captured_image).convert('RGB')
        
        # Afficher l'image originale
        col_img, col_info = st.columns([2, 1])
        
        with col_img:
            st.subheader("🖼️ Votre image:")
            st.image(original_image, use_column_width=True, caption="Image à analyser")
        
        with col_info:
            # Bouton d'analyse
            if st.button("🔍 Analyser la morphologie", type="primary", use_container_width=True):
                if classifier_model is not None:
                    with st.spinner("🔍 Analyse en cours..."):
                        # Prétraiter l'image pour le classificateur
                        from PIL import Image
                        import numpy as np
                        
                        # Redimensionner et normaliser l'image
                        processed_image = original_image.resize((224, 224))
                        img_array = np.array(processed_image) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        # Faire la prédiction avec le classificateur
                        predictions = classifier_model.predict(img_array, verbose=0)
                        
                        # Obtenir la classe prédite
                        predicted_class = np.argmax(predictions[0])
                        confidence = predictions[0][predicted_class] * 100
                        
                        # Récupérer le nom de la morphologie
                        body_type_key = BODY_TYPES[predicted_class]
                        body_type_display = BODY_TYPES_DISPLAY[predicted_class]
                        
                        # Obtenir les produits recommandés (sans filtre d'occasion initial)
                        recommended_products = get_recommended_products(
                            body_type_key, 
                            None,  # Pas de filtre d'occasion initial
                            product_df, 
                            num_recommendations=6
                        )
                        
                        # Stocker dans la session
                        st.session_state['prediction'] = {
                            'body_type': body_type_display,
                            'simple_type': body_type_key,
                            'confidence': confidence,
                            'predictions': predictions[0],
                            'original_image': original_image
                        }
                        
                        # Stocker les produits recommandés initialement
                        st.session_state['recommended_products'] = recommended_products
                        
                        # Réinitialiser l'occasion sélectionnée
                        st.session_state['selected_occasion'] = "Toutes les occasions"
                        
                        st.success("✅ Analyse terminée!")
                        st.rerun()
                else:
                    st.error("❌ Modèle non chargé. Vérifiez le fichier de modèle.")

    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")

# Section des résultats
if 'prediction' in st.session_state:
    pred = st.session_state['prediction']
    selected_occasion = st.session_state.get('selected_occasion', 'Toutes les occasions')
    
    st.header("📊 Résultats de l'analyse")
    
    # Afficher les résultats dans des colonnes
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # Affichage principal
        st.markdown(f"### {pred['body_type']}")
        
        # Jauge de confiance
        st.metric("Confiance", f"{pred['confidence']:.1f}%")
        st.progress(float(pred['confidence'] / 100))
        
        # Description
        st.subheader("📝 Description")
        st.info(DESCRIPTIONS.get(pred['simple_type'], "Description non disponible"))
    
    with col2:
        # Conseils de style
        st.subheader("💡 Conseils de style")
        st.success(STYLE_TIPS.get(pred['simple_type'], "Conseils généraux de style"))
        
        # Occasion sélectionnée
        st.subheader("🎯 Occasion choisie")
        st.success(f"**{selected_occasion}**")
    
    with col3:
        # Graphique des probabilités
        st.subheader("📈 Probabilités par type")
        
        # Créer un graphique simple
        prob_data = []
        for i, (key, value) in enumerate(BODY_TYPES_DISPLAY.items()):
            prob = pred['predictions'][i] * 100
            simple = value.split(" ")[0]
            prob_data.append({"Type": simple, "Probabilité": prob})
        
        # Créer le graphique
        df = pd.DataFrame(prob_data)
        st.bar_chart(df.set_index("Type"))
        
        # Bouton pour réinitialiser
        if st.button("🔄 Nouvelle analyse", use_container_width=True):
            for key in ['prediction', 'recommended_products', 'selected_occasion']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# Section des produits recommandés
if 'prediction' in st.session_state and 'recommended_products' in st.session_state:
    pred = st.session_state['prediction']
    recommended_products = st.session_state['recommended_products']
    selected_occasion = st.session_state.get('selected_occasion', 'Toutes les occasions')
    
    st.header(f"🛍️ Produits Recommandés")
    st.markdown(f"**Morphologie:** {pred['simple_type']} | **Occasion:** {selected_occasion}")
    
    if recommended_products:
        # Statistiques
        total_products = len(product_df)
        morphology_count = len(product_df[product_df['morphology'] == pred['simple_type'].upper()])
        
        if selected_occasion != "Toutes les occasions":
            # Compter les produits pour cette morphologie ET cette occasion
            filtered_count = len(product_df[
                (product_df['morphology'] == pred['simple_type'].upper()) &
                (product_df['occasion_list'].apply(
                    lambda occasions: selected_occasion in [str(o).strip() for o in occasions]
                ))
            ])
            st.success(f"✅ {len(recommended_products)} produits sur {filtered_count} disponibles")
        else:
            st.success(f"✅ {len(recommended_products)} produits sur {morphology_count} disponibles")
        
        # Afficher les produits dans une grille
        cols = st.columns(3)
        
        for i, product in enumerate(recommended_products):
            with cols[i % 3]:
                with st.container():
                    st.markdown("---")
                    
                    # Afficher l'image du produit
                    if product.get('image'):
                        try:
                            # Redimensionner l'image pour l'affichage
                            img = product['image'].copy()
                            img.thumbnail((300, 300))
                            st.image(img, use_column_width=True)
                        except Exception as e:
                            st.warning("⚠️ Erreur d'affichage de l'image")
                    else:
                        # Afficher un placeholder
                        placeholder = Image.new('RGB', (200, 200), color='lightgray')
                        st.image(placeholder, caption="Image non disponible", use_column_width=True)
                    
                    # Informations du produit
                    product_name = product.get('name', f'Produit {i+1}')
                    product_category = product.get('category', 'Non spécifié')
                    product_price = product.get('price', 'N/A')
                    product_url = product.get('url', '#')
                    product_occasion = product.get('occasion', 'Non spécifié')
                    
                    st.markdown(f"**{product_name}**")
                    st.caption(f"📂 {product_category}")
                    st.caption(f"🎯 {product_occasion}")
                    
                    if product_price != 'N/A':
                        st.markdown(f"💰 **{product_price}**")
                    
                    # Bouton pour voir le produit
                    if product_url and product_url != '#':
                        st.markdown(f"[🔗 Voir le produit]({product_url})")
    else:
        st.warning("⚠️ Aucun produit disponible pour cette combinaison morphologie/occasion")
        
        # Afficher des produits alternatifs
        st.info("Voici quelques suggestions alternatives:")
        alt_products = get_recommended_products(pred['simple_type'], None, product_df, 3)
        
        if alt_products:
            alt_cols = st.columns(3)
            for i, product in enumerate(alt_products):
                with alt_cols[i]:
                    if product.get('image'):
                        product['image'].thumbnail((150, 150))
                        st.image(product['image'], use_column_width=True)
                    st.caption(product.get('name', f'Produit {i+1}'))

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>👗 <strong>Détecteur de Morphologie IA + Recommandations Intelligentes</strong></p>
    <p><small>Analyse de morphologie + Filtrage par occasion + Recommandations personnalisées</small></p>
</div>
""", unsafe_allow_html=True)

# CSS pour améliorer l'interface
st.markdown("""
<style>
    .stButton button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
        border-radius: 8px;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .stProgress > div > div > div {
        background: linear-gradient(45deg, #667eea, #764ba2);
    }
    .stMetric {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 10px;
        border-radius: 10px;
    }
    .product-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)
