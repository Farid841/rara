import os
import json
import uuid
import base64
import requests
import streamlit as st
import PyPDF2
import docx
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
import time

# Charger les variables d'environnement
load_dotenv()

# Configuration de la page Streamlit
st.set_page_config(
    page_title="RARA - Recherche et Analyse des Maladies Rares",
    page_icon="🧠",
    layout="wide",
)

# Intégration de Tailwind CSS via CDN
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
<style>
    .main, .block-container, .css-1y4p8pa {
        max-width: 1200px !important;
        padding-top: 2rem !important;
        padding-right: 1rem !important;
        padding-left: 1rem !important;
        margin: 0 auto !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .status-indicator {
        display: inline-flex;
        align-items: center;
        margin-left: 0.5rem;
    }
    .spinner {
        border: 2px solid #f3f3f3;
        border-top: 2px solid #3498db;
        border-radius: 50%;
        width: 12px;
        height: 12px;
        animation: spin 1s linear infinite;
        display: inline-block;
        margin-right: 5px;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .message-container {
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #F1F8E9;
        border-left: 4px solid #8BC34A;
    }
    .loader {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(0,0,0,.3);
        border-radius: 50%;
        border-top-color: #3498db;
        animation: spin 1s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)

# Créer le dossier models/ s'il n'existe pas
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Configuration Azure
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")
AZURE_SEARCH_API_VERSION = os.getenv("AZURE_SEARCH_API_VERSION", "2023-07-01-Preview")

AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")
AZURE_EMBEDDING_API_KEY = os.getenv("AZURE_EMBEDDING_API_KEY")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

# Fonctions utilitaires
def status_indicator(message, status=None, show_spinner=False):
    """Affiche un indicateur de statut (✅, ❌, ou spinner)"""
    if status is True:
        return f"{message} <span class='status-indicator'>✅</span>"
    elif status is False:
        return f"{message} <span class='status-indicator'>❌</span>"
    elif show_spinner:
        return f"{message} <span class='status-indicator'><div class='spinner'></div></span>"
    return message

def extract_text_from_pdf(pdf_file):
    """Extraire le texte d'un fichier PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"❌ Erreur lors de l'extraction du PDF: Le fichier pourrait être corrompu ou protégé. Détails: {str(e)}")
        return ""

def extract_text_from_docx(docx_file):
    """Extraire le texte d'un fichier DOCX"""
    try:
        doc = docx.Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"❌ Erreur lors de l'extraction du DOCX: Format non reconnu ou fichier corrompu. Détails: {str(e)}")
        return ""

def extract_text_from_file(file):
    """Extraire le texte de différents types de fichiers"""
    file_content = file.read()
    file_type = file.name.split(".")[-1].lower()
    
    if file_type == "pdf":
        return extract_text_from_pdf(BytesIO(file_content))
    elif file_type == "docx":
        return extract_text_from_docx(BytesIO(file_content))
    elif file_type in ["txt", "md", "html"]:
        try:
            return file_content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return file_content.decode("latin-1")
            except:
                st.error(f"❌ Impossible de décoder le fichier {file.name}: Le fichier contient des caractères non reconnus. Essayez de le sauvegarder avec l'encodage UTF-8.")
                return ""
    else:
        st.error(f"❌ Type de fichier non pris en charge: {file_type}. Seuls les formats PDF, DOCX, TXT, MD et HTML sont acceptés.")
        return ""

def generate_embedding(text):
    """Générer un embedding pour le texte via Azure OpenAI"""
    try:
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_EMBEDDING_API_KEY
        }
        
        # Tronquer le texte si nécessaire (limite d'Azure OpenAI)
        max_tokens = 8191
        truncated_text = text[:max_tokens * 4]  # Approximation grossière
        
        payload = {
            "input": truncated_text,
            "model": AZURE_EMBEDDING_DEPLOYMENT
        }
        
        response = requests.post(
            AZURE_EMBEDDING_ENDPOINT,
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()["data"][0]["embedding"]
        else:
            error_msg = response.text
            st.error(f"❌ Erreur lors de la génération de l'embedding: Le service Azure OpenAI a retourné une erreur (code {response.status_code}). Vérifiez votre configuration API et votre quota. Détails: {error_msg}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Erreur de connexion: Impossible de se connecter au service Azure OpenAI. Vérifiez votre connexion internet et l'URL du point de terminaison.")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ Délai d'attente dépassé: Le service Azure OpenAI n'a pas répondu à temps. Réessayez plus tard.")
        return None
    except Exception as e:
        st.error(f"❌ Erreur inattendue lors de la génération de l'embedding: {str(e)}")
        return None

def store_in_azure_search(doc_id, content, embedding, model_id):
    """Stocker le contenu et son embedding dans Azure Cognitive Search"""
    try:
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_SEARCH_KEY
        }
        
        # Adapté à la structure réelle de l'index
        document = {
            "id": doc_id,
            "content": content,
            "embedding": embedding
        }
        
        response = requests.post(
            f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX}/docs/index?api-version={AZURE_SEARCH_API_VERSION}",
            headers=headers,
            json={"value": [document]}
        )
        
        if response.status_code in [200, 201, 204]:
            return True
        else:
            error_content = response.text
            st.error(f"❌ Erreur lors de l'enregistrement dans Azure Search (code {response.status_code}): Vérifiez que l'index '{AZURE_SEARCH_INDEX}' existe et qu'il contient bien le champ 'embedding'. Détails: {error_content}")
            return False
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Erreur de connexion: Impossible de se connecter au service Azure Cognitive Search. Vérifiez votre connexion internet et l'URL du point de terminaison ({AZURE_SEARCH_ENDPOINT}).")
        return False
    except Exception as e:
        st.error(f"❌ Erreur inattendue lors de l'enregistrement dans Azure Search: {str(e)}")
        return False

def search_in_azure_search(query_embedding, top=3):
    """Rechercher les documents pertinents dans Azure Cognitive Search"""
    try:
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_SEARCH_KEY
        }
        
        # Adapté à la structure réelle de l'index
        payload = {
            "vectorQueries": [
                {
                    "kind": "vector",
                    "vector": query_embedding,
                    "fields": "embedding",
                    "k": top
                }
            ],
            "select": "content"
        }
        
        response = requests.post(
            f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX}/docs/search?api-version={AZURE_SEARCH_API_VERSION}",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            results = response.json().get("value", [])
            if not results:
                st.warning("⚠️ Aucun document pertinent trouvé pour cette requête. Essayez de reformuler votre question ou d'ajouter plus de documents.")
            return [result["content"] for result in results]
        else:
            error_content = response.text
            st.error(f"❌ Erreur lors de la recherche dans Azure Search (code {response.status_code}): Vérifiez que l'index est correctement configuré pour les recherches vectorielles. Détails: {error_content}")
            return []
    except requests.exceptions.ConnectionError:
        st.error("❌ Erreur de connexion: Impossible de se connecter au service Azure Cognitive Search. Vérifiez votre connexion internet et l'URL du point de terminaison.")
        return []
    except Exception as e:
        st.error(f"❌ Erreur inattendue lors de la recherche dans Azure Search: {str(e)}")
        return []

def get_chat_completion(messages):
    """Obtenir une réponse de GPT-4 via Azure OpenAI"""
    try:
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_API_KEY
        }
        
        payload = {
            "messages": messages,
            "model": AZURE_DEPLOYMENT_NAME,
            "temperature": 0.7,
            "max_tokens": 800
        }
        
        response = requests.post(
            f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT_NAME}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            error_content = response.text
            st.error(f"❌ Erreur lors de l'appel à Azure OpenAI (code {response.status_code}): Vérifiez que le modèle '{AZURE_DEPLOYMENT_NAME}' existe dans votre déploiement. Détails: {error_content}")
            return "Désolé, je n'ai pas pu générer une réponse en raison d'un problème avec le service Azure OpenAI. Veuillez consulter les messages d'erreur pour plus d'informations."
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Erreur de connexion: Impossible de se connecter au service Azure OpenAI à l'adresse {AZURE_OPENAI_ENDPOINT}. Vérifiez votre connexion internet et l'URL du point de terminaison.")
        return "Désolé, je n'ai pas pu générer une réponse en raison d'un problème de connexion au service Azure OpenAI."
    except requests.exceptions.Timeout:
        st.error("❌ Délai d'attente dépassé: Le service Azure OpenAI n'a pas répondu à temps. Réessayez plus tard.")
        return "Désolé, le temps de réponse du service Azure OpenAI est trop long. Veuillez réessayer plus tard."
    except Exception as e:
        st.error(f"❌ Erreur inattendue lors de l'appel à Azure OpenAI: {str(e)}")
        return "Désolé, une erreur inattendue est survenue lors de la génération de la réponse. Veuillez réessayer ou contacter l'administrateur."

def get_models():
    """Récupérer la liste des modèles disponibles"""
    models = []
    try:
        for filename in os.listdir(MODELS_DIR):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(MODELS_DIR, filename), "r") as f:
                        model = json.load(f)
                        models.append(model)
                except json.JSONDecodeError:
                    st.error(f"❌ Erreur lors de la lecture du modèle {filename}: Fichier JSON invalide. Le fichier pourrait être corrompu.")
                except Exception as e:
                    st.error(f"❌ Erreur lors de la lecture du modèle {filename}: {str(e)}")
        
        if not models:
            st.info("ℹ️ Aucun modèle trouvé dans le dossier 'models'. Créez d'abord un modèle.")
            
        return models
    except FileNotFoundError:
        st.error(f"❌ Dossier 'models' introuvable à l'emplacement {MODELS_DIR}. Vérifiez que le dossier existe.")
        return []
    except PermissionError:
        st.error(f"❌ Erreur de permission: Impossible d'accéder au dossier 'models'. Vérifiez les droits d'accès.")
        return []
    except Exception as e:
        st.error(f"❌ Erreur inattendue lors de la récupération des modèles: {str(e)}")
        return []

def save_model(model_name, instructions):
    """Sauvegarder un modèle dans un fichier JSON"""
    model_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    
    model_data = {
        "id": model_id,
        "name": model_name,
        "instructions": instructions,
        "created_at": created_at
    }
    
    try:
        with open(os.path.join(MODELS_DIR, f"{model_id}.json"), "w") as f:
            json.dump(model_data, f, indent=2)
        return model_id
    except PermissionError:
        st.error(f"❌ Erreur de permission: Impossible d'écrire dans le dossier 'models'. Vérifiez les droits d'accès au répertoire {MODELS_DIR}.")
        return None
    except Exception as e:
        st.error(f"❌ Erreur lors de l'enregistrement du modèle: {str(e)}")
        return None

def create_model_ui():
    """Interface utilisateur pour créer un modèle personnalisé"""
    st.markdown("<h2 class='text-xl font-bold mb-4'>📄 Création d'un modèle d'assistance pour maladies rares</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="bg-blue-50 p-4 rounded-lg mb-4 shadow">
        <p class="text-blue-800">
            <span class="font-bold">💡 Conseil :</span> Téléversez des documents médicaux sur des maladies rares et définissez des instructions 
            pour créer un assistant IA personnalisé qui vous aidera à comprendre ces documents complexes.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form(key="create_model_form"):
        col1, col2 = st.columns([2, 1])
        with col1:
            model_name = st.text_input("Nom du modèle", placeholder="Ex: Assistant Maladies Génétiques Rares")
        
        instructions = st.text_area(
            "Instructions personnalisées",
            height=150,
            placeholder="Ex: Tu es un assistant médical spécialisé dans les maladies rares. Réponds de manière claire, précise et pédagogique aux questions médicales. Utilise un langage accessible tout en étant scientifiquement rigoureux..."
        )
        
        uploaded_files = st.file_uploader(
            "Téléverser des documents médicaux (.pdf, .docx, .md, .html, .txt)",
            accept_multiple_files=True,
            type=["pdf", "docx", "md", "html", "txt"]
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit_button = st.form_submit_button(
                label="🔍 Créer le modèle d'assistance", 
                use_container_width=True,
                type="primary"
            )
    
    if submit_button:
        if not model_name:
            st.error("⚠️ Veuillez donner un nom au modèle.")
            return
        
        if not instructions:
            st.error("⚠️ Veuillez fournir des instructions pour le modèle.")
            return
        
        if not uploaded_files:
            st.error("⚠️ Veuillez téléverser au moins un document.")
            return
            
        # Création du modèle
        with st.spinner("Création du modèle en cours..."):
            model_id = save_model(model_name, instructions)
            
            if not model_id:
                st.error("🚫 Échec de la création du modèle.")
                return
            
            # Traitement des fichiers
            st.markdown("""
            <div class="bg-gray-50 p-4 rounded-lg shadow mb-4">
                <h3 class="font-bold text-gray-700 mb-2">Traitement des documents</h3>
                <div id="progress-container"></div>
            </div>
            """, unsafe_allow_html=True)
            
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                progress_text.markdown(status_indicator(f"Traitement de {file.name}", show_spinner=True), unsafe_allow_html=True)
                
                # Extraction du texte
                text = extract_text_from_file(file)
                if not text:
                    progress_text.markdown(status_indicator(f"Échec du traitement de {file.name}", status=False), unsafe_allow_html=True)
                    time.sleep(1)
                    continue
                
                # Génération de l'embedding
                embedding = generate_embedding(text)
                if not embedding:
                    progress_text.markdown(status_indicator(f"Échec de la génération d'embedding pour {file.name}", status=False), unsafe_allow_html=True)
                    time.sleep(1)
                    continue
                
                # Stockage dans Azure Search
                doc_id = f"{model_id}_{i}_{uuid.uuid4()}"
                success = store_in_azure_search(doc_id, text, embedding, model_id)
                
                if success:
                    progress_text.markdown(status_indicator(f"Fichier {file.name} traité avec succès", status=True), unsafe_allow_html=True)
                else:
                    progress_text.markdown(status_indicator(f"Échec du stockage de {file.name}", status=False), unsafe_allow_html=True)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
                time.sleep(0.5)
            
            st.success(f"✅ Modèle '{model_name}' créé avec succès ! Vous pouvez maintenant l'utiliser pour discuter de vos documents médicaux.")
            time.sleep(2)
            st.rerun()

def chat_model_ui():
    """Interface utilisateur pour interagir avec un modèle"""
    st.markdown("<h2 class='text-xl font-bold mb-4'>💬 Discuter avec votre assistant médical</h2>", unsafe_allow_html=True)
    
    models = get_models()
    if not models:
        st.info("🔍 Aucun modèle d'assistance n'a été créé. Veuillez d'abord créer un modèle dans l'onglet 'Créer un modèle'.")
        return
    
    model_names = [model["name"] for model in models]
    selected_model_name = st.selectbox("Choisir un modèle d'assistance médicale", model_names)
    
    selected_model = next((model for model in models if model["name"] == selected_model_name), None)
    
    if selected_model:
        st.markdown(f"""
        <div class='bg-blue-50 p-4 rounded-lg mb-4 shadow'>
            <p class='font-semibold mb-2'>📋 Instructions pour cet assistant :</p>
            <p class='text-sm text-gray-700'>{selected_model['instructions']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialiser l'historique de conversation s'il n'existe pas
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Zone pour afficher l'historique des messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="message-container user-message">
                        <p><strong>Vous :</strong> {message['content']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="message-container assistant-message">
                        <p><strong>Assistant médical :</strong> {message['content']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Zone de saisie utilisateur
        user_input = st.text_area("Votre question sur cette maladie rare :", 
                                  height=100, 
                                  key="user_question",
                                  placeholder="Ex: Quels sont les symptômes principaux de cette maladie ? Y a-t-il des traitements disponibles ?")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit_button = st.button("📨 Envoyer ma question", 
                                      use_container_width=True, 
                                      type="primary",
                                      key="send_button")
        
        if submit_button:
            if user_input:
                # Ajouter la question à l'historique
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input
                })
                
                with st.spinner("Recherche dans la documentation médicale..."):
                    # Générer l'embedding de la question
                    query_embedding = generate_embedding(user_input)
                    
                    if query_embedding:
                        # Rechercher les documents pertinents
                        # Modification: on ne filtre plus par model_id car ce champ n'existe pas dans l'index
                        relevant_docs = search_in_azure_search(query_embedding)
                        
                        if relevant_docs:
                            # Préparer le contexte pour GPT-4
                            context = "\n\n".join(relevant_docs)
                            
                            # Construire le prompt pour GPT-4
                            messages = [
                                {"role": "system", "content": f"{selected_model['instructions']}\n\nUtilise ces informations médicales pour répondre à la question de l'utilisateur sur cette maladie rare:\n{context}"},
                                {"role": "user", "content": user_input}
                            ]
                            
                            # Obtenir la réponse de GPT-4
                            response = get_chat_completion(messages)
                            
                            # Ajouter la réponse à l'historique
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": response
                            })
                        else:
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": "Désolé, je n'ai pas trouvé d'informations pertinentes sur cette maladie rare dans les documents fournis. Pourriez-vous reformuler votre question ou consulter un professionnel de santé ?"
                            })
                    else:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": "Désolé, je n'ai pas pu traiter votre question. Il semble y avoir un problème technique. Veuillez réessayer plus tard."
                        })
                
                st.rerun()

# Interface principale
def main():
    st.markdown("""
    <div class="flex items-center justify-center mb-6">
        <h1 class='text-3xl font-bold text-blue-800'>🧠 RARA - Recherche et Analyse des Maladies Rares</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Vérifier la configuration
    config_errors = []
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        config_errors.append("⚠️ Azure OpenAI n'est pas configuré correctement. Vérifiez les variables AZURE_OPENAI_ENDPOINT et AZURE_OPENAI_API_KEY dans le fichier .env")
    if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_KEY or not AZURE_SEARCH_INDEX:
        config_errors.append("⚠️ Azure Cognitive Search n'est pas configuré correctement. Vérifiez les variables AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY et AZURE_SEARCH_INDEX dans le fichier .env")
    if not AZURE_EMBEDDING_ENDPOINT or not AZURE_EMBEDDING_API_KEY:
        config_errors.append("⚠️ Azure Embedding n'est pas configuré correctement. Vérifiez les variables AZURE_EMBEDDING_ENDPOINT et AZURE_EMBEDDING_API_KEY dans le fichier .env")
    
    if config_errors:
        st.error("⚠️ Configuration incomplète des services Azure:")
        for error in config_errors:
            st.warning(error)
        st.info("""
        ℹ️ Pour configurer l'application correctement:
        1. Créez un fichier .env à la racine du projet
        2. Copiez le contenu du fichier .env.example
        3. Remplacez les valeurs par vos propres clés et points de terminaison Azure
        4. Redémarrez l'application
        """)
        return
    
    # Sidebar pour navigation
    with st.sidebar:
        st.markdown("""
        <div class="p-4 bg-blue-100 rounded-lg shadow-md mb-6">
            <h2 class='text-xl font-bold mb-4 text-blue-800'>Navigation</h2>
            <p class="text-sm text-gray-700 mb-2">Utilisez cette application pour :</p>
            <ul class="list-disc pl-5 text-sm text-gray-700">
                <li>Créer un assistant IA pour comprendre des documents médicaux</li>
                <li>Poser des questions sur des maladies rares</li>
                <li>Obtenir des explications claires et précises</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        tab = st.radio("Navigation", ["📄 Créer un modèle", "💬 Discuter avec un modèle"], label_visibility="collapsed")
    
    if tab == "📄 Créer un modèle":
        create_model_ui()
    else:
        chat_model_ui()

if __name__ == "__main__":
    main()
