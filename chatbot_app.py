import streamlit as st
import time

# --- Configuration de la page ---
st.set_page_config(page_title="RARA - Assistant IA Maladies Rares", page_icon="🧬", layout="centered")

# --- CSS personnalisé pour intégrer Tailwind depuis CDN ---
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        .message-bubble {
            max-width: 75%;
            padding: 0.75rem 1rem;
            border-radius: 1rem;
            margin-bottom: 0.5rem;
            animation: fadeIn 0.5s ease-in-out;
        }
        .user-bubble {
            background-color: #3b82f6;
            color: white;
            align-self: flex-end;
        }
        .bot-bubble {
            background-color: #f3f4f6;
            color: black;
            align-self: flex-start;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
        }
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(10px);}
            to {opacity: 1; transform: translateY(0);}
        }
    </style>
""", unsafe_allow_html=True)

# --- Titre principal ---
st.markdown("""
    <div class="text-center my-4">
        <h1 class="text-4xl font-bold">🧬 RARA</h1>
        <p class="text-lg text-gray-600">Votre assistant IA pour les maladies rares</p>
    </div>
""", unsafe_allow_html=True)

# --- Sélection du modèle ---
model = st.selectbox("Choisissez un modèle de langage :", ["GPT-3.5", "GPT-4", "Mistral", "Claude"])

# --- Historique de chat ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "bot", "content": "Bonjour ! Je suis RARA, votre assistant IA spécialisé dans les maladies rares. En quoi puis-je vous aider aujourd’hui ?"}
    ]

# --- Fonction de réponse fictive selon modèle ---
def generate_response(model, prompt):
    if model == "GPT-4":
        prompt = f"Réponds en détail et avec précision : {prompt}"
    elif model == "GPT-3.5":
        prompt = f"Réponds simplement et clairement : {prompt}"
    elif model == "Mistral":
        prompt = f"Fais une synthèse concise : {prompt}"
    elif model == "Claude":
        prompt = f"Sois empathique et bienveillant : {prompt}"

    time.sleep(1.5)
    return f"[Réponse simulée par {model}] Je suis désolé, je ne suis qu'une démonstration pour l'instant."

# --- Zone principale de chat ---
chat_placeholder = st.container()
with chat_placeholder:
    for message in st.session_state.chat_history:
        alignment = "user-bubble" if message["role"] == "user" else "bot-bubble"
        st.markdown(f"""
            <div class="chat-container">
                <div class="message-bubble {alignment}">{message['content']}</div>
            </div>
        """, unsafe_allow_html=True)

# --- Entrée utilisateur avec envoi via Entrée ---
with st.container():
    user_input = st.text_input("Votre message :", key="user_input", label_visibility="collapsed", placeholder="Écrivez ici et appuyez sur Entrée pour envoyer...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("RARA rédige une réponse..."):
            bot_response = generate_response(model, user_input)
        st.session_state.chat_history.append({"role": "bot", "content": bot_response})
        st.experimental_rerun()
