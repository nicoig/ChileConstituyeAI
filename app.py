# Para crear el requirements.txt ejecutamos 
# pipreqs --encoding=utf8 --force

# Primera Carga a Github
# git init
# git add .
# git commit -m "primer commit"
# git remote add origin https://github.com/nicoig/ChileConstituyeAI.git
# git push -u origin master

# Actualizar Repo de Github

'''
git add .
git commit -m "Se actualizan las variables de entorno"
git push origin master
'''

# Para eliminar un repo cargado
# git remote remove origin

# Pasando a master
# git checkout -b master
# git push origin 

# Verificando donde está montado el proyecto
# git remote -v

# Cambiar la URL del control remoto "origin":
# git remote set-url origin https://github.com/nicoig/ChileConstituyeAI.git


###############################################################


import os
import pickle
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template, scroll_js
from langchain.prompts.prompt import PromptTemplate



import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

@st.cache_data
def get_pdf_text(filepaths):
    text = ""
    for filepath in filepaths:
        with open(filepath, "rb") as file:
            pdf = PdfReader(file)
            for page in pdf.pages:
                text += page.extract_text()
    return text


@st.cache_data
def get_text_chunks(text, chunks_file):
    if os.path.exists(chunks_file):
        with open(chunks_file, 'rb') as f:
            chunks = pickle.load(f)
    else:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        with open(chunks_file, 'wb') as f:
            pickle.dump(chunks, f)
    return chunks


@st.cache_data
def get_vectorstore(text_chunks, vectorstore_file):
    if os.path.exists(vectorstore_file):
        with open(vectorstore_file, 'rb') as f:
            vectorstore = pickle.load(f)
    else:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        with open(vectorstore_file, 'wb') as f:
            pickle.dump(vectorstore, f)
    return vectorstore

def get_conversation_chain(vectorstore, model_name):
    llm = ChatOpenAI(model_name=model_name)
    qa_template = """
        Eres un experto en temas políticos y legales, con especial énfasis en el proceso constituyente de Chile. 
        Tu función es proporcionar información clara y precisa sobre el plebiscito de la nueva constitución de Chile,
        basándote en una serie de documentos y fuentes relevantes.

        Contexto: {context}
        =========
        Pregunta: {question}
        ======
    """
    # Aquí podrías añadir el código para generar la respuesta usando el modelo.

    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context","question" ])
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': QA_PROMPT}
    )
    return conversation_chain




def handle_userinput(user_question, chat_placeholder, loading_placeholder):
    # Añadir la pregunta del usuario al historial del chat
    st.session_state.chat_history.append(user_question)
    
    # Mostrar el historial del chat actualizado
    update_chat(chat_placeholder)
    
    # Mostrar el mensaje de "Generando respuesta..." en su propio placeholder
    loading_placeholder.text("Generando respuesta...")
    
    # Obtener la respuesta real del chatbot
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history.append(response['chat_history'][-1].content)  # Añadir la respuesta real al historial del chat
    
    # Actualizar el chat con la respuesta real y limpiar el placeholder de "Cargando"
    update_chat(chat_placeholder)
    loading_placeholder.empty()  # Limpiar el placeholder de "Cargando" después de que la respuesta real ha sido añadida

def update_chat(chat_placeholder):
    chat_content = ""
    for i, message in enumerate(st.session_state.chat_history):
        content = message.content if hasattr(message, 'content') else message  # Ajuste aquí
        if i % 2 == 0:
            chat_content += user_template.replace("{{MSG}}", content)
        else:
            chat_content += bot_template.replace("{{MSG}}", content)
    chat_placeholder.write(chat_content, unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="ChileConstituye IA", page_icon=":flag-cl:", layout="wide")
    st.sidebar.title('Menu Plebiscito Constituyente')

    st.sidebar.title('Menu')
    model_name = st.sidebar.selectbox(
        'Selecciona un modelo de LLM:',
        ('gpt-3.5-turbo', 'gpt-4', 'gpt-3.5-turbo-16k','text-davinci-003') # Puedes poner los modelos que quieras aquí
    )
    temperature = st.sidebar.slider('Ajusta la temperatura:', min_value=0.0, max_value=1.0, value=0.1, step=0.1)

    if "initialized" not in st.session_state:
        file_directory = 'files'
        filepaths = [os.path.join(file_directory, file) for file in os.listdir(file_directory) if file.endswith('.pdf')]
        
        text = get_pdf_text(filepaths)
        chunks_file = 'chunks.pkl'
        chunks = get_text_chunks(text, chunks_file)
        vectorstore_file = 'vectorstore.pkl'
        vectorstore = get_vectorstore(chunks, vectorstore_file)
        
        st.session_state.conversation = get_conversation_chain(vectorstore, model_name)
        st.session_state.llm_temperature = temperature
        st.session_state.chat_history = []
        st.session_state.initialized = True

    # Título de la aplicación
    st.title("ChileConstituye IA")


        # Mostrar la imagen
    st.image('img/constitucion.png', width=300)

    st.write("""
    Hola, soy tu asistente virtual especializado en el plebiscito de la nueva constitución de Chile. 
    Estoy aquí para ofrecerte información y resolver tus dudas relacionadas con este importante evento político.
    Vota este domingo 18 de Diciembre de manera informada
    """)

    chat_placeholder = st.empty()  # Placeholder para el historial del chat
    loading_placeholder = st.empty() 
    
    user_question = st.chat_input("Realiza tu consulta:")
    if user_question:
        handle_userinput(user_question, chat_placeholder, loading_placeholder)  # Pasar ambos placeholders como argumentos

if __name__ == '__main__':
    main()