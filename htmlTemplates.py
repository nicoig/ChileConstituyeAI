css = '''
<style>
body {
    background-color: #000000;  /* Color de fondo para todo el cuerpo de la página */
    color: #ffffff;  /* Color de texto general para todo el cuerpo de la página */
}
.reportview-container {
    max-width: 70%;
    margin: auto;
    background-color: #262626;  /* Cambia el color de fondo del contenedor de la aplicación Streamlit */
    color: #ffffff;  /* Cambia el color de texto del contenedor de la aplicación Streamlit */
    padding: 0px;  /* Agrega algo de espacio alrededor del contenedor de la aplicación Streamlit */
}
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
    width: 70%;  /* Ajusta este valor según tus necesidades */
    word-wrap: break-word;  /* Romper palabras si es necesario para evitar el desbordamiento */
    overflow-wrap: break-word;  /* Permite que las palabras se rompan para evitar el desbordamiento */
}

/* Nuevo código para mover la barra de chat hacia arriba en móviles */
@media (max-width: 768px) {
    .stTextInput input {
        margin-top: -300px !important;
    }
}

/* Nuevo código para eliminar el logo de Streamlit */
.reportview-container .main footer {visibility: hidden;}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTlDn6yGHhViogJV4sUFM-8HlrM3k-V6rOctQ&usqp=CAU" style="max-height: 45px; max-width: 45px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''



user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/186/186313.png" style="max-height: 45px; max-width: 45px; border-radius: 50%; object-fit: cover;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''



scroll_js = """
<script>
// Get scroll position from session storage and scroll to it
const scrollY = sessionStorage.getItem('scrollY')
if (scrollY) window.scrollTo(0, parseInt(scrollY))

// Store scroll position in session storage when user scrolls
window.addEventListener('scroll', () => {
  sessionStorage.setItem('scrollY', window.scrollY)
})
</script>
"""
