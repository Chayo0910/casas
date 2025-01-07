/* Estilos generales */
body, html {
  height: 100%;
  margin: 0;
  font-family: sans-serif;
}

/* Contenedor principal */
.container {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding: 2rem;
  width: 100%; /* Ocupa todo el ancho disponible */
}

/* Cuadrícula para los inputs */
.input-grid {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  width: 100%; /* Ocupa todo el ancho disponible */
}

/* Estilos para cada input */
.input-box {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border: 1px solid #ccc;
  border-radius: 5px;
  box-sizing: border-box;
  text-align: center;
  width: 100%; 
  margin-bottom: 20px;
  padding: 20px;
  box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
  transition: transform 0.3s ease-in-out;
  gap: 30px;
}

.input-box:hover {
  transform: scale(1.05);
}

/* Estilos para las imágenes */
.input-box img {
  max-width: 100%;
  height: 350px;
  margin-bottom: 10px;
  width: 80%;
}

h1 {
  text-align: center;
  font-size: clamp(3rem, 2vh + 1rem, 2.5rem);
  color: #333;
}

h2 {
  font-size: clamp(2rem, 2vh + 1rem, 2.5rem);
  color: #333;
}

label {
  font-size: clamp(1.8rem, 2vh + 1rem, 2.2rem);
  color: #444;
}

/* Estilos para los inputs */
.input-box input {
  height: 100px;
  padding: 10px;
  font-size: 16px;
  width: 324px;
  box-sizing: border-box;
  text-align: center;
  font-size: clamp(1rem, 2vh + 1rem, 2.0rem);
  border-radius: 5px;
  border: 1px solid #ccc;
  margin-top: 10px;
}

.input-box input:focus {
  border-color: #4CAF50;
  outline: none;
  box-shadow: 0 0 5px rgba(76, 175, 80, 0.5);
}

/* Estilos para el botón */
button {
  background-color: #4CAF50;
  font-size: clamp(1rem, 2vh + 1rem, 2.0rem);
  color: white;
  padding: 15px 20px;
  margin-top: 5%;
  border: none;
  cursor: pointer;
  border-radius: 5px;
}

button:hover {
  background-color: #45a049;
}

#result {
  margin-top: 20px;
  font-size: 2rem;
  font-weight: bold;
  color: #4CAF50;
  text-align: center;
}
/* Estilos para la flecha de scroll */
.scroll-arrow {
  position: fixed;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  font-size: 3rem;
  background-color: rgba(0, 0, 0, 0.7);
  color: white;
  border-radius: 50%;
  padding: 15px;
  cursor: pointer;
  opacity: 0;  /* Oculta inicialmente */
  transition: opacity 0.3s ease-in-out;
}

.scroll-arrow.visible {
  opacity: 1;  /* Muestra la flecha cuando se debe mostrar */
}


@media (min-width: 1000px) {
  h2 {
    color: #4CAF50;
  }
  .input-box img {
    width: 60%;
    height: 50%;
  }
}
