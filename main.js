// Función para controlar la visibilidad de la flecha
function toggleScrollArrow() {
  const scrollArrow = document.getElementById('scroll-arrow');
  const scrollPosition = window.scrollY || document.documentElement.scrollTop;
  const documentHeight = document.documentElement.scrollHeight;
  const windowHeight = window.innerHeight;

  // Si no estamos en el final de la página y hay más contenido para ver
  if (scrollPosition < documentHeight - windowHeight - 100) {
    scrollArrow.classList.add('visible');
  } else {
    scrollArrow.classList.remove('visible');
  }
}

// Detectar el desplazamiento de la página
window.addEventListener('scroll', toggleScrollArrow);

// Crear el modelo
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [4] })); // Capa densa para regresión lineal

// Compilar el modelo
model.compile({
  loss: 'meanSquaredError',
  optimizer: 'sgd'
});

// Función para entrenar el modelo
async function trainModel() {
  // Características de las casas
  const xs = tf.tensor2d([
    [1, 1, 1, 1], // 1 habitación, 1 baño, sala, estado de la casa = buen estado
    [2, 1, 1, 1], // 2 habitaciones, 1 baño, sala, estado de la casa = buen estado
    [2, 1, 1, 2], // 2 habitaciones, 1 baño, sala, estado de la casa = mal estado
    [3, 2, 1, 1], // 3 habitaciones, 1 baño, sala, estado de la casa = buen estado
    [4, 3, 1, 1], // 4 habitaciones, 2 baños, sala, estado de la casa = buen estado
    [5, 3, 2, 1], // 5 habitaciones, 3 baños, sala, estado de la casa = buen estado
    [5, 3, 2, 2]  // 5 habitaciones, 3 baños, sala, estado de la casa = mal estado
  ]);

  // Valores aproximados de precios (en miles de dólares)
  const ys = tf.tensor1d([150.000, 250.000, 200.000, 350.000, 450.000, 600.000, 530.000]); // Precios de las viviendas en dólares

  // Entrenar el modelo
  await model.fit(xs, ys, { epochs: 100 });

  // Evaluar el modelo usando las mismas entradas
  const preds = model.predict(xs).dataSync();
  const realValues = ys.dataSync();
  const mae = preds.reduce((acc, pred, i) => acc + Math.abs(pred - realValues[i]), 0) / preds.length;
  const meanY = realValues.reduce((acc, val) => acc + val, 0) / realValues.length;
  const ssTotal = realValues.reduce((acc, val) => acc + Math.pow(val - meanY, 2), 0);
  const ssRes = preds.reduce((acc, pred, i) => acc + Math.pow(realValues[i] - pred, 2), 0);
  const r2 = 1 - (ssRes / ssTotal);

  console.log(`Entrenamiento completo. MAE: ${mae}, R²: ${r2}`);
}

// Función para realizar la predicción
function makePrediction(event) {
  // Evitar que el formulario se recargue
  event.preventDefault();

  const rooms = parseFloat(document.getElementById('rooms').value);
  const size = parseFloat(document.getElementById('size').value);
  const bathrooms = parseFloat(document.getElementById('bathrooms').value);
  const condition = parseFloat(document.getElementById('condition').value);

  // Asegurarse de que los valores no sean NaN
  if (isNaN(rooms) || isNaN(size) || isNaN(bathrooms) || isNaN(condition)|| size > 3 || condition > 2) {
    alert("Por favor ingresa valores válidos.");
    return;
  }

  const inputTensor = tf.tensor2d([[rooms, size, bathrooms, condition]]);
  const prediction = model.predict(inputTensor);
  const resultElement = document.getElementById('result');

  // Mostrar solo dos decimales en el resultado
  resultElement.textContent = `El valor aproximado es: $${prediction.dataSync()[0].toFixed(3)} Dólares`;

  // Desplazar hacia el resultado
  resultElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Entrena el modelo al cargar la página
window.onload = () => {
  trainModel();

  // Evento para el formulario
  const form = document.getElementById('prediction-form');
  form.addEventListener('submit', makePrediction);
};
