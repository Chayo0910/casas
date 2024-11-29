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
  const xs = tf.tensor2d([ 
    [1, 1, 1, 1], // 1 habitación, 1 baño ,sala ,buen estado 1  
    [2, 2, 1, 1], // 2 habitaciones, 1 baño,sala ,buen estado 1   
    [2, 2, 2, 2], // 3 habitaciones, 2 baños ,sala ,buen estado 1  
    [3, 3, 2, 2], // 4 habitaciones, 4 baños ,sala ,buen estado 1    
    [3, 3, 3, 3]  // 5 habitaciones, 4 baños ,sala ,buen estado 1  
  ]);
  const ys = tf.tensor1d([200, 300, 400, 500, 600]); // Valores de las viviendas correspondientes

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
function makePrediction() {
  const inputValue = parseFloat(document.getElementById('input-value').value);
  const inputValue2 = parseFloat(document.getElementById('input-value2').value);
  const inputValue3 = parseFloat(document.getElementById('input-value3').value);
  const inputValue4 = parseFloat(document.getElementById('input-value4').value);
  const inputTensor = tf.tensor2d([[inputValue, inputValue2, inputValue3, inputValue4]]); 
  const prediction = model.predict(inputTensor);
  const resultElement = document.getElementById('result');
  resultElement.textContent = `La predicción es: ${prediction.dataSync()[0]}`;
}

// Entrena el modelo al cargar la página
window.onload = trainModel;
