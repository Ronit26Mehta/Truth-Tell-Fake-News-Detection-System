const express = require('express');
const axios = require('axios');
const app = express();
const port = 5001; 

app.use(express.json()); 

const FLASK_API_URL = 'http://127.0.0.1:5000'; 

app.post('/train', async (req, res) => {
  try {
    const { X_train, y_train } = req.body;
    const response = await axios.post(`${FLASK_API_URL}/train`, { X_train, y_train });
    res.json(response.data);
  } catch (error) {
    console.error('Error in training:', error);
    res.status(500).json({ message: 'Error in training the model' });
  }
});

app.post('/predict', async (req, res) => {
  try {
    const { texts } = req.body;
    const response = await axios.post(`${FLASK_API_URL}/predict`, { texts });
    res.json(response.data);
  } catch (error) {
    console.error('Error in prediction:', error);
    res.status(500).json({ message: 'Error in prediction' });
  }
});

app.post('/evaluate', async (req, res) => {
  try {
    const { X_test, y_test } = req.body;
    const response = await axios.post(`${FLASK_API_URL}/evaluate`, { X_test, y_test });
    res.json(response.data);
  } catch (error) {
    console.error('Error in evaluation:', error);
    res.status(500).json({ message: 'Error in model evaluation' });
  }
});

app.listen(port, () => {
  console.log(`Node.js server running at http://localhost:${port}`);
});