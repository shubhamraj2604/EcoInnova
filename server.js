const express = require('express');
const app = express();
const path = require('path');
const axios = require('axios');

// Serve static files (CSS, JS, etc.)
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());  // To parse JSON bodies

// Route for the homepage
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Route for the home page
app.get('/home', (req, res) => {
    res.sendFile(path.join(__dirname, 'home.html'));
});

// Route for the about page
app.get('/about', (req, res) => {
    res.sendFile(path.join(__dirname, 'about.html'));
});

// Route to handle requests from the frontend and forward them to the Flask server
app.post('/recommendation_API/app', async (req, res) => {
    try {
        // Forward the request to the Flask API
        const response = await axios.post('http://127.0.0.1:5000/recommendation', req.body);
        res.json(response.data);  // Send the Flask response back to the client
    } catch (error) {
        console.error('Error connecting to Flask API:', error);
        res.status(500).json({ error: 'Failed to fetch recommendations' });
    }
});

// Start the server on port 3000
const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server is running on http://127.0.0.1:${PORT}`);
});
