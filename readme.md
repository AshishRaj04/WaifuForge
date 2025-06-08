# WaifuForge

WaifuForge is an AI-powered anime character generator that creates unique character faces and names. Built with WGAN-GP architecture and Flask, it features a cyberpunk-themed interface for an immersive generation experience.

## Features

- AI-powered character face generation using WGAN-GP
- Neural network-based anime character name generation
- Real-time generation through REST API
- Cyberpunk-themed responsive UI
- Temperature-controlled name generation
- Canvas-based image rendering

## Tech Stack

### Frontend
- HTML5 with Canvas API
- CSS3 with custom properties
- Vanilla JavaScript with async/await

### Backend
- Flask web server
- Flask-CORS for cross-origin support
- TensorFlow 2.x for model inference

### AI Models
- WGAN-GP for face generation
- Custom neural network for name generation
- TensorFlow and NumPy for computations

## Setup

1. Install required Python packages:
```bash
pip install -r requirements.txt
```

2. Ensure model checkpoints are in place:
```
checkpoints/
├── checkpoints_84/
│   └── wgan.weights.h5    # Face generation model
└── checkpoints_names/
    └── checkpoint         # Name generation model
```

3. Start the Flask server:
```bash
python backend.py
```

4. Open `index.html` in your browser or use a local server:
```bash
python -m http.server 8000
```

## Project Structure

```
img_gen/
├── backend.py          # Flask server and API endpoints
├── face_engine.py      # WGAN-GP model for face generation
├── engine.py          # Name generation model
├── index.html         # Web interface
├── styles.css         # Cyberpunk theme styling
├── requirements.txt   # Python dependencies
└── checkpoints/      # Model weights
```

## API Endpoints

### Server Status
- **GET /** 
  - Returns server status
  - Response: `{"message": "Server is running"}`

### Generate Character
- **POST /api/generate**
  - Generates both face and name
  - Returns: 
    ```json
    {
      "status": "success",
      "name": "generated_name",
      "face": "face_data"
    }
    ```

## Configuration

- Face generation parameters in `face_engine.py`
- Name generation temperature in `engine.py`
- UI theme variables in `styles.css`
- Server settings in `backend.py`

## Known Limitations

- Face generation requires significant GPU/CPU power
- Initial model loading may take a few seconds
- Limited to 64x64 face generation currently

## License

Copyright © 2025 Ashish Raj. All rights reserved.

## Author

Created by Ashish Raj
- Date: June 2025
- Project: WaifuForge - AI Character Generator