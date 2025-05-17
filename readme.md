# WaifuForge

WaifuForge is an AI-powered anime character name generator built using TensorFlow and Flask. The project features a modern, cyberpunk-inspired web interface for generating unique character names.

## Features

- AI-powered name generation using a deep learning model
- Real-time name generation through API
- Cyberpunk-inspired UI design
- Responsive web interface
- Temperature-controlled name generation for varied creativity

## Tech Stack

- **Frontend**: HTML5, CSS3
- **Backend**: Flask
- **AI Model**: TensorFlow , Numpy
- **API**: RESTful endpoints with CORS support

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure model checkpoints are in place:
```
checkpoints/
└── checkpoints_84/
    └── ... (checkpoint files)
```

3. Run the Flask server:
```bash
python backend.py
```

4. Open `index.html` in a web browser

## Project Structure

```
img_gen/
├── backend.py          # Flask server
├── engine.py          # AI model and generation logic
├── index.html         # Frontend interface
├── styles.css         # UI styling
├── requirements.txt   # Python dependencies
└── checkpoints/       # Model checkpoint files
```

## API Endpoints

- `GET /`: Server status check
- `POST /api/generate-name`: Generate a new name

## Configuration

- Model temperature can be adjusted in `backend.py`
- UI theme colors can be modified in `styles.css`
- Model parameters are defined in `engine.py`

## License

Copyright © 2025. Ashish Raj.

## Author

Created by Ashish Raj