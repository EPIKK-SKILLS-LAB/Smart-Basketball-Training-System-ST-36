# FastAPI Video Stream

This project is a simple FastAPI application that streams video files to a web frontend. It allows users to view videos in their browser using a clean and responsive interface.

## Project Structure

```
fastapi-video-stream
├── app
│   ├── main.py                # Entry point of the FastAPI application
│   ├── routes
│   │   └── video.py           # API routes for video streaming
│   ├── services
│   │   └── video_stream.py     # Logic for handling video streaming
│   ├── templates
│   │   └── index.html          # Main HTML template for the frontend
│   └── static
│       ├── css
│       │   └── styles.css      # Styles for the HTML frontend
│       └── js
│           └── app.js          # JavaScript for frontend interactions
├── videos                      # Directory for storing video files
├── requirements.txt            # Dependencies for the FastAPI application
└── README.md                   # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd fastapi-video-stream
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   uvicorn app.main:app --reload
   ```

5. **Access the application:**
   Open your browser and go to `http://127.0.0.1:8000`.

## Usage

- Upload your video files to the `videos` directory.
- Navigate to the main page to view and stream the videos.

## License

This project is licensed under the MIT License.