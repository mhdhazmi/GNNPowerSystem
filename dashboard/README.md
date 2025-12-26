# Physics-SSL Power Grid Dashboard

Interactive dashboard for exploring the Physics-Guided Self-Supervised Learning for Power Grid Analysis paper.

## Features

- **Overview**: Paper summary with key metrics and findings
- **Architecture**: Interactive model visualization with PhysicsGuidedEncoder details
- **Results Explorer**: D3.js charts comparing SSL vs Scratch performance
- **Dataset Explorer**: IEEE 24-bus and 118-bus topology visualization
- **Live Inference**: Run model predictions on sample data

## Quick Start

### Development Mode

1. **Start the backend**:
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

2. **Start the frontend**:
```bash
cd frontend
npm install
npm run dev
```

3. Open http://localhost:5173

### Docker Deployment

```bash
docker-compose up --build
```

Then open http://localhost:3000

## Project Structure

```
dashboard/
├── frontend/              # React + Vite + Tailwind
│   ├── src/
│   │   ├── components/    # React components
│   │   │   ├── Layout/    # Navigation, sidebar
│   │   │   ├── Overview/  # Landing page
│   │   │   ├── Architecture/ # Model visualization
│   │   │   ├── Results/   # D3 charts
│   │   │   ├── Dataset/   # Grid explorer
│   │   │   └── Inference/ # Live demo
│   │   └── App.jsx        # Main router
│   └── package.json
│
├── backend/               # FastAPI
│   ├── app/
│   │   ├── main.py        # FastAPI app
│   │   └── routers/       # API endpoints
│   └── requirements.txt
│
├── data/                  # Static data
│   └── results/           # JSON results files
│
└── docker-compose.yml     # Full stack deployment
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/results/comparison` | SSL vs Scratch results |
| `GET /api/results/graphmae` | GraphMAE comparison |
| `GET /api/results/robustness` | Robustness analysis |
| `GET /api/data/statistics/{grid}` | Dataset stats |
| `POST /api/inference/cascade` | Cascade prediction |
| `POST /api/inference/power_flow` | PF prediction |
| `POST /api/inference/line_flow` | LF prediction |

## Tech Stack

- **Frontend**: React 18, Vite, Tailwind CSS v4, D3.js, Framer Motion
- **Backend**: FastAPI, Uvicorn, PyTorch, PyG
- **Deployment**: Docker, Nginx
