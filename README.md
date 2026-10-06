# 🛰️ SATELLITE INTELLIGENCE
### Autonomous Earth Observation Analytics & Geospatial Environmental Platform for India

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![DuckDB](https://img.shields.io/badge/DuckDB-1.1+-FFF000?logo=duckdb&logoColor=black)](https://duckdb.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Zero-Cost Deployment](https://img.shields.io/badge/Deploy%20Free-%240%20Hugging%20Face%20Spaces-FFD21E?logo=huggingface&logoColor=black)](#-free-live-demo-deployment-0-cost)

---

## 🌍 Executive Overview

**SATELLITE INTELLIGENCE** is an enterprise-grade, dataset-adaptive Earth Observation (EO) and geospatial environmental analytics platform. Built from the ground up to address real-world climate risks and remote sensing challenges across the Indian subcontinent, the system combines **DuckDB in-memory columnar OLAP**, **scikit-learn unsupervised machine learning**, and an **embedded reactive Single Page Application (SPA)** with real-time **3D orbital mechanics**.

Whether processing multi-spectral Sentinel-2 bands, Landsat surface reflectance, or climate reanalysis grids, the platform automatically adapts its schema, calculates standard vegetation/water/urban indices, detects environmental anomalies, and renders crisp, official 1:1 vector state boundaries with zero browser lag.

---

## ✨ Key Capabilities & Innovations

### 1. 🌐 3D Live Earth & LEO Constellation Visualizer
* **Procedural 3D Canvas Globe:** Custom high-performance orthographic canvas projection rendering continents, latitude/longitude graticules, atmospheric glow, and axial rotation.
* **650+ Simulated LEO Satellites:** Multi-shell orbital physics modeling **Starlink**, **Polar Sun-Synchronous (SSO)**, **Commercial Earth Observation**, **Scientific Climate**, **GNSS Navigation**, and **Copernicus Sentinel** constellations.
* **Interactive Orbital Mechanics:** Real depth sorting with dynamic z-index sizing, solar illumination shading, orbital trail rendering, and interactive mouse drag rotation.

### 2. 🗺️ 1:1 Official Indian Geospatial Footprints
* **Accurate State & UT Boundaries:** Real boundary vector polygons for all major Indian States & Union Territories (Maharashtra, Karnataka, Tamil Nadu, Delhi NCT, Rajasthan, Gujarat, Kerala, West Bengal, Telangana, Assam, etc.).
* **Zero-Lag Vector Canvas:** Replaces slow third-party web map tiles with native hardware-accelerated 2D canvas drawing.
* **City & State Filtering:** Dynamic dual-selector linking 20 observation hubs with their parent states, showing observation density, spatial coverage radii, and coordinates.

### 3. 🔬 Multi-Spectral Satellite & Agro-Climatic Indices
* **Automated Band Derivation:** Real-time computation of standard radiometric indices:
  * **NDVI** (*Normalized Difference Vegetation Index*): Canopy health and photosynthetic activity.
  * **NDWI** (*Normalized Difference Water Index*): Surface water extent and reservoir storage.
  * **EVI** (*Enhanced Vegetation Index*): High-biomass canopy monitoring with atmospheric resistance.
  * **SAVI** (*Soil-Adjusted Vegetation Index*): Arid and semi-arid crop monitoring with soil brightness compensation.
  * **NDBI** (*Normalized Difference Built-up Index*): Urban sprawl and impervious surface density.
  * **NBR** (*Normalized Burn Ratio*): Wildfire burn scars and agricultural stubble burning.
* **Statistical Distribution:** Kernel-density and histogram binning with standard deviation markers.

### 4. 📈 High-Resolution Temporal Intelligence
* **Adaptive Aggregation:** Rolling time-series grouping by **Daily**, **Weekly**, or **Monthly** intervals.
* **Trend & Volatility Metrics:** 7-period Moving Averages, Ordinary Least Squares (OLS) linear trend slopes, rolling standard deviation bands, and seasonality diagnostics.
* **Composite Time Cards:** Instant computation of baseline mean, peak maximum, trough minimum, net percentage change, and standard deviation over the observation window.

### 5. 🤖 Machine Learning & Risk Intelligence
* **Unsupervised Anomaly Detection:** Spatial and spectral outlier isolation via scikit-learn **Isolation Forest** algorithms.
* **K-Means Clustering:** Multi-dimensional environmental zonation with automatic **Silhouette Score** validation.
* **Composite Environmental Risk Matrix:** Weighted multi-factor risk scoring incorporating vegetation stress, hydrologic deficits, thermal anomalies, and urban expansion.

### 6. 💡 Evidence-Backed Operational Insights
* 8 automated, rule-and-model-backed actionable intelligence cards providing field teams, urban planners, and agricultural departments with prioritized advisories.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph DataIngestion ["Data Ingestion & Adaptation Layer"]
        A[Sentinel-2 / Landsat / CSV / Parquet] --> B[Schema Auto-Detection & Alias Resolution]
        B --> C[DuckDB In-Memory OLAP Engine]
    end

    subgraph AnalyticalEngine ["Computational & ML Core"]
        C --> D[Multi-Spectral Index Engine\nNDVI, NDWI, EVI, SAVI, NDBI, NBR]
        C --> E[Temporal Aggregation & Trend Engine\nMA-7, OLS, Volatility]
        C --> F[scikit-learn ML Pipelines\nIsolation Forest & K-Means]
        C --> G[Composite Risk Matrix & Insights Engine]
    end

    subgraph APILayer ["FastAPI REST Services"]
        D --> H[REST Endpoints\n/api/indices, /api/geospatial]
        E --> H
        F --> H
        G --> H
    end

    subgraph ClientUI ["Zero-Dependency Single Page App"]
        H --> I[3D Live Earth & 650+ LEO Satellite Engine]
        H --> J[1:1 Indian State Vector Canvas]
        H --> K[Analytical Dashboard & Visual Analytics]
    end
```

---

## 🚀 Free Live Demo Deployment ($0 Cost)

You can launch a public, permanent live demo of this dashboard with **zero server costs ($0)** using one of the following methods:

### Option A: Hugging Face Spaces (Recommended — 100% Free, Zero Card Required)

Hugging Face Spaces provides 100% free hosting with 16GB RAM and 2 vCPUs:

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Click **New Space** ([huggingface.co/new-space](https://huggingface.co/new-space)).
3. Name your space (e.g. `satellite-intelligence-india`).
4. Select **Docker** as the Space SDK and choose **Blank**.
5. Choose **Public** and click **Create Space**.
6. Clone your Space repository or push this repository's code to Hugging Face:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/satellite-intelligence-india
   git push space main
   ```
7. Hugging Face will automatically detect the bundled `Dockerfile`, build the image, and provide a live public HTTPS URL (e.g. `https://<your-username>-satellite-intelligence-india.hf.space`).

---

### Option B: Render.com (Free Web Service)

1. Create a free account at [render.com](https://render.com).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository.
4. Set the following settings:
   * **Runtime:** `Python 3` (or `Docker`)
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `python satellite_intelligence.py`
5. Click **Create Web Service**. Your dashboard is live at `https://<service-name>.onrender.com`!

---

### Option C: GitHub Codespaces (1-Click Instant Browser Run)

Every GitHub user receives 60 free hours of GitHub Codespaces per month:
1. In your GitHub repository, click the green **Code** button.
2. Select the **Codespaces** tab and click **Create codespace on main**.
3. In the terminal, run:
   ```bash
   pip install -r requirements.txt
   python satellite_intelligence.py
   ```
4. A notification will pop up: *"Your application running on port 8000 is available"*. Click **Open in Browser** to view the live dashboard!

---

## 💻 Local Quickstart

### Prerequisites
* Python 3.10, 3.11, or 3.12
* Git

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/satellite-intelligence.git
cd satellite-intelligence

# 2. Create and activate a virtual environment
# Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# 3. Install verified dependencies
pip install -r requirements.txt

# 4. (Optional) Configure environment
cp .env.example .env

# 5. Launch the platform
python satellite_intelligence.py
```

Open your browser at **`http://localhost:8000`** to access the live dashboard.

---

## 🐳 Docker Deployment

Run anywhere with the included production Dockerfile:

```bash
# Build the Docker image
docker build -t satellite-intelligence:latest .

# Run the container on port 8000
docker run -d -p 8000:8000 --name satellite-platform satellite-intelligence:latest
```

---

## 📡 REST API Reference

The platform includes a self-documenting FastAPI REST interface. Interactive OpenAPI documentation is accessible at **`http://localhost:8000/docs`**.

| Endpoint | Method | Description | Key Parameters |
| :--- | :--- | :--- | :--- |
| `/api/health` | `GET` | Healthcheck and active system status | None |
| `/api/summary` | `GET` | High-level dataset statistics and metrics | None |
| `/api/indices` | `GET` | Derived multi-spectral radiometric indices | `index`, `region` |
| `/api/geospatial` | `GET` | 1:1 Indian state and city geospatial telemetry | `state`, `city` |
| `/api/temporal` | `GET` | Time-series trends and moving averages | `indicator`, `granularity` |
| `/api/anomalies` | `GET` | Unsupervised Isolation Forest outlier scores | `contamination` |
| `/api/clusters` | `GET` | Multi-spectral K-Means spatial clustering | `k` |
| `/api/risk-matrix` | `GET` | Multi-hazard environmental risk ranking | None |
| `/api/insights` | `GET` | Evidence-backed operational advisories | None |
| `/api/export` | `GET` | Export analytical subsets as CSV or Parquet | `format=csv\|parquet` |

---

## ⚙️ Configuration & Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `APP_HOST` / `HOST` | `127.0.0.1` | Network interface to bind to (`0.0.0.0` in cloud/Docker) |
| `APP_PORT` / `PORT` | `8000` | Port for the HTTP ASGI server |
| `SATELLITE_DATA_PATH` | *Empty* | Custom path to local CSV or Parquet telemetry file |
| `CACHE_TTL` | `300` | In-memory analytical cache expiration in seconds |
| `MAX_UPLOAD_MB` | `500` | Maximum file size for upload via web interface |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `AUTO_DOWNLOAD_INDIA_DATA`| `true` | Automated synthesis of India 20-city satellite telemetry |

---

## 🗺️ Geospatial Coverage (India)

The platform includes 1:1 official vector borders and meteorological telemetry across 20 primary observation centers:

* **Northern Region:** Srinagar (J&K), Delhi NCT, Jaipur (Rajasthan), Lucknow (Uttar Pradesh).
* **Western Region:** Ahmedabad (Gujarat), Mumbai (Maharashtra), Pune (Maharashtra), Nagpur (Maharashtra).
* **Central & Eastern Region:** Bhopal (Madhya Pradesh), Patna (Bihar), Ranchi (Jharkhand), Kolkata (West Bengal), Bhubaneswar (Odisha).
* **Southern Region:** Hyderabad (Telangana), Bengaluru (Karnataka), Chennai (Tamil Nadu), Kochi (Kerala), Thiruvananthapuram (Kerala).
* **Northeastern & Island Territories:** Guwahati (Assam), Port Blair (Andaman & Nicobar).

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>Built with ❤️ for Earth Observation, Climate Resilience, and Advanced Geospatial Intelligence.</b>
</p>
