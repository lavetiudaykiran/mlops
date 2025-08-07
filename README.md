# 🧠 MLOps Project - California Housing

This project implements a complete MLOps pipeline using the California Housing dataset. The pipeline includes data versioning, model development, experiment tracking, API packaging, CI/CD automation, and monitoring.

---

## ✅ Part 1: Repository and Data Versioning (4 marks)

### 📁 Folder Structure

```
.
├── data/
│   └── raw_data.csv             # Raw dataset (generated)
├── src/
│   ├── load_data.py             # Loads and saves raw dataset
│   └── preprocess.py            # Cleans and splits dataset
├── dvc.yaml                     # DVC metadata (optional for this part)
├── .dvc/                        # DVC cache and config
├── .gitignore
└── requirements.txt
```

### 🛠️ Setup Instructions

#### 1️⃣ Clone the repo and install dependencies

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
```

#### 2️⃣ Load the raw dataset

```bash
python src/load_data.py
```

This creates the `data/raw_data.csv` file.

#### 3️⃣ Track dataset using DVC

```bash
pip install dvc
dvc init
dvc add data/raw_data.csv
git add data/.gitignore data/raw_data.csv.dvc .dvc .gitignore
git commit -m "Track data using DVC"
```

---

## 🔍 Part 2: Model Development & Experiment Tracking (6 marks)

- Implement models in `src/train.py`
- Use MLflow to track experiments: parameters, metrics, and models.
- Select and register the best model.

```bash
python src/train.py
mlflow ui
```

---

## 🌐 Part 3: API & Docker Packaging (4 marks)

- Implement FastAPI app in `app/main.py`
- Build Docker container:

```bash
docker build -t housing-api .
docker run -p 8000:8000 housing-api
```

---

## 🔄 Part 4: CI/CD with GitHub Actions (6 marks)

- Defined in `.github/workflows/main.yml`
- On every push:
  - Run tests
  - Build Docker image
  - Push to DockerHub
  - Optional: Deploy via shell script

---

## 📊 Part 5: Logging and Monitoring (4 marks)

- Logs stored in `logs/predictions.log`
- SQLite used for storing prediction history.
- `/metrics` endpoint exposed for monitoring.

---

## 🧾 Requirements

```bash
pip install -r requirements.txt
```

---

## ✅ Tips

- Run `dvc push` if you want to push tracked data to remote storage.
- To remove data from Git but keep it tracked by DVC: `git rm --cached data/raw_data.csv`