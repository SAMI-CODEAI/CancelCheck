# 🏨 Hotel Booking Cancellation Predictor

Predict whether a hotel reservation will be **cancelled** or **not** using a full MLOps pipeline:
- Data from **Google Cloud Storage (GCS)**
- Processing & training (LightGBM)
- Packaging with **Docker**
- CI/CD with **Jenkins**
- Deployment to **Cloud Run**



## ✨ What I Built

1. **End-to-End ML Pipeline**
   - **Data ingestion** from GCS → `artifacts/raw/`
   - **Preprocessing** (encoding, skew handling, SMOTE, feature selection) → `artifacts/processed/`
   - **Model training** (LightGBM + RandomizedSearchCV) → `artifacts/models/lgbm_model.pkl`
   - **MLflow** logging (datasets, params, metrics, model)

2. **Web App**
   - Flask server with a clean HTML/CSS form
   - Real-time prediction using the saved model

3. **Dockerized & Cloud Native**
   - Production image
   - Deployed to **Cloud Run** (serverless)

4. **CI/CD with Jenkins**
   - Clone → setup venv → train (optional) → build & push to **GCR** → deploy to **Cloud Run**


## 📁 Repository Structure

```

sami-codeai-hotel\_reservation\_prediction/
├── application.py                 # Flask app (serves predictions)
├── Dockerfile                     # Container image for Cloud Run
├── Jenkinsfile                    # Pipeline: build → push → deploy
├── requirements.txt               # Python deps
├── setup.py                       # Editable install
├── README.md                      # (this file)
│
├── config/
│   ├── config.yaml                # Ingestion & processing config (bucket, columns, etc.)
│   ├── model\_params.py            # LightGBM + RandomSearch params
│   └── paths\_config.py            # Canonical paths for artifacts
│
├── pipeline/
│   └── training\_pipeline.py       # Orchestrates ingestion → processing → training
│
├── src/
│   ├── data\_ingestion.py          # Download from GCS + train/test split
│   ├── data\_preprocessing.py      # Encoding, skew handling, SMOTE, feature selection
│   ├── model\_training.py          # Train, evaluate, save model, MLflow logs
│   ├── logger.py                  # Daily rotating file logs
│   └── custom\_exception.py        # Exception wrapper with context
│
├── templates/index.html           # UI
├── static/style.css               # Styles
└── utils/common\_functions.py      # read\_yaml(), load\_data()

````

---

## 🧠 End-to-End Workflow (Step-by-Step)

```mermaid
flowchart TD
    A[Upload CSV to GCS] --> B[Run training_pipeline.py]
    B --> C[artifacts/raw: raw/train/test]
    C --> D[Preprocess: encode, skew, SMOTE, select features]
    D --> E[artifacts/processed: processed_train/test]
    E --> F[Train LightGBM + RandomizedSearchCV]
    F --> G[Evaluate + MLflow log]
    G --> H[Save model: artifacts/models/lgbm_model.pkl]
    H --> I[Docker build image]
    I --> J[Push to gcr.io/PROJECT/ml-project:latest]
    J --> K[Deploy to Cloud Run]
    K --> L[Flask serves predictions]
````

---

## 🔧 Local Setup (Dev/Training)

### 1) Python environment

```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

### 2) GCP prerequisites

* A Google Cloud **project ID** (e.g., `hidden-phalanx-464505-h0`)
* Enable APIs:

  * **Cloud Run Admin API**
  * **Cloud Build API** (optional for Cloud Build)
  * **Container Registry API** (or **Artifact Registry API** if you migrate)
  * **Cloud Storage API**
* Create **Cloud Storage bucket** (example): `my_buckethotel`
  Upload dataset: `Hotel_Reservations.csv`

```bash
# Example using gcloud
gsutil mb -l us-central1 gs://my_buckethotel/
gsutil cp Hotel_Reservations.csv gs://my_buckethotel/Hotel_Reservations.csv
```

### 3) Service Account & Roles

Create a **service account** (e.g., `mlops-ci@PROJECT_ID.iam.gserviceaccount.com`) and grant roles (min set for CI/CD and data access):

* `roles/storage.objectViewer` (read dataset)
* `roles/storage.admin` (if you need to manage buckets/objects)
* `roles/run.admin` (deploy to Cloud Run)
* `roles/iam.serviceAccountUser` (act-as for Cloud Run deploy)
* `roles/storage.objectAdmin` (for Container Registry images if needed)
* *(If using Artifact Registry)* `roles/artifactregistry.writer`

Download the **JSON key** for local dev and Jenkins:

```
mlops-ci-key.json
```

Set locally:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/mlops-ci-key.json"
# Windows PowerShell: $env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\mlops-ci-key.json"
```

### 4) Configure `config/config.yaml`

Ensure:

```yaml
data_ingestion:
  bucket_name: "my_buckethotel"
  bucket_file_name: "Hotel_Reservations.csv"
  train_ratio: 0.8
```

### 5) Run the training pipeline

```bash
python pipeline/training_pipeline.py
```

Outputs:

* `artifacts/raw/` → `raw.csv`, `train.csv`, `test.csv`
* `artifacts/processed/` → `processed_train.csv`, `processed_test.csv`
* `artifacts/models/` → `lgbm_model.pkl`
* `mlruns/` (MLflow local artifacts folder)

### 6) Launch the web app locally

```bash
python application.py
# App listens on port 8080 (http://127.0.0.1:8080)
```

---

## 🐳 Docker (Build & Run Locally)

> **Important fix:** In your Dockerfile, change `EXPOSE 5000` → `EXPOSE 8080` (your Flask app runs on 8080).

**Build:**

```bash
docker build -t hotel-ml:latest .
```

**Run:**

```bash
docker run -p 8080:8080 hotel-ml:latest
```

---

## ☁️ CI/CD with Jenkins → GCR → Cloud Run

### 0) Jenkins Node Requirements

* Docker engine available (Jenkins user in `docker` group)
* Google Cloud SDK (`gcloud`) available on PATH **or** use a Jenkins agent image with Cloud SDK preinstalled
* Credentials configured in Jenkins:

| ID                  | Type                       | What it is                                         |
| ------------------- | -------------------------- | -------------------------------------------------- |
| `Hotel_Reservation` | Username/Password or Token | GitHub credentials for the repo                    |
| `gcp-key`           | **Secret file**            | The service account JSON key (`mlops-ci-key.json`) |

> Your repo also contains `custom_jenkins/Dockerfile` which installs Docker inside the Jenkins image.
> You **still need** to install the **Google Cloud SDK** in the Jenkins container or use an image that already has it.

### 1) Jenkinsfile Overview

Pipeline stages provided:

1. **Clone** the GitHub repository
2. **Setup venv & install deps**
3. **Build & Push Docker image** to `gcr.io/${GCP_PROJECT}/ml-project:latest`
4. **Deploy to Cloud Run** (region `us-central1`)

> **Tip (training)**: training currently runs during Docker build (see Dockerfile).
> This requires credentials **inside the Docker build** to download from GCS, which is not ideal.
> Prefer **training before build** (Jenkins stage), then the Dockerfile just copies the `artifacts/` folder.

#### Recommended Adjustments

**Option A (Simpler & Secure) – Train in Jenkins, then build image**

* **Remove** this line from Dockerfile:

  ```dockerfile
  RUN python pipeline/training_pipeline.py
  ```
* **Add training stage** in Jenkins **before** docker build:

  ```groovy
  stage('Train model') {
    steps {
      withCredentials([file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
        sh '''
          . ${VENV_DIR}/bin/activate
          python pipeline/training_pipeline.py
        '''
      }
    }
  }
  ```
* **Ensure** `artifacts/models/lgbm_model.pkl` exists in workspace so Docker can copy it.

**Option B – Keep training in Docker build (advanced)**

* Use Docker BuildKit secrets to mount the key during build:

  ```bash
  DOCKER_BUILDKIT=1 docker build \
    --secret id=gcpkey,src=/path/to/mlops-ci-key.json \
    -t gcr.io/$GCP_PROJECT/ml-project:latest .
  ```
* Update Dockerfile:

  ```dockerfile
  # syntax=docker/dockerfile:1.4
  RUN --mount=type=secret,id=gcpkey,target=/tmp/key.json \
      export GOOGLE_APPLICATION_CREDENTIALS=/tmp/key.json && \
      python pipeline/training_pipeline.py
  ```
* **Do not** bake the key into the image.

### 2) Fix Dockerfile Port

```dockerfile
# Expose the port that Flask will run on
EXPOSE 8080
```

### 3) Fix GCLOUD\_PATH in Jenkinsfile

Your Jenkinsfile sets:

```groovy
GCLOUD_PATH = "D:/Softwares/google-cloud-sdk/bin"  // Windows path
```

If your Jenkins agent runs on Linux, set it to something like:

```groovy
GCLOUD_PATH = "/usr/bin" // or leave PATH as-is if gcloud is already available
```

### 4) Build & Push (Jenkins)

The Jenkinsfile already runs:

```bash
gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
gcloud config set project ${GCP_PROJECT}
gcloud auth configure-docker --quiet
docker build -t gcr.io/${GCP_PROJECT}/ml-project:latest .
docker push gcr.io/${GCP_PROJECT}/ml-project:latest
```

### 5) Deploy to Cloud Run (Jenkins)

The Jenkinsfile runs:

```bash
gcloud run deploy ml-project \
  --image=gcr.io/${GCP_PROJECT}/ml-project:latest \
  --platform=managed \
  --region=us-central1 \
  --allow-unauthenticated
```

**Ensure Cloud Run picks up the correct port (8080).**
Cloud Run uses the `PORT` env variable; Flask is already set to port `8080`. You’re good.

---

## 🧩 Configuration Details

### Data Processing choices

* **Categoricals:** label encoding (per `config.yaml`)
* **Numericals:** skewness correction (`np.log1p`) for skew > `skewness_threshold`
* **Imbalance:** **SMOTE** on training data
* **Feature selection:** top N features by **RandomForest feature\_importances\_** (`no_of_features`)

### Model

* **LightGBM** classifier with **RandomizedSearchCV**
* Metrics: `accuracy`, `precision`, `recall`, `f1`
* Artifacts & metrics logged to **MLflow** (local `mlruns/`)

### Web App (Flask)

* Loads `artifacts/models/lgbm_model.pkl` at startup
* Accepts numeric inputs from form and makes predictions
* Runs on **port 8080**

---

## 🚦 Running MLflow UI (optional)

```bash
# from repo root (will detect ./mlruns)
pip install mlflow
mlflow ui --port 5001
# open http://127.0.0.1:5001
```

---

## 🧪 Quick Verification Checklist

* **Local training** creates:

  * `artifacts/models/lgbm_model.pkl`
  * `artifacts/processed/processed_test.csv`
* **Docker build** succeeds and runs:

  * `curl http://localhost:8080/` returns the form HTML
* **Jenkins**:

  * `gcloud` found in PATH
  * Credentials `gcp-key` recognized
  * Image pushed to `gcr.io/PROJECT/ml-project:latest`
* **Cloud Run**:

  * Service reachable without auth (if `--allow-unauthenticated`)
  * App responds on **8080**

---

## ⚠️ Important Notes (Avoid Common Pitfalls)

1. **Port mismatch**

   * Dockerfile originally `EXPOSE 5000` but Flask uses **8080** → set to `EXPOSE 8080`.

2. **Training inside Docker build**

   * Requires **GCS credentials inside build**. Prefer training **before** build (Jenkins stage), then just copy artifacts.

3. **GCP SDK on Jenkins**

   * The pipeline uses `gcloud`. Ensure Cloud SDK is installed on the Jenkins agent or use an image that includes it.

4. **Dataset file path**

   * `config.yaml` must match your bucket + CSV name:

     * `my_buckethotel`, `Hotel_Reservations.csv`

5. **Categorical mappings at inference**

   * Training uses `LabelEncoder` without persisting encoders.
   * The web form hard-codes integers for categories. Ensure these integer codes **match** the label encoding learned during training.
   * **Recommended improvement:** persist the encoding mappings during training and apply them in `application.py` before prediction.

---

## 📈 Future Improvements

* Persist and load **encoders/pipelines** (sklearn `ColumnTransformer` + `Pipeline`) to guarantee consistent inference
* Use **Artifact Registry** (instead of deprecated Container Registry) for images
* Add **unit tests** and a **Makefile**
* Add **structured logging** + Cloud Logging integration
* Remote **MLflow Tracking Server** (+ GCS backend store)
* Canary deploys / rollbacks via Cloud Deploy or GitHub Actions

---

## 🧑‍💻 How to Contribute

1. Fork and clone
2. Create a branch: `feat/your-feature`
3. Commit changes, open a PR
4. Ensure:

   * `black`/`flake8` (style)
   * `pytest` (if tests added)


