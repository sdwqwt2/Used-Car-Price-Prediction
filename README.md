# Used Car Price Prediction — MLOps Pipeline

An automated MLOps pipeline that cleans/splits used-car listing data, trains
a `RandomForestRegressor` to predict selling price, and deploys the model
behind a FastAPI service with a Streamlit front end. The full pipeline is
orchestrated by Airflow and runs every 5 minutes.

## Repository structure

```
├── code
│   ├── datasets/            # Stage 1: data engineering
│   ├── models/               # Stage 2: model engineering
│   └── deployment/
│       ├── api/               # FastAPI model-serving service
│       └── app/                # Streamlit web app
│       └── docker-compose.yml
├── data
│   ├── raw/                    # Place used_cars.csv here
│   └── processed/            # train.csv / test.csv (generated)
├── models/                    # model.joblib (generated)
├── services/airflow/
│   ├── dags/                     # used_car_pipeline_dag.py
│   └── logs/
└── requirements.txt
```

## 1. Setup

```bash
git clone <your-repo-url>
cd used-car-price-prediction
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the dataset

Download a "Used Car Price Prediction" / CarDekho-style dataset (e.g. from
Kaggle) and place it at:

```
data/raw/used_cars.csv
```

Required columns (rename in `code/datasets/data_pipeline.py::COLUMN_MAP` if
yours differ): `name, year, selling_price, km_driven, fuel, seller_type,
transmission, owner`.

## 3. Run the pipeline manually (without Airflow)

```bash
python code/datasets/data_pipeline.py      # -> data/processed/train.csv, test.csv
python code/models/train.py                # -> models/model.joblib, MLflow run in ./mlruns
cd code/deployment
docker compose build
docker compose up -d
```

- API: http://localhost:8000/docs (Swagger UI, `POST /predict`)
- App: http://localhost:8501

To inspect training runs/metrics:

```bash
mlflow ui --backend-store-uri ./mlruns
```

## 4. Run the automated pipeline with Airflow (every 5 minutes)

```bash
export AIRFLOW_HOME=$(pwd)/services/airflow
export PROJECT_ROOT=$(pwd)
airflow db init
airflow users create --username admin --password admin \
  --firstname a --lastname b --role Admin --email admin@example.com

# copy/symlink the DAG so Airflow can find it (AIRFLOW_HOME/dags is the default)
# services/airflow/dags already matches this path

airflow webserver -p 8080 &
airflow scheduler &
```

Open http://localhost:8080, enable the `used_car_price_pipeline` DAG. It runs
`data_engineering -> model_engineering -> deployment` every 5 minutes,
rebuilding and restarting the API/app containers with the freshly trained
model each cycle. If a run takes longer than 5 minutes, increase
`schedule_interval` in `services/airflow/dags/used_car_pipeline_dag.py`.

## 5. Stopping

```bash
cd code/deployment && docker compose down
```

## Notes

- The API and app run in **separate Docker containers** (see
  `code/deployment/docker-compose.yml`); the app calls the API over the
  Docker network at `http://api:8000`.
- The trained model (`models/model.joblib`) is git-ignored — regenerate it
  by running Stage 1 + Stage 2, or via the Airflow DAG.
