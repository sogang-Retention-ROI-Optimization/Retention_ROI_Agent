# Retention ROI Dashboard + Cohort Analysis + Backend API


## 핵심 방향

- Dashboard는 `data/raw/*.csv`를 **직접 읽습니다**.
- Cohort 분석은 `src/simulator/cohort_analysis.py` 및 `rebuild_cohort_retention.py` 기준으로 동작합니다.
- FastAPI는 같은 `data/raw` 산출물을 읽어 API로도 제공합니다.
- 따라서 **대시보드와 API가 동일한 산출물 소스**를 공유합니다.

## 1) 설치

```bash
pip install -r requirements.txt
```

## 2) Docker로 한 번에 실행

```bash
docker compose up --build
```

실행 후:

- Dashboard: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

## 3) 로컬 실행

### Dashboard

```bash
streamlit run dashboard/app.py
```

### API

```bash
python scripts/run_api.py
```

## 4) 시뮬레이터 CSV 생성

프로젝트 루트에서:

```bash
python src/main.py --mode simulate
```

또는:

```bash
python3 -c "from src.simulator.pipeline import run_simulation; run_simulation(export=True)"
```

생성 파일은 기본적으로 `data/raw/` 아래 저장됩니다.

## 5) 코호트 리텐션 CSV만 재생성

```bash
python -m src.simulator.rebuild_cohort_retention
```

생성/갱신 파일:

- `data/raw/cohort_retention.csv`

## 6) 통합 파이프라인 진입점

### 이탈 예측 학습

```bash
python src/main.py --mode train
```

출력:

- `models/churn_model.joblib`
- `results/churn_auc_roc.png`
- `results/churn_shap_summary.png`
- `results/churn_metrics.json`

### Uplift 세그먼테이션

```bash
python src/main.py --mode uplift
```

출력:

- `results/uplift_segmentation.csv`
- `results/uplift_summary.json`

### 예산 최적화

```bash
python src/main.py --mode optimize --budget 50000000
```

출력:

- `results/optimization_selected_customers.csv`
- `results/optimization_segment_budget.csv`
- `results/optimization_summary.json`

## 7) Dashboard 데이터 로딩 방식

Dashboard는 기본적으로 아래 파일을 직접 읽습니다.

- `data/raw/customer_summary.csv`
- `data/raw/cohort_retention.csv`

추가 raw 파일(`customers.csv`, `events.csv`, `orders.csv` 등)이 있으면 함께 로드하지만,
현재 주요 화면 렌더링의 핵심은 위 두 파일입니다.

## 8) API 엔드포인트

- `GET /health`
- `GET /tables`
- `GET /api/v1/analytics/summary`
- `GET /api/v1/analytics/churn`
- `GET /api/v1/analytics/cohort-retention`
- `GET /api/v1/analytics/uplift/top`
- `GET /api/v1/analytics/retention-targets`
- `GET /api/v1/analytics/optimization/budget`
- `POST /api/v1/simulation/run`
- `POST /api/v1/pipeline/train`
- `POST /api/v1/pipeline/uplift`
- `POST /api/v1/pipeline/optimize`

## 9) LLM 요약/Q&A 기능

권장: API 키를 환경변수로 설정합니다.

```bash
export OPENAI_API_KEY="your-api-key"
streamlit run dashboard/app.py
```

또는 대시보드 사이드바에서 런타임에 직접 입력할 수 있습니다.


