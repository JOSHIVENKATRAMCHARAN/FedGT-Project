# FedGT — Federated Learning Security Platform

A Django-based college-level machine learning application for detecting malicious clients in Federated Learning using the FedGT algorithm.

## Quick Start

### 1. Install Dependencies
```bash
pip install django scikit-learn pandas numpy matplotlib seaborn
```

### 2. Apply Migrations
```bash
python manage.py migrate
```

### 3. Create Admin User
```bash
python manage.py createsuperuser
```
Or use the pre-created credentials: **admin / admin123**

### 4. Run the Server
```bash
python manage.py runserver
```

Visit: http://127.0.0.1:8000

## Pages

| Page | URL | Description |
|------|-----|-------------|
| Login | /login/ | Admin authentication |
| Dashboard | / | Overview stats & charts |
| Client Monitor | /clients/ | All 15 federated clients |
| Group Viewer | /groups/ | FedGT group assignments & test results |
| FL Training | /training/ | Training accuracy & loss per round |
| ML Models | /models/ | Train & compare 4 ML models |
| Attack Analyzer | /attacks/ | Poisoning attack detection |
| Detection | /detection/ | Run predictions on new samples |
| Reports | /reports/ | Full analytics report |

## ML Models
- Random Forest
- Logistic Regression
- Support Vector Machine
- K-Nearest Neighbors

## Dataset
- 5000 samples across 15 clients
- 10 features per sample
- 10 class labels (0–9)
- 4 malicious clients (label_flip, data_poisoning attacks)
- 20 federated learning rounds, 8 groups

## Tech Stack
- **Backend**: Django 4.x
- **ML**: scikit-learn
- **Data**: Pandas, NumPy
- **Visualization**: Chart.js (frontend), Matplotlib + Seaborn (confusion matrix)
- **DB**: SQLite
- **Frontend**: Bootstrap 5 + Font Awesome
