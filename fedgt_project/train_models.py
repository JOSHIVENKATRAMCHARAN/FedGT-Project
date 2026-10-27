import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fedgt_project.settings')
django.setup()

from core.ml_engine import train_all_models

print("Training all models...")
results = train_all_models()
for r in results:
    print(f"{r.name}: Acc={r.accuracy}, Prec={r.precision}, Rec={r.recall}, F1={r.f1_score}")
