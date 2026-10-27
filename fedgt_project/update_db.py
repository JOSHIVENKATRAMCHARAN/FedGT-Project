import os
import django
import pandas as pd

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fedgt_project.settings')
django.setup()

from core.models import ClientData

df = pd.read_csv('core/client_data.csv')

print("Updating database...")
for index, row in df.iterrows():
    ClientData.objects.filter(sample_id=row['sample_id']).update(
        feature_1=row['feature_1'],
        feature_2=row['feature_2'],
        feature_3=row['feature_3'],
        feature_4=row['feature_4'],
        feature_5=row['feature_5'],
        feature_6=row['feature_6'],
        feature_7=row['feature_7'],
        feature_8=row['feature_8'],
        feature_9=row['feature_9'],
        feature_10=row['feature_10']
    )
print("Done.")
