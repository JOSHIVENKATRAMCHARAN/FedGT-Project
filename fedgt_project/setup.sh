#!/bin/bash
echo "=== FedGT Setup ==="
pip install django scikit-learn pandas numpy matplotlib seaborn
python manage.py migrate
python manage.py shell -c "
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@fedgt.com', 'admin123')
    print('Created admin user: admin / admin123')
else:
    print('Admin user already exists')
"
echo "=== Setup complete. Run: python manage.py runserver ==="
