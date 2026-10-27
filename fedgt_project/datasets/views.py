from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from .services import extract_dataset_zip


@login_required
def upload_dataset(request):
    result = None
    if request.method == 'POST':
        dataset_zip = request.FILES.get('dataset_zip')
        replace = request.POST.get('replace') == 'on'
        if not dataset_zip:
            messages.error(request, 'Choose a FedGT dataset ZIP first.')
        elif not dataset_zip.name.lower().endswith('.zip'):
            messages.error(request, 'Upload a .zip file containing the six required CSV files.')
        else:
            try:
                result = extract_dataset_zip(dataset_zip)
                if replace:
                    messages.success(request, 'Dataset extracted and imported into the database.')
                else:
                    messages.success(request, 'Dataset extracted and imported into the database.')
            except Exception as exc:
                messages.error(request, f'Dataset import failed: {exc}')

    return render(request, 'datasets/upload.html', {'result': result})

# Create your views here.
