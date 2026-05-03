"""CV_04 URL configuration."""

from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

from detector import views

urlpatterns = [
    path("",                        views.home,                  name="home"),
    path("api/upload/",             views.api_upload,            name="api_upload"),
    # Thresholding
    path("api/optimal-threshold/",  views.api_optimal_threshold, name="api_optimal_threshold"),
    path("api/otsu-threshold/",     views.api_otsu_threshold,    name="api_otsu_threshold"),
    path("api/spectral-threshold/", views.api_spectral_threshold,name="api_spectral_threshold"),
    path("api/local-threshold/",    views.api_local_threshold,   name="api_local_threshold"),
    # Segmentation
    path("api/kmeans/",             views.api_kmeans,            name="api_kmeans"),
    path("api/region-growing/",     views.api_region_growing,    name="api_region_growing"),
    path("api/agglomerative/",      views.api_agglomerative,     name="api_agglomerative"),
    path("api/mean-shift/",         views.api_mean_shift,        name="api_mean_shift"),
    # Save
    path("api/save-result/",        views.api_save_result,       name="api_save_result"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
