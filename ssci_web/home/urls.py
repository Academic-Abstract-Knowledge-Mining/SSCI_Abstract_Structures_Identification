from django.urls import path
from home import views


urlpatterns = [
    path('', views.home, name='home'),
    path('about', views.about, name='about'),
    path('search', views.search, name="search"),
    path('detail', views.detail, name="search"),
    path("struct", views.struct, name="struct"),
    path("get_struct", views.get_struct, name="get_struct"),
]