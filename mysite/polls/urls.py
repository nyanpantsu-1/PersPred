from . import views
from django.urls import path

urlpatterns=[
    path("",views.index,name="index"),
    path("one",views.first,name="first"),
    path("two",views.search,name="search")
]