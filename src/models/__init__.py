"""
Models Module
=============

Machine learning models for classification and forecasting.

Classes/Functions:
------------------
- supervised: Classification models (Logistic, Tree, RF, XGBoost)
- semi_supervised: Self-training, Label Propagation
- forecasting: Time series models (ARIMA, SARIMA)
"""

from .supervised import *
from .semi_supervised import *
from .forecasting import *
