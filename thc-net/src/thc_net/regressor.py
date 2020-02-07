import numpy as np
from sklearn.base import RegressorMixin

from thc_net.abstract_estimator import AbstractThcNetEstimator
from sklearn.preprocessing import StandardScaler


class ThcNetRegressor(AbstractThcNetEstimator, RegressorMixin):
    """
    Scikit regressor for Keras MLP
    """

    def __init__(
        self,
        n_layer,
        mul_input,
        loss="auto",
        activation=None,
        metrics=[],
        cat_idxs=[],
        cat_emb_dims=[],
        dropout=0.5,
        normalize=False,
        max_emb=50,
        noise=0.01,
        patience=20,
        use_snn=False,
        target_scaler_class=StandardScaler
    ):
        """
        Constructor (see AbstractThcNetEstimator for other infos)
        
        Additional parameters
        ----------
        loss : str, optional
            loss to use to train the classifier, by default "auto"
            "auto" is using mean_squared_error
        activation : str, optional
            activation function to use for last layer, by default None
        """

        super().__init__(
            n_layer=n_layer,
            mul_input=mul_input,
            metrics=metrics,
            cat_idxs=cat_idxs,
            cat_emb_dims=cat_emb_dims,
            dropout=dropout,
            normalize=normalize,
            max_emb=max_emb,
            noise=noise,
            patience=patience,
            use_snn=use_snn
        )
        self.loss = loss
        self.activation = activation
        self.scaler = target_scaler_class()

    def fit(self, *, X, y, X_valid=None, y_valid=None, **kwargs):
        self.out_dim = 1
        self.in_dim = X.shape[1]

        if self.loss == "auto":
            self.loss = "mean_squared_error"

        self.scaler.fit(y.reshape(-1, 1))

        return super().fit(
            X=X,
            y=self.scaler.transform(y.reshape(-1, 1)).reshape(-1),
            X_valid=X_valid,
            y_valid=self.scaler.transform(y_valid.reshape(-1, 1)).reshape(-1)
            if y_valid is not None
            else None,
            **kwargs
        )

    def predict(self, X):
        return self.scaler.inverse_transform(super().predict(X).reshape(-1, 1)).reshape(
            -1
        )

    def predict_proba(self, X):
        return self.predict(X)
