import numpy as np
from sklearn.base import ClassifierMixin
from tensorflow.keras.utils import to_categorical

from thc_net.abstract_estimator import AbstractThcNetEstimator

class ThcNetClassifier(AbstractThcNetEstimator, ClassifierMixin):
    """
    Scikit classifier for Keras MLP
    """

    def __init__(
        self,
        n_layer,
        mul_input,
        loss="auto",
        activation="softmax",
        metrics=[],
        cat_idxs=[],
        cat_emb_dims=[],
        dropout=0.5,
        normalize=False,
        max_emb=50,
        noise=0.01,
        patience=20,
        use_snn=False
    ):
        """
        Constructor (see AbstractThcNetEstimator for other infos)
        
        Additional parameters
        ----------
        loss : str, optional
            loss to use to train the classifier, by default "auto"
            "auto" is using either binary_crossentropy or sparse_categorical_crossentropy
        activation : str, optional
            activation function to use for last layer, by default "softmax"
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

    def fit(self, *, X, y, X_valid=None, y_valid=None, **kwargs):
        self.out_dim = len(np.unique(y))
        self.in_dim = X.shape[1]

        input_y = y
        input_y_valid = y_valid

        loss = self.loss
        if loss == "auto":
            if self.out_dim == 2:
                loss = "binary_crossentropy"
                input_y = to_categorical(input_y)
                if input_y_valid is not None:
                    input_y_valid = to_categorical(input_y_valid)
            else:
                loss = "sparse_categorical_crossentropy"
        self.loss = loss

        return super().fit(
            X=X, y=input_y, X_valid=X_valid, y_valid=input_y_valid, **kwargs
        )

    def predict_proba(self, X):
        return super().predict(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
