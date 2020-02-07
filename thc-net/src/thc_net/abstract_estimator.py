import gc

import numpy as np
from sklearn.base import BaseEstimator

from tensorflow.keras.callbacks import EarlyStopping

from thc_net.model_utils import build_keras_mlp, build_keras_snn
from thc_net.input_utils import prepare_input_data

SEED = 42


class AbstractThcNetEstimator(BaseEstimator):
    """
    Abstract class for Keras based classifier 
    """

    def __init__(
        self,
        n_layer,
        mul_input,
        metrics=[],
        cat_idxs=[],
        cat_emb_dims=[],
        dropout=0.5,
        normalize=False,
        max_emb=50,
        noise=0.001,
        patience=20,
        use_snn=False,
    ):
        """
        Constructor.
        
        Parameters
        ----------
        n_layer : int
            The number of hidden layer.
        mul_input : float
            The multiplicative factor to determine the size of the first hidden layer
            ie, with 10 features and 8, will be 80
            or with 5 categorical features (dim 5), 5 other features and mul 8, will be 240
        metrics : list, optional
            list of Keras metrics, by default []
        cat_idxs : list, optional
            list of indexs for categorical features, by default []
        cat_emb_dims : list, optional
            list of unique values for each categorical, by default []
        dropout : float, optional
            dropout value to be used, by default 0.5
        normalize : bool, optional
            should batch normalisation be used, by default False
        max_emb : int, optional
            maximum size of embeddings, by default 50
        noise : float, optional
            should gaussian noise be added, and which ratio, by default 0.1
        patience : int, optional
            should use early stopping, and which patience, by default 20
        use_snn : bool, optional
            should use self normalizing network, by default False

        """
        self.n_layer = n_layer
        self.mul_input = mul_input
        self.cat_idxs = cat_idxs
        self.metrics = metrics
        self.cat_idxs = cat_idxs
        self.cat_emb_dims = cat_emb_dims
        self.dropout = dropout
        self.normalize = normalize
        self.max_emb = max_emb
        self.noise = noise
        self.patience = patience
        self.use_snn = use_snn

    def fit(self, *, X, y, X_valid=None, y_valid=None, callbacks=[], **kwargs):
        """
        Fit method, to train the model
        
        Parameters
        ----------
        X : numpy matrice
            Training data
        y : numpy array
            target for training
        X_valid : numpy matrice, optional
            validation data, by default None
        y_valid : numpy array, optional
            validation data, by default None
        callbacks : list, optional
            Keras callbacks, by default []
        
        Returns
        -------
        history
            Keras fit history
        """

        if self.use_snn:
            self.network = build_keras_snn(
                self.n_layer,
                self.in_dim,
                self.out_dim,
                self.mul_input,
                self.loss,
                self.activation,
                self.metrics,
                self.cat_emb_dims,
                self.dropout,
                self.max_emb,
                self.noise,
            )
        else:
            self.network = build_keras_mlp(
                self.n_layer,
                self.in_dim,
                self.out_dim,
                self.mul_input,
                self.loss,
                self.activation,
                self.metrics,
                self.cat_emb_dims,
                self.dropout,
                self.normalize,
                self.max_emb,
                self.noise,
            )

        prepared_inputs = self.prepare_input_(X)
        val_data = []

        if X_valid is not None and y_valid is not None:
            prepared_val_inputs = self.prepare_input_(X_valid)
            val_data = [prepared_val_inputs, y_valid]

        all_cbs = [*callbacks]
        if self.patience is not None:
            es = EarlyStopping(
                monitor="val_loss",
                verbose=1,
                mode="min",
                patience=self.patience,
                restore_best_weights=True,
            )
            es.set_model(self.network)
            all_cbs.append(es)

        history = self.network.fit(
            prepared_inputs, y, validation_data=val_data, callbacks=all_cbs, **kwargs
        )


        gc.collect()

        return history

    def predict(self, X):
        """
        Inferring method
        
        Parameters
        ----------
        X : numpy matrice
            data to infer
        
        Returns
        -------
        numpy array
            inferrence results
        """
        return self.network.predict(self.prepare_input_(X))

    def predict_proba(self, X):
        """
        Inferring method
        
        Parameters
        ----------
        X : numpy matrice
            data to infer
        
        Returns
        -------
        numpy array
            inferrence results
        """
        return self.predict(X)

    def prepare_input_(self, x):
        inputs = []

        for idx in self.cat_idxs:
            inputs.append(x[:, idx])
        inputs.append(np.delete(x, self.cat_idxs, axis=1))

        return inputs
