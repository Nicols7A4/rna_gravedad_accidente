"""
neurox — biblioteca de redes neuronales desde cero
====================================================
Uso básico:

    from neurox import Network, Dense

    model = Network(
        layers=[
            Dense(6, 16, activation='relu'),
            Dense(16, 1, activation='sigmoid')
        ],
        cost='binary_crossentropy'
    )

    model.train(X_train, y_train, epochs=1000, alpha=0.05)
    model.plot_cost()
    result = model.evaluate(X_test, y_test)
    print(result)
"""

from neurox.layers        import Dense, Dropout
from neurox.network       import Network
from neurox.activations   import list_activations
from neurox.costs         import list_costs
from neurox.preprocessing import (
    Normalizer, normalize,
    one_hot, LabelEncoder,
    train_test_split, to_numpy,
    k_fold_split
)

__all__ = [
    'Dense', 'Dropout', 'Network',
    'list_activations', 'list_costs',
    'Normalizer', 'normalize',
    'one_hot', 'LabelEncoder',
    'train_test_split', 'to_numpy',
    'k_fold_split',
]
__version__ = '0.1.0'