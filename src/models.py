import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model, Sequential

def build_baseline_mlp(input_dim=30, learning_rate=0.001):
    """
    Model 1: Baseline Shallow MLP
    A straightforward single-hidden-layer feedforward network.
    """
    model = Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu', name='hidden_1'),
        layers.Dense(1, activation='sigmoid', name='output')
    ], name='Baseline_Shallow_MLP')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_deep_mlp(input_dim=30, learning_rate=0.001):
    """
    Model 2: Deep MLP
    A 3-layer deep network without explicit regularization to evaluate deeper feature hierarchies.
    """
    model = Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu', name='hidden_1'),
        layers.Dense(64, activation='relu', name='hidden_2'),
        layers.Dense(32, activation='relu', name='hidden_3'),
        layers.Dense(1, activation='sigmoid', name='output')
    ], name='Deep_MLP')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_regularized_mlp(input_dim=30, learning_rate=0.001, l2_reg=1e-4, dropouts=(0.3, 0.2, 0.1)):
    """
    Model 3: Regularized Deep MLP
    Equipped with Batch Normalization, Dropout, and L2 Weight Decay to curb overfitting.
    """
    inputs = layers.Input(shape=(input_dim,), name='input')
    
    # Block 1
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(l2_reg), use_bias=False, name='dense_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Activation('relu', name='act_1')(x)
    x = layers.Dropout(dropouts[0], name='drop_1')(x)
    
    # Block 2
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(l2_reg), use_bias=False, name='dense_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Activation('relu', name='act_2')(x)
    x = layers.Dropout(dropouts[1], name='drop_2')(x)
    
    # Block 3
    x = layers.Dense(32, kernel_regularizer=regularizers.l2(l2_reg), use_bias=False, name='dense_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.Activation('relu', name='act_3')(x)
    x = layers.Dropout(dropouts[2], name='drop_3')(x)
    
    # Output
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='Regularized_Deep_MLP')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_residual_mlp(input_dim=30, learning_rate=0.001, hidden_dim=64):
    """
    Model 4: Deep Residual MLP (Res-MLP)
    Utilizes skip connections (residual addition) and Layer Normalization with GELU activation
    to facilitate smooth gradient flow and feature preservation.
    """
    inputs = layers.Input(shape=(input_dim,), name='input')
    
    # Initial projection
    x = layers.Dense(hidden_dim, activation='gelu', name='proj')(inputs)
    x = layers.LayerNormalization(name='ln_proj')(x)
    
    # Residual Block 1
    res1 = layers.Dense(hidden_dim, activation='gelu', name='res1_dense1')(x)
    res1 = layers.Dropout(0.2, name='res1_drop')(res1)
    res1 = layers.Dense(hidden_dim, name='res1_dense2')(res1)
    x = layers.Add(name='res1_add')([x, res1])
    x = layers.LayerNormalization(name='res1_ln')(x)
    
    # Residual Block 2
    res2 = layers.Dense(hidden_dim, activation='gelu', name='res2_dense1')(x)
    res2 = layers.Dropout(0.2, name='res2_drop')(res2)
    res2 = layers.Dense(hidden_dim, name='res2_dense2')(res2)
    x = layers.Add(name='res2_add')([x, res2])
    x = layers.LayerNormalization(name='res2_ln')(x)
    
    # Classification Head
    head = layers.Dense(32, activation='gelu', name='head_dense')(x)
    head = layers.Dropout(0.1, name='head_drop')(head)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(head)
    
    model = Model(inputs=inputs, outputs=outputs, name='Residual_Deep_MLP')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def get_model_by_name(name, input_dim=30, learning_rate=0.001, optimizer_name='adam'):
    """Factory helper to build and re-compile models with specified optimizers."""
    if name == 'Baseline_MLP':
        model = build_baseline_mlp(input_dim, learning_rate)
    elif name == 'Deep_MLP':
        model = build_deep_mlp(input_dim, learning_rate)
    elif name == 'Regularized_MLP':
        model = build_regularized_mlp(input_dim, learning_rate)
    elif name == 'Residual_MLP':
        model = build_residual_mlp(input_dim, learning_rate)
    else:
        raise ValueError(f"Unknown model name: {name}")
    
    # Compile with chosen optimizer
    if optimizer_name.lower() == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    else:
        opt = optimizer_name
        
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

