import tensorflow as tf

# Load the EfficientNet-V2 model
model = tf.keras.applications.efficientnet_v2.EfficientNetV2L(weights='imagenet',include_top=False,classes=2)
x = tf.keras.layers.GlobalAveragePooling2D()(model.output) 
outputs = tf.keras.layers.Dense(2, name="output_layer")(x)
model_0 = tf.keras.models.Model(inputs=model.input, outputs=outputs)

# Load the training and validation datasets
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    './images/train/',
    batch_size=2,
    image_size=(224, 224)
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    './images/valid/',
    batch_size=2,
    image_size=(224, 224)
)

# Compile the model
model_0.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model
history = model_0.fit(
    train_dataset,
    epochs=10,
    validation_data=validation_dataset
)