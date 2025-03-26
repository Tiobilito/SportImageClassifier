# predict.py
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def predict_image(model_path, image_path, target_size=(128, 128)):
    model = load_model(model_path)
    # Cargar y preprocesar la imagen
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    result = model.predict(image)
    predicted_class_index = np.argmax(result)
    return predicted_class_index, result

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Predicción de una imagen con un modelo entrenado.')
    parser.add_argument('--model_path', type=str, required=True, help='Ruta del modelo guardado (ej. models/cnn_model.h5 o models/transfer_model.h5)')
    parser.add_argument('--image', type=str, required=True, help='Ruta de la imagen a predecir (por ejemplo, single_test/imagen.jpg)')
    parser.add_argument('--dataset', type=str, default='dataset', help='Ruta del dataset para obtener el mapeo de clases')
    args = parser.parse_args()

    # Obtener el mapeo de clases desde el dataset
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    test_datagen = ImageDataGenerator(rescale=1./255)
    data_gen = test_datagen.flow_from_directory(
        args.dataset,
        target_size=(128, 128),
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    class_labels = data_gen.class_indices
    class_labels = {v: k for k, v in class_labels.items()}

    predicted_class_index, probabilities = predict_image(args.model_path, args.image)
    predicted_class_label = class_labels[predicted_class_index]
    print("Probabilidades predichas:", probabilities)
    print("La imagen pertenece a la clase:", predicted_class_label)
