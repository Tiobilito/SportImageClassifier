import os
from predict import predict_image
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.models import load_model

def load_class_labels(dataset_path, target_size=(128, 128)):
    test_datagen = ImageDataGenerator(rescale=1./255)
    data_gen = test_datagen.flow_from_directory(
        dataset_path,
        target_size=target_size,
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    return {v: k for k, v in data_gen.class_indices.items()}

def batch_predict(model_path, test_folder, dataset_path, target_size=(128, 128)):
    model = load_model(model_path)
    class_labels = load_class_labels(dataset_path, target_size)
    results = {}
    for filename in os.listdir(test_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_folder, filename)
            idx, probabilities = predict_image(model_path, image_path, target_size)
            results[filename] = {
                'predicted_index': idx,
                'predicted_class': class_labels[idx],
                'probabilities': probabilities.tolist()
            }
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Predicción en lote para imágenes en la carpeta single_test.')
    parser.add_argument('--model_path', type=str, required=True, help='Ruta del modelo (.h5)')
    parser.add_argument('--test_folder', type=str, default='single_test', help='Carpeta con imágenes de test')
    parser.add_argument('--dataset', type=str, default='dataset', help='Carpeta del dataset para obtener el mapeo de clases')
    parser.add_argument('--target_size', type=int, default=128, help='Tamaño al que se redimensionarán las imágenes (ejemplo: 128 para 128x128)')
    args = parser.parse_args()

    target_size = (args.target_size, args.target_size)
    results = batch_predict(args.model_path, args.test_folder, args.dataset, target_size)
    for file, data in results.items():
        print(f"Imagen: {file}")
        print(f"  Clase predicha: {data['predicted_class']} (índice: {data['predicted_index']})")
        print(f"  Probabilidades: {data['probabilities']}\n")
