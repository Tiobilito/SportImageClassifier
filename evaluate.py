import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.models import load_model

def evaluate_model(model, dataset_dir, target_size=(128, 128), batch_size=32):
    # Generador para evaluación sin aumento de datos
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Realizar predicciones
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    # Matriz de confusión y reporte de clasificación
    cm = confusion_matrix(y_true, y_pred)
    print("Matriz de Confusión:")
    print(cm)
    print("\nReporte de Clasificación:")
    print(classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys())))

    # Graficar la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(test_generator.class_indices.keys()), 
                yticklabels=list(test_generator.class_indices.keys()))
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.title('Matriz de Confusión')
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluación de un modelo entrenado.')
    parser.add_argument('--model_path', type=str, required=True, help='Ruta del modelo guardado (.h5)')
    parser.add_argument('--dataset', type=str, default='dataset', help='Ruta al dataset para evaluación')
    args = parser.parse_args()

    model = load_model(args.model_path)
    evaluate_model(model, args.dataset)
