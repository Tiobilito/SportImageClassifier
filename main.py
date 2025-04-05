import argparse
import os
from data_loader import get_data_generators
from models import build_cnn_model, build_transfer_model
from keras._tf_keras.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

def train_model(model, train_gen, valid_gen, epochs, model_save_path):
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    history = model.fit(train_gen, validation_data=valid_gen, epochs=epochs, callbacks=[checkpoint, earlystop])
    return history

def plot_history(history, output_dir):
    # Graficar evolución de precisión
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Evolución de Precisión')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'accuracy.png'))

    # Graficar evolución de pérdida
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Evolución de Pérdida')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    plt.close('all')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenamiento de modelo CNN o Transfer Learning para imágenes deportivas.')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'transfer'], help='Tipo de modelo a entrenar: cnn o transfer')
    parser.add_argument('--dataset', type=str, default='dataset', help='Ruta a la carpeta del dataset')
    parser.add_argument('--epochs', type=int, default=20, help='Número de épocas de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamaño del batch')
    parser.add_argument('--output', type=str, default='models', help='Directorio para guardar el modelo y gráficos')
    # Puedes agregar más argumentos para registrar hiperparámetros
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()

    # Crear directorio de salida si no existe
    os.makedirs(args.output, exist_ok=True)

    target_size = (128, 128)
    input_shape = (128, 128, 3)

    train_gen, valid_gen = get_data_generators(args.dataset, target_size=target_size, batch_size=args.batch_size)
    num_classes = len(train_gen.class_indices)
    print("Clases encontradas:", train_gen.class_indices)

    # Seleccionar y construir el modelo según el argumento
    if args.model == 'cnn':
        print("Entrenando modelo CNN desde cero")
        model = build_cnn_model(input_shape, num_classes, learning_rate=args.learning_rate)
        model_save_path = os.path.join(args.output, 'cnn_model.keras')
    else:
        print("Entrenando modelo con Transfer Learning (VGG16)")
        model = build_transfer_model(input_shape, num_classes, learning_rate=args.learning_rate)
        model_save_path = os.path.join(args.output, 'transfer_model.keras')

    # Entrenar el modelo y obtener la historia
    history = train_model(model, train_gen, valid_gen, args.epochs, model_save_path)
    plot_history(history, args.output)

    # Imprimir hiperparámetros y resultados finales (Overall Accuracy)
    final_accuracy = history.history['val_accuracy'][-1]
    print("\n----- Resumen de Hiperparámetros y Resultados -----")
    print("Modelo:", args.model)
    print("Learning rate:", args.learning_rate)
    print("Épocas:", args.epochs)
    print("Tamaño de batch:", args.batch_size)
    print("Número de capas convolucionales: 3")  # Si usas la arquitectura definida
    print("Overall Validation Accuracy:", final_accuracy)
