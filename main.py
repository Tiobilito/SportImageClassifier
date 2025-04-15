import argparse
import os
import csv
from data_loader import get_data_generators
from models import build_cnn_model, build_transfer_model
from keras._tf_keras.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

def train_model(model, train_gen, valid_gen, epochs, model_save_path):
    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    earlystop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1
    )
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=epochs,
        callbacks=[checkpoint, earlystop]
    )
    return history

def plot_history_prefixed(history, graphs_dir, model_name):
    # Precisión
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Evolución de Precisión')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(graphs_dir, f'{model_name}_accuracy.png'))

    # Pérdida
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Evolución de Pérdida')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(graphs_dir, f'{model_name}_loss.png'))
    plt.close('all')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Entrenamiento de modelo CNN o Transfer Learning'
    )
    parser.add_argument('--model', choices=['cnn', 'transfer'], default='cnn',
                        help='Tipo de modelo a entrenar: cnn o transfer')
    parser.add_argument('--dataset', default='dataset',
                        help='Ruta a la carpeta del dataset')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Número de épocas de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Tamaño del batch')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate para el optimizador')
    parser.add_argument('--output', default='models',
                        help='Directorio para guardar el modelo entrenado')
    parser.add_argument('--results_dir', default='results',
                        help='Directorio raíz para volcar resultados')
    args = parser.parse_args()

    model_name = args.model

    # Crear directorios
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    graphs_dir = os.path.join(args.results_dir, 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)

    # Preparar datos
    target_size = (128, 128)
    input_shape = (128, 128, 3)
    train_gen, valid_gen = get_data_generators(
        args.dataset,
        target_size=target_size,
        batch_size=args.batch_size
    )
    num_classes = len(train_gen.class_indices)
    print("Clases encontradas:", train_gen.class_indices)

    # Construir modelo
    if model_name == 'cnn':
        print("Construyendo y entrenando CNN desde cero...")
        model = build_cnn_model(input_shape, num_classes, learning_rate=args.learning_rate)
    else:
        print("Construyendo y entrenando modelo con Transfer Learning (VGG16)...")
        model = build_transfer_model(input_shape, num_classes, learning_rate=args.learning_rate)

    model_path = os.path.join(args.output, f'{model_name}_model.keras')

    # Entrenar y guardar gráficos
    history = train_model(model, train_gen, valid_gen, args.epochs, model_path)
    plot_history_prefixed(history, graphs_dir, model_name)

    # Exportar hiperparámetros y accuracy final
    final_acc = history.history['val_accuracy'][-1]
    csv_path = os.path.join(args.results_dir, f'{model_name}_hyperparameters.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'model',
            'learning_rate',
            'epochs',
            'batch_size',
            'conv_layers',
            'val_accuracy'
        ])
        writer.writerow([
            model_name,
            args.learning_rate,
            args.epochs,
            args.batch_size,
            3,
            final_acc
        ])

    print(f"\nResultados guardados en '{args.results_dir}/'")
