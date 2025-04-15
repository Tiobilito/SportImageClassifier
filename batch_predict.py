import os
import csv
import json
from datetime import datetime
from predict import predict_image
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

def load_class_labels(dataset_path, target_size):
    datagen = ImageDataGenerator(rescale=1./255)
    gen = datagen.flow_from_directory(
        dataset_path,
        target_size=target_size,
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    return {v: k for k, v in gen.class_indices.items()}

def batch_predict(model_path, test_folder, dataset_path,
                  target_size, results_dir, model_name):
    # Cargar modelo y mapeo de clases
    model = load_model(model_path)
    class_labels = load_class_labels(dataset_path, target_size)

    # Crear carpetas de resultados
    os.makedirs(results_dir, exist_ok=True)
    pred_dir = os.path.join(results_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)

    results = []
    for fname in os.listdir(test_folder):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(test_folder, fname)
        idx, probs = predict_image(model_path, img_path, target_size)
        pred_class = class_labels[idx]
        confidence = probs[0][idx] * 100  # porcentaje

        # Cargar imagen y anotar
        img = Image.open(img_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        font_size = max(20, img.width // 20)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        text = f"{pred_class}: {confidence:.1f}%"
        text_w, text_h = draw.textsize(text, font=font)
        draw.rectangle([(0, 0), (img.width, text_h + 10)], fill=(0, 0, 0, 127))
        draw.text((5, 5), text, font=font, fill=(255, 255, 255))

        # Nombre único con modelo y timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{model_name}_{timestamp}_{fname}"
        out_path = os.path.join(pred_dir, out_name)
        img.save(out_path)

        # Registrar resultado
        results.append({
            'original_filename': fname,
            'output_image': out_name,
            'predicted_class': pred_class,
            'predicted_index': idx,
            'confidence': round(confidence, 2),
            'probabilities': probs.tolist()[0]
        })

    # Exportar CSV con prefijo del modelo
    csv_path = os.path.join(results_dir, f'{model_name}_predictions.csv')
    with open(csv_path, 'w', newline='') as f:
        fieldnames = [
            'original_filename',
            'output_image',
            'predicted_class',
            'predicted_index',
            'confidence',
            'probabilities'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            row['probabilities'] = json.dumps(row['probabilities'])
            writer.writerow(row)

    print(f"\nPredicciones guardadas en:\n"
          f" - Imágenes anotadas: {pred_dir}/\n"
          f" - CSV resumen: {csv_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Predicción en lote con exportación de resultados anotados'
    )
    parser.add_argument('--model_path', required=True,
                        help='Ruta al modelo (.keras)')
    parser.add_argument('--model_name', choices=['cnn', 'transfer'], required=True,
                        help='Prefijo que indica el modelo usado (cnn o transfer)')
    parser.add_argument('--test_folder', default='single_test',
                        help='Carpeta con imágenes de test')
    parser.add_argument('--dataset', default='dataset',
                        help='Carpeta del dataset para clases')
    parser.add_argument('--target_size', type=int, default=128,
                        help='Tamaño de redimensionado (e.g. 128 para 128×128)')
    parser.add_argument('--results_dir', default='results',
                        help='Directorio raíz de resultados')
    args = parser.parse_args()

    ts = (args.target_size, args.target_size)
    batch_predict(
        model_path=args.model_path,
        test_folder=args.test_folder,
        dataset_path=args.dataset,
        target_size=ts,
        results_dir=args.results_dir,
        model_name=args.model_name
    )
