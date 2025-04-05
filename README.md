# SportImageClassifier

## Entrenar el modelo CNN desde cero:
```
python main.py --model cnn --dataset dataset --epochs 20 --batch_size 32 --learning_rate 0.001 --output models
```

## Para entrenar con Transfer Learning:
```
python main.py --model transfer --dataset dataset --epochs 20 --batch_size 32 --learning_rate 0.001 --output models
```

## Evaluación del Modelo:
Una vez entrenado el modelo, evalúalo con:
```
python evaluate.py --model_path models/cnn_model.keras --dataset dataset
```

## Predicción de una Imagen Individual:
Para probar una imagen específica:
```
python predict.py --model_path models/cnn_model.keras --image single_test/imagen.jpg --dataset dataset
```

## Predicción en Lote (batch_predict):
Para recorrer y predecir todas las imágenes en la carpeta `single_test`:
```
python batch_predict.py --model_path models/cnn_model.keras --test_folder single_test --dataset dataset --target_size 128
```