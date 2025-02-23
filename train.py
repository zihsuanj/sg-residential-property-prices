from src.model import Model

model = Model()
print("Preprocessing data...")
model.preprocess_data()
print("Training model...")
model.fit()
print("Evaluating model...")
model.evaluate(save_model_metadata=True)
print("Model training completed.")