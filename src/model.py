import os
import json
import joblib
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error
from .load_data import load_data
from .data_preprocessor import DataPreprocessor

class Model:
    def __init__(self, EDA:bool=False):
        self.EDA = EDA
        self.pp_data = None
        self.model = None
        self.best_params = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_cat_cols = None

        with open(os.path.join("config", "train_config.json"), "rb") as f:
            train_config = json.load(f)
            self.OUTPUT_FOLDER = train_config["OUTPUT_FOLDER"]
            self.TRAIN_COLS = train_config["TRAIN_COLS"]
            self.TARGET_COL = train_config["TARGET_COL"]
            self.TRAIN_TEST_SPLIT = train_config["TRAIN_TEST_SPLIT"]
            self.HYPERPARAMETER_TUNING = train_config["HYPERPARAMETER_TUNING"]
            self.PARAM_GRID = train_config["PARAM_GRID"]
            self.OBJECTIVE_FUNCTION = train_config["OBJECTIVE_FUNCTION"]
    
    def preprocess_data(self) -> None:
        """
        Description:
            This method first loads the data and performs data preprocessing using the 
            preprocess_data method of the DataPreprocessor class.
        """
        df = load_data()
        self.dp = DataPreprocessor(df)
        self.pp_data = self.dp.preprocess_data(return_df=self.EDA)

    def split_data(self) -> None:
        """
        Description:
            This method splits the dataset using the train_test_split method of the DataPreprocessor class.
        """
        self.X_train, self.y_train, self.X_test, self.y_test, self.X_cat_cols = self.dp.train_test_split(train_cols=self.TRAIN_COLS, 
                                                                                                         target_col=self.TARGET_COL, 
                                                                                                         split_type=self.TRAIN_TEST_SPLIT["SPLIT_TYPE"], 
                                                                                                         split_value=self.TRAIN_TEST_SPLIT["SPLIT_VALUE"])
        
    def fit(self, model_filename:str="model.pkl") -> None:
        """
        Description:
            This method does the following:
                1) Calls the split_data method to split the dataset.
                2) Trains a CatBoost regressor based on the parameters specified in config/train_config.json
                3) Saves the trained model to the specified output folder, i.e. OUTPUT_FOLDER

        """
        # Split the dataset
        self.split_data()

        # Train the model based on specified parameters
        self.model = CatBoostRegressor(cat_features=self.X_cat_cols, 
                                       verbose=False, 
                                       loss_function=self.OBJECTIVE_FUNCTION, 
                                       eval_metric=self.OBJECTIVE_FUNCTION, 
                                       random_state=32)
        
        if self.HYPERPARAMETER_TUNING:
            self.model.best_params_, self.model.cv_results_ = self.model.grid_search(param_grid=self.PARAM_GRID, X=self.X_train, y=self.y_train, cv=3)
        else:
            self.model.fit(self.X_train, self.y_train)
        
        # Save training and test dataset characteristics to the model object
        self.model.num_train_rows = self.X_train.shape[0]
        self.model.train_max_year = self.X_train.year.max()
        self.model.num_test_rows = self.X_test.shape[0]
        self.model.test_max_year = self.X_test.year.max()

        # Save the model object to the specified output folder
        with open(os.path.join(self.OUTPUT_FOLDER, model_filename), "wb") as f:
            joblib.dump(self.model, f)

    def evaluate(self, model_filename:str=None, save_model_metadata:bool=False) -> None:
        """
        Description:
            This method does the following:
                1) Evaluates the trained model by MAPE on two datasets:
                    a) The entire training dataset, i.e. TRAIN_MAPE.
                    b) The test dataset, i.e. TEST_MAPE.
                2) If the method is called as part of the model training process:
                    a) Extracts feature importances from the trained model.
                    b) Extracts the training and test dataset characteristics from the model object.
                    c) Saves all the data above into the model_metadata.json file in the specified output folder.
        """
        # Checks if the model is loaded.
        if self.model is None:
            # If model is not loaded, attempt to load model from specified model_filename in the output folder.
            if model_filename is None:
                raise AttributeError("Train a new model or specify the model filename in the output folder.")
            else:
                self.model = joblib.load(os.path.join(self.OUTPUT_FOLDER, model_filename))

        # Load and split the datasets if not previously loaded.
        if self.X_train is None or self.y_train is None or self.X_test is None or self.y_test is None:
            self.preprocess_data()
            self.split_data()

        if save_model_metadata:
            # Ascertain train and test MAPE.
            y_pred_train = self.model.predict(self.X_train)
            train_mape = mean_absolute_percentage_error(self.y_train, y_pred_train)
            y_pred_test = self.model.predict(self.X_test)
            test_mape = mean_absolute_percentage_error(self.y_test, y_pred_test)
            self.eval_metrics = {"TRAIN_MAPE": train_mape, "TEST_MAPE": test_mape}

            # Extract model feature importances.
            self.feature_importances = dict(sorted({i:j for i, j in zip(self.model.feature_names_, self.model.feature_importances_)}.items(), key=lambda x: x[1], reverse=True))

            # Extract training and test dataset characteristics.
            model_metadata = {"DATA": {
                                "NUM_TRAIN_ROWS": int(self.model.num_train_rows),
                                "NUM_TEST_ROWS":  int(self.model.num_test_rows),
                                "TRAIN_MAX_YEAR": int(self.model.train_max_year),
                                "TEST_MAX_YEAR":  int(self.model.test_max_year)
                            },
                            "EVAL_METRICS": self.eval_metrics, 
                            "FEATURE_IMPORTANCES": self.feature_importances}
            
            # Save all data to model_metadata.json in the output folder.
            with open(os.path.join(self.OUTPUT_FOLDER, "model_metadata.json"), "w") as f:
                json.dump(model_metadata, f, indent=4)