import re
import pandas as pd

class DataPreprocessor:
    def __init__(self, df:pd.DataFrame):
        self.df = df
    
    def __calc_remaining_lease(self, remaining_lease: None | int | str, sale_period:str, lease_commence_date:int) -> int:
        """
        Description:
            This method calculates the remaining lease of the property in number of years at the time of sale 
            if it was not provided in the training data, based on the assumption that the initial lease length is 99 years. 
            If the remaining lease was provided, the number of years is converted into an integer or extracted from the string.
        """
        # If remaining lease was not provided, calculate remaining lease at time of sale assuming 99 year initial lease length
        if str(remaining_lease) == "nan":
            sale_year = int(sale_period.split("-")[0])
            remaining_lease = abs(99 - (sale_year - lease_commence_date))

        # If remaining lease was provided, check if it can be converted into an integer
        else:
            try:
                remaining_lease = int(remaining_lease)

            # If remaining lease cannot be converted into a integer, extract the number of years from the string
            except:
                regex = re.compile(r"^\d{2} years")
                if regex.match(remaining_lease):
                    remaining_lease = remaining_lease[0:2]
        return int(remaining_lease)

    def __calc_storey(self, storey_range:str) -> float:
        """
        Description:
            This method calculates the approximate storey of the property by averaging the minimum and maximum
            stories provided. E.g., 05 TO 07 is converted to 6 by averaging 5 and 7.
        """
        list_storey_range = [int(i) for i in storey_range.split(" TO ")]
        return sum(list_storey_range) / len(list_storey_range)

    def preprocess_data(self, calc_remaining_lease:bool=True, calc_story:bool=True, extract_year:bool=True, return_df:bool=False) -> pd.DataFrame | None:
        """
        Description:
            This method preprocesses the dataset.
        """
        # Calculate remaining lease
        if calc_remaining_lease:
            self.df.loc[:, "remaining_lease"] = [self.__calc_remaining_lease(i, j, k) for i, j, k in zip(self.df["remaining_lease"], self.df["month"], self.df["lease_commence_date"])]
            self.df["remaining_lease"] = self.df["remaining_lease"].astype(int)
        
        # Convert stories to the average of the min and max
        if calc_story:
            self.df.loc[:, "storey"] = [self.__calc_storey(i) for i in self.df["storey_range"]]

        # Extract the year from the 'month' column
        if extract_year:
            self.df.loc[:, "year"] = [int(i[0:4]) for i in self.df["month"]]

        # Ensure that all values in the 'flat_model' column are in uppercase
        self.df.loc[:, "flat_model"] = [i.upper() for i in self.df["flat_model"]]

        # Standardise the spelling of 'MULTI-GENERATION' flat types
        self.df.loc[self.df["flat_type"] == "MULTI GENERATION", "flat_type"] = "MULTI-GENERATION"

        # Order the dataset by the 'month' column for splitting
        self.df = self.df.sort_values(by="month", ascending=True)
        if return_df:
            return self.df

    def train_test_split(self, train_cols:list, target_col:str, split_type:str="train_ratio", split_value:float | int=0.9) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list]:
        """
        Description:
            This method does the following:
                1) Filters the dataset to contain only the columns required for training.
                2) Splits the dataset into training and test datasets.
                    a) Splitting can be done in two ways:
                        i)  By ratio, e.g., the training dataset should contain the first 80% of the data, 
                            with the test dataset containing the remaining 20%, or
                        ii) By maximum training year offset, e.g., if a value of -2 is specified,
                            the training dataset will span the from the earliest year to the maximum year - 2.
                            Therefore, if the minimum and maximum years are 1990 and 2020 respectively,
                            the training dataset will span from 1990 to 2018, with the test dataset spanning
                            from 2019 to 2020.
                3) For the training and test datasets, further split into X and y datasets.
                4) Identify and return the categorical features in the X_train dataset.               
        """
        # Filter the dataset
        df = self.df[train_cols + [target_col]]

        # Split the dataset into training and test datasets
        if split_type == "train_ratio":
            if split_value > 1 or split_value < 0:
                raise ValueError("train_ratio must be between 0 and 1.")       
            cutoff = int(split_value*self.df.shape[0])
            df_train, df_test = df.iloc[:cutoff].copy() , df.iloc[cutoff:].copy()
        elif split_type == "max_train_year_offset" and "year" in df.columns:
            max_train_year = df["year"].max() + split_value
            df_train, df_test = df.loc[df["year"] <= max_train_year], df.loc[df["year"] > max_train_year]
        else:
            raise ValueError("Invalid split_type specified: Only values accepted are 'train_ratio' or 'max_train_year'.")

        # Further split training and test datasets into X and y datasets
        X_train = df_train[train_cols]
        y_train = df_train[target_col]
        X_test = df_test[train_cols]
        y_test = df_test[target_col]

        # Identify and return categorical features in X_train
        X_cat_cols = [i for i, j in X_train.dtypes.items() if j.name == "object"]

        return (X_train, y_train, X_test, y_test, X_cat_cols)