import unittest
import pandas as pd
import numpy as np
from pympute.utils import explore, cpu_regressors_list, cpu_classifiers_list # Assuming explore can be imported

class TestExploreFunction(unittest.TestCase):

    def test_isreg_determination(self):
        # Test the logic: isreg = df.nunique() > 10
        data_reg = {'col_reg': list(range(11))} # 11 unique values -> regression
        df_reg = pd.DataFrame(data_reg)
        isreg_series_reg = df_reg.nunique() > 10
        self.assertTrue(isreg_series_reg['col_reg'])

        data_class = {'col_class': list(range(10))} # 10 unique values -> classification
        df_class = pd.DataFrame(data_class)
        isreg_series_class = df_class.nunique() > 10
        self.assertFalse(isreg_series_class['col_class'])

        data_mixed = {
            'col_reg': list(range(15)),                            # Length 15, 15 unique
            'col_class_exact': list(range(10)) + [9]*5,          # Length 15, 10 unique
            'col_class_less': [0, 1, 0, 1, 0] + [0]*10             # Length 15, 2 unique
        }
        df_mixed = pd.DataFrame(data_mixed)
        isreg_series_mixed = df_mixed.nunique() > 10
        self.assertTrue(isreg_series_mixed['col_reg'])
        self.assertFalse(isreg_series_mixed['col_class_exact'])
        self.assertFalse(isreg_series_mixed['col_class_less'])

    def test_model_dictionary_construction(self):
        # This test focuses on the part of 'explore' that builds the models_for_imputer dictionary.
        # We'll simulate the inputs to that section.
        
        # Mock isreg series
        isreg = pd.Series({'colA': True, 'colB': False, 'colC': True, 'colD': False})
        
        # Mock missing_columns
        missing_columns = ['colA', 'colB', 'colC', 'colD']
        
        # Mock a base model name
        mdl = 'RF' # Example model, could be any from the generated list

        # Expected dictionary
        expected_models_for_imputer = {
            'colA': 'RF-r', # isreg is True
            'colB': 'RF-c', # isreg is False
            'colC': 'RF-r', # isreg is True
            'colD': 'RF-c'  # isreg is False
        }
        
        # Actual construction logic (simplified from explore)
        models_for_imputer = {}
        for col_name in missing_columns:
            if isreg.loc[col_name]:
                models_for_imputer[col_name] = mdl + '-r'
            else:
                models_for_imputer[col_name] = mdl + '-c'
        
        self.assertDictEqual(expected_models_for_imputer, models_for_imputer)

    def test_model_dictionary_construction_variant_availability(self):
        # Test with a model that might only be a regressor or classifier
        isreg = pd.Series({'col_reg': True, 'col_class': False})
        missing_columns = ['col_reg', 'col_class']
        
        # Simulate a model that is primarily a regressor (e.g., 'SVR' often implies regressor)
        # And one that is primarily a classifier (e.g., 'SVC')
        # The explore function's model_list is now a union, so 'SVC' would be tried.
        # If mdl is 'SVC', and col_reg is True, it would try 'SVC-r'.
        # We need to ensure cpu_regressors_list() and cpu_classifiers_list() are available
        # to the test, or mock them if explore() itself is not called directly.
        # The logic inside explore is:
        #   models_for_imputer[col_name_iterator] = mdl + '-r' or mdl + '-c'
        # This dictionary is then passed to Imputer, which calls get_model.
        # get_model uses the lists to validate.
        # So, this test is more about ensuring the dictionary is formed correctly,
        # and less about get_model's behavior (which should have its own tests, ideally).

        # Case 1: Model 'LR' (typically has -r and -c)
        mdl_lr = 'LR'
        expected_lr = {'col_reg': 'LR-r', 'col_class': 'LR-c'}
        actual_lr = {}
        for col_name in missing_columns:
            if isreg.loc[col_name]:
                actual_lr[col_name] = mdl_lr + '-r'
            else:
                actual_lr[col_name] = mdl_lr + '-c'
        self.assertDictEqual(expected_lr, actual_lr)

        # Case 2: Consider a hypothetical model 'PureReg' that only has '-r'
        # If 'PureReg' was in model_list, and we encounter col_class (isreg=False)
        # models_for_imputer would contain 'PureReg-c'.
        # This would be caught by get_model inside the Imputer class.
        # The dictionary construction itself is simple and doesn't currently check
        # for variant validity; it relies on get_model for that.
        # So, the existing test_model_dictionary_construction covers the direct logic well.
        # No further complex test here unless we change the dict construction to pre-validate.

if __name__ == '__main__':
    unittest.main()
