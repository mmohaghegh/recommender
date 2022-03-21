#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from helper import *

class data():
    """
    Class for preprocessing of the data.
    
    attributes:
        X_test(train)(val): numpy.ndarray
            feature space of test (train)(validation) sets after preprocessing
        y_test(train)(val): numpy.ndarray
            target (label) of test (train)(validation) sets after preprocessing
        show_id: numpy.ndarray
            all possible labels of the dataset
        
    methods:
        transform_features:
            preprocess features
        transform_posted_data:
            apply the same preprocessing on the unlabeled data (streamed)
        concat_features:
            collect all feature after being preprocessed
        transform_target:
            preprocess the target (label)
        split_train_val_test:
            split the data to train, validation and test
            
    """
    
    def __init__(self, param_file):
        self._orig_data = pd.read_parquet("adition_forecaster_application_data"+\
                                          ".snappy.parquet", engine="fastparquet")
        self._features = []
        self._get_targetfeatures()
        
    def _get_targetfeatures(self):
        """
        extract target and features from the dataset
        """
        self._orig_target = self._orig_data.pop("show_id")
        self._orig_features = self._orig_data
        
    def transform_features(self, feat):
        """
        preprocessing feature 'feat'
        
        Parameter
        ---------
        feat: str
        feature name
        """
        if "_age" in feat:
            age = self._orig_features.user_age.values
            self._age = {"min": age.min(), "max": age.max()}
            self._features.append(self._transform_age(age))
        elif "int" in feat:
            interests = self._orig_features.user_interests.values
            hobby, kind = get_uniques_from_dict(interests)
            self._interests = {"hobby": hobby, "kind": kind}
            hobby, kind = self._transform_interests(interests)
            self._features.append(hobby)
            self._features.append(kind)
        elif "lang" in feat:
            languages = self._orig_features.user_languages.values
            self._languages = np.unique(np.concatenate(languages))
            self._features.append(self._transform_languages(languages))
        else:
            raise NameError("feature '{}' does not exist!".format(feat))
            
    def _transform_age(self, age):
        """
        transform the feature 'user_age'
        
        Parameter
        ---------
        age: numpy.ndarray
        """
        # age = self.orig_features.user_age.values
        # self.age = {"min": age.min(),
        #             "max": age.max()}
        age = (age - self._age["min"])/(self._age["max"] - self._age["min"])
        return age.reshape(-1, 1)
        # self.features.append(age.reshape(-1, 1))
        
    def _transform_interests(self, interests):
        """
        transform the feature 'user_interests'
        
        Parameter
        ---------
        interests: numpy.ndarray
        """
        # interests = self.orig_features.user_interests.values
        # hobby_uniq, kind_uniq = get_uniques_from_dict(interests)
        hobby, kind = [], []
        for _dict in interests:
            hobby.append(to_indicator(list(_dict.keys()), self._interests["hobby"]))
            kind.append(to_indicator(list(_dict.values()), self._interests["kind"]))
        return np.array(hobby), np.array(kind)
        # self.features.append(hobby)
        # self.features.append(kind)
        # self.interests = {"hobby": hobby_uniq, "kind": kind_uniq}
        
    def transform_posted_data(self, inp):
        """
        transform the posted feature data
        
        Parameter
        ---------
        inp: dict
        received in json format
        
        Return
        ------
        feature_vec: numpy.ndarray
        feature vector after preprocessing
        """
        feature_vec = []
        age_range = np.arange(inp["age_range"]["range_start"],
                              inp["age_range"]["range_end"]+1)
        age = self._transform_age(age_range)
        hobby, kind = self._transform_interests([inp["user_interests"]]*age_range.size)
        languages = self._transform_languages([inp["user_languages"]]*age_range.size)
        for feat in (age, hobby, kind, languages):
            feature_vec.append(feat)
        return np.hstack(feature_vec)
        
        
    def _transform_languages(self, languages):
        """
        transform the feature 'user_languages'
        
        Parameter
        ---------
        languages: numpy.ndarray
        """ 
        # languages = self.orig_features.user_languages.values
        # lang_uniq = np.unique(np.concatenate(languages))
        lang = []
        for _list in languages:
            lang.append(to_indicator(_list, self._languages))
        return np.array(lang)
        # self.features.append(lang)
        # self.languages = lang_uniq
        
    def concat_features(self):
        """
        collect all features
        """
        self._features = np.hstack(self._features)
        
    def transform_target(self):
        """
        transform target vector 'show_id'
        
        Parameter
        ---------
        languages: numpy.ndarray
        """ 
        showids_uniq = self._orig_target.unique()
        self._target = []
        for sh_id in self._orig_target.values:
            self._target.append(np.where(showids_uniq==sh_id)[0][0])
        self._target = np.array(self._target)
        self.show_id = showids_uniq
        
    def split_train_val_test(self, test_prop=0.2, val_prop=0.2):
        """
        split the entire data set to training, validation and testing sets
        
        Parameters:
            test_prop: float
            the percentage of the entire data to leave out for testing
            val_prop: float
            the percentage of the data that is left after leaving out for test
            for validation
            
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self._features,
                                                                                self._target,
                                                                                test_size=test_prop)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self._features,
                                                                              self._target,
                                                                              test_size=val_prop)
        
class model():
    
    def __init__(self, data):
        self.model = None
        self.data = data
    
    def setup(self, num_hid_lay=1, num_units=100, act_func="relu"):
        hidden_layers = []
        input_layer = keras.Input(shape=(self.data.X_train.shape[1],))
        num_units = expand_to_list(num_units, num_hid_lay)
        act_func = expand_to_list(act_func, num_hid_lay)
        for h_l in range(num_hid_lay):
            hidden_layers.append(layers.Dense(num_units[h_l],
                                              activation=act_func[h_l]))
        output_layer = layers.Dense(self.data.show_id.size, activation="softmax")
        self.model = tf.keras.Sequential([input_layer] + \
                                         hidden_layers + \
                                         [output_layer])
        self.model.compile(optimizer="adam",
                           loss=tf.keras.losses.sparse_categorical_crossentropy,
                           metrics=["accuracy"])
    
    def fit(self):
        self.model.fit(self.data.X_train, self.data.y_train,
                       validation_data=(self.data.X_val, self.data.y_val),
                       epochs=10, batch_size=100)
    
    def predict(self, X, y):
        return self.model.predict(X, y)