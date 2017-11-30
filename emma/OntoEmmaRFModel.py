import sys
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer

# class for random forest ontology matcher model
class OntoEmmaRFModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_jobs=1, random_state=0)
        self.vectorizer = DictVectorizer(sparse=False)

    def save(self, model_path):
        """
        Save model to file
        :return:
        """
        sys.stdout.write("Saving Model...\n")
        pickle.dump([self.model, self.vectorizer], open(model_path, 'wb'))
        return

    def load(self, model_path):
        """
        Load model from file
        :return:
        """
        sys.stdout.write("\tLoading %s...\n" % model_path)
        try:
            [self.model, self.vectorizer] = pickle.load(open(model_path, 'rb'))
        except IndexError:
            sys.stderr.out("Model format doesn't match expectations; you are probably trying to load an old model.")
            sys.exit(1)
        return

    def train(self, f_dicts, labels):
        """
        Calculate features and train model
        :param labels: labels for each entity pair
        :param f_dicts: feature vectors for each entity pair
        :return:
        """
        f_vectors = self.vectorizer.fit_transform(f_dicts)
        self.model.fit(f_vectors, labels)
        return

    def score_accuracy(self, f_dicts, labels):
        """
        Calculate the accuracy score on input labels and features
        :param labels:
        :param f_dicts:
        :return:
        """
        f_vectors = self.vectorizer.transform(f_dicts)
        accuracy = self.model.score(f_vectors, labels)
        return accuracy

    def predict_entity_pair(self, f_dict):
        """
        Make prediction for alignment given a feature vector
        :param features: feature vector
        :return:
        """
        f_vector = self.vectorizer.transform(f_dict)
        return self.model.predict_proba(f_vector)
