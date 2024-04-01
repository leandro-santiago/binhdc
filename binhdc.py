from bitarray import bitarray
from bitarray.util import urandom, count_xor
import numpy as np

class BinHDC:
    def __init__(self, num_classes, dimension, labels = None) -> None:
        self.hv_classes = []
        self.binhv_classes = []
        self.labels = {}
        self.num_classes = num_classes
        self.dimension = dimension

        # Defining fixed Hyper Vector containing -1 values 
        zero_array = np.zeros((dimension,), dtype=int)   
        one_array = np.ones((dimension,), dtype=int)
        self.minus_array = zero_array - one_array

        # Initializing Hyper Vector Classes
        for i in range(num_classes):
            self.hv_classes.append(np.zeros((dimension,), dtype=int)) 
        
        if not labels is None:
            self.set_labels(labels)

    def set_labels(self, labels):
        id = 0
        for label in labels:
            self.labels[label] = id
            id += 1     

    def fit(self, data, labels):
        for entry, label in zip(data, labels):
            hv_one_minus = entry.tolist() + self.minus_array + entry.tolist()
            label_id = self.labels[label]
            self.hv_classes[label_id] += hv_one_minus 
        
        #print(self.hv_classes)
        for label_id in range(self.num_classes):
            self.binhv_classes.append(bitarray(self.dimension))
            for i in range(self.dimension):
                self.binhv_classes[label_id][i] = 1 if self.hv_classes[label_id][i] > 0 else 0 
            
    #print(self.binhv_classes)      
    def predict(self, data):
        result = []
        for entry in data:
            min_hd = self.dimension
            best_class = self.num_classes
            class_id = 0  
            
            for class_hv in self.binhv_classes:
                hamming_distance = count_xor(entry, class_hv)
                if hamming_distance < min_hd:
                    min_hd = hamming_distance
                    best_class = class_id
                class_id += 1
            
            result.append(best_class) 
        
        return result

    def accuracy(self, predicted_labels, test_labels):
        total = len(predicted_labels)
        hits = 0

        for label1, label2 in zip(predicted_labels, test_labels):
            label2_id = self.labels[label2]
            if (label1 == label2_id):
                hits += 1
        
        acc = hits/total 

        return acc