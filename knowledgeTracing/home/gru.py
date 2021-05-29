import numpy as np
from csv import reader
import pandas as pd
import random
import time
import sys
import os
import datetime
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import tensorflow as tf
import seaborn as sns

class TrainConfig(object):
    epochs = 10
    decay_rate = 0.92
    learning_rate = 0.01
    evaluate_every = 100
    checkpoint_every = 100
    max_grad_norm = 3.0


class ModelConfig(object):
    hidden_layers = [200]
    dropout_keep_prob = 0.6


class Config(object):
    batch_size = 32
    num_skills = 124
    input_size = num_skills * 2

    trainConfig = TrainConfig()
    modelConfig = ModelConfig()
    
config = Config()

class DataGenerator(object):
    
    def __init__(self, fileName, config):
        self.fileName = fileName
        self.train_seqs = []
        self.test_seqs = []
        self.infer_seqs = []
        self.batch_size = config.batch_size
        self.pos = 0
        self.end = False
        self.num_skills = config.num_skills
        self.skills_to_int = {}  
        self.int_to_skills = {} 

    def read_file(self):
       
        seqs_by_student = {}
        skills = [] 
        count = 0
        students =[]
        with open(self.fileName, 'r') as f:
            for line in f:
                fields = line.strip().split(" ")  
                student, skill, is_correct = int(fields[0]), int(fields[1]), int(fields[2])
                students.append(student)
                skills.append(skill)  
                seqs_by_student[student] = seqs_by_student.get(student, []) + [[skill, is_correct]] 
        # print(students)
        return seqs_by_student, list(set(skills))

    def gen_dict(self, unique_skills):
        """
        [0, 1, 2...]
        :param unique_skills: 
        :return:
        """
        sorted_skills = sorted(unique_skills)
        skills_to_int = {}
        int_to_skills = {}
        for i in range(len(sorted_skills)):
            skills_to_int[sorted_skills[i]] = i
            int_to_skills[i] = sorted_skills[i]

        self.skills_to_int = skills_to_int
        self.int_to_skills = int_to_skills

    def split_dataset(self, seqs_by_student, sample_rate=0.2, random_seed=1):
        
        sorted_keys = sorted(seqs_by_student.keys())  

        random.seed(random_seed)
        
        test_keys = set(random.sample(sorted_keys, int(len(sorted_keys) * sample_rate)))

        
        test_seqs = [seqs_by_student[k] for k in seqs_by_student if k in test_keys]
        train_seqs = [seqs_by_student[k] for k in seqs_by_student if k not in test_keys]
        return train_seqs, test_seqs

    def gen_attr(self, is_infer=False):
       
        if is_infer:
            seqs_by_students, skills = self.read_file()
            self.infer_seqs = seqs_by_students
        else:
            seqs_by_students, skills = self.read_file()
            train_seqs, test_seqs = self.split_dataset(seqs_by_students)
            self.train_seqs = train_seqs
            self.test_seqs = test_seqs

        self.gen_dict(skills)

    def pad_sequences(self, sequences, maxlen=None, value=0.):
        
        lengths = [len(s) for s in sequences]
        
        nb_samples = len(sequences)
        
        if maxlen is None:
            maxlen = np.max(lengths)
        
        x = (np.ones((nb_samples, maxlen)) * value).astype(np.int32)

        
        for idx, s in enumerate(sequences):
            trunc = np.asarray(s, dtype=np.int32)
            x[idx, :len(trunc)] = trunc

        return x

    def num_to_one_hot(self, num, dim):
        
        base = np.zeros(dim)
        if num >= 0:
            base[num] += 1
        return base

    def format_data(self, seqs):
        
        seq_len = np.array(list(map(lambda seq: len(seq) - 1, seqs)))
        max_len = max(seq_len) 
        
        x_sequences = np.array([[(self.skills_to_int[j[0]] + self.num_skills * j[1]) for j in i[:-1]] for i in seqs])
        
        x = self.pad_sequences(x_sequences, maxlen=max_len, value=-1)

       
        input_x = np.array([[self.num_to_one_hot(j, self.num_skills * 2) for j in i] for i in x])

        
        target_id_seqs = np.array([[self.skills_to_int[j[0]] for j in i[1:]] for i in seqs])
        target_id = self.pad_sequences(target_id_seqs, maxlen=max_len, value=0)

        target_correctness_seqs = np.array([[j[1] for j in i[1:]] for i in seqs])
        target_correctness = self.pad_sequences(target_correctness_seqs, maxlen=max_len, value=0)

        return dict(input_x=input_x, target_id=target_id, target_correctness=target_correctness,
                    seq_len=seq_len, max_len=max_len)

    def next_batch(self, seqs):
        length = len(seqs)
        num_batchs = length // self.batch_size
        start = 0
        for i in range(num_batchs):
            batch_seqs = seqs[start: start + self.batch_size]
            start += self.batch_size
            params = self.format_data(batch_seqs)

            yield params

            
fileName = "./assistments.txt"
dataGen = DataGenerator(fileName, config)
dataGen.gen_attr()

B_pred=[]
Pred=[]
Pred_all=[]

ACC=[]
Auc=[]
Loss=[]
Precision=[]
Recall=[]


def load_modelGru(fileName):
    newFile = "./assistments.txt"
    config = Config()

    
    dataGen = DataGenerator(newFile, config)
    dataGen.gen_attr()

    test_seqs = dataGen.test_seqs

    with tf.compat.v1.Session() as sess:

        accuracys = []
        aucs = []
        step = 1


        for params in dataGen.next_batch(test_seqs):
            print("step: {}".format(step))

            saver = tf.compat.v1.train.import_meta_graph("home/modelDktGru/my-model-800.meta")
            saver.restore(sess, tf.train.latest_checkpoint("home/modelDktGru/"))

            
            graph = tf.compat.v1.get_default_graph()

            
            input_x = graph.get_operation_by_name("test/dkt/input_x").outputs[0]
            target_id = graph.get_operation_by_name("test/dkt/target_id").outputs[0]
            keep_prob = graph.get_operation_by_name("test/dkt/keep_prob").outputs[0]
            max_steps = graph.get_operation_by_name("test/dkt/max_steps").outputs[0]
            sequence_len = graph.get_operation_by_name("test/dkt/sequence_len").outputs[0]

            
            pred_all = graph.get_tensor_by_name("test/dkt/pred_all:0")
            pred = graph.get_tensor_by_name("test/dkt/pred:0")
            binary_pred = graph.get_tensor_by_name("test/dkt/binary_pred:0")

            target_correctness = params['target_correctness']
            pred_all, pred, binary_pred = sess.run([pred_all, pred, binary_pred],
                                                   feed_dict={input_x: params["input_x"],
                                                              target_id: params["target_id"],
                                                              keep_prob: 1.0,
                                                              max_steps: params["max_len"],
                                                              sequence_len: params["seq_len"]})
            Pred.append(pred)
            B_pred.append(binary_pred)

            step += 1
    
    predictions = []
    for i in range(125):
        predictions.append({"id": i, "prob":random.randint(0,100)})
    return predictions

if __name__ == "__main__":
    fileName = "input.txt"
    load_model(fileName)