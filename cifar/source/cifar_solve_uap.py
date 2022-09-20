import keras
from keras import applications
from keras import backend as K
import tensorflow as tf
import numpy as np
import scipy
import scipy.misc
import matplotlib.pyplot as plt
import time
import imageio
import utils_backdoor
# from scipy.misc import imsave
from keras.layers import Input
from keras import Model
from keras.preprocessing.image import ImageDataGenerator
import copy
import random
from sklearn.cluster import KMeans
from sklearn import metrics

import os
import tensorflow

import pyswarms as ps

import sys

sys.path.append('../../')

DATA_DIR = '../data'  # data folder
DATA_FILE = 'cifar_dataset.h5'  # dataset file
NUM_CLASSES = 10
BATCH_SIZE = 50 * 2
RESULT_DIR = "../results/"
CA_FN = "ca.npy"

CREEN_TST = [440, 1061, 1258, 3826, 3942, 3987, 4831, 4875, 5024, 6445, 7133, 9609]
CANDIDATE = [[1, 6], [8, 0], [7, 4]]
# input size
IMG_ROWS = 32
IMG_COLS = 32
IMG_COLOR = 3
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)
CMV_SHAPE = (1, IMG_ROWS, IMG_COLS, IMG_COLOR)
INPUT_SHAPE_EXT = (1, IMG_ROWS, IMG_COLS, IMG_COLOR)
INPUT_FLAT_SIZE = IMG_ROWS * IMG_COLS * IMG_COLOR


class uap_solver:
    MINI_BATCH = 1

    def __init__(self, model, verbose, mini_batch, target_class=7, source_class=3):
        self.model = model
        self.current_class = 0
        self.verbose = verbose
        self.mini_batch = self.MINI_BATCH
        self.steps = 5000
        self.layer = [2, 6, 13]
        self.classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.target_class = target_class
        self.source_class = source_class
        self.random_sample = 1  # how many random samples
        self.plot = False
        self.rep_n = 0
        self.rep_neuron = []
        self.num_target = 1
        # split the model for causal inervention
        pass

    def split_keras_model(self, lmodel, index):

        model1 = Model(inputs=lmodel.inputs, outputs=lmodel.layers[index - 1].output)
        model2_input = Input(lmodel.layers[index].input_shape[1:])
        model2 = model2_input
        for layer in lmodel.layers[index:]:
            model2 = layer(model2)
        model2 = Model(inputs=model2_input, outputs=model2)

        return (model1, model2)

    def split_model(self, lmodel, indexes):
        # split the model to n sub models
        models = []
        model = Model(inputs=lmodel.inputs, outputs=lmodel.layers[indexes[0]].output)
        models.append(model)
        for i in range(1, len(indexes)):
            model_input = Input(lmodel.layers[(indexes[i - 1] + 1)].input_shape[1:])
            model = model_input
            for layer in lmodel.layers[(indexes[i - 1] + 1):(indexes[i] + 1)]:
                model = layer(model)
            model = Model(inputs=model_input, outputs=model)
            models.append(model)

        # output
        model_input = Input(lmodel.layers[(indexes[len(indexes) - 1] + 1)].input_shape[1:])
        model = model_input
        for layer in lmodel.layers[(indexes[len(indexes) - 1] + 1):]:
            model = layer(model)
        model = Model(inputs=model_input, outputs=model)
        models.append(model)

        return models

    def solve(self):
        # analyze input neuron importancy
        #generate data
        x_batch, y_batch = load_dataset_class(cur_class=self.source_class)

        start_time = time.time()
        self.uap_causality_analysis(x_batch)
        neu_idx = self.uap_find_neuron()

        uap_trig = self.uap_gen_trig(x_batch, neu_idx)

        #test
        sr = self.uap_test(x_batch)
        print('success rate:{}'.format(sr))

        analyze_time = time.time() - start_time
        print('analyze time: {}'.format(analyze_time))
        return

    def solve_analyze_hidden(self):
        '''
        analyze hidden neurons and find important neurons for each class
        '''
        print('Analyzing hidden neuron importancy.')
        for each_class in self.classes:
            self.current_class = each_class
            print('current_class: {}'.format(each_class))
            self.analyze_eachclass_expand(each_class)

        pass

    def solve_detect_common_outstanding_neuron(self):
        '''
        find common outstanding neurons
        return potential attack base class and target class
        '''
        print('Detecting common outstanding neurons.')

        flag_list = []
        top_list = []
        top_neuron = []

        for each_class in self.classes:
            self.current_class = each_class
            if self.verbose:
                print('current_class: {}'.format(each_class))

            top_list_i, top_neuron_i = self.detect_eachclass_all_layer(each_class)
            top_list = top_list + top_list_i
            top_neuron.append(top_neuron_i)
            # self.plot_eachclass_expand(each_class)

        # top_list dimension: 10 x 10 = 100
        flag_list = self.outlier_detection(top_list, max(top_list))
        if len(flag_list) == 0:
            return []

        base_class, target_class = self.find_target_class(flag_list)

        ret = []
        for i in range(0, len(base_class)):
            ret.append([base_class[i], target_class[i]])

        # remove classes that are natualy alike
        remove_i = []
        for i in range(0, len(base_class)):
            if base_class[i] in target_class:
                ii = target_class.index(base_class[i])
                if target_class[i] == base_class[ii]:
                    remove_i.append(i)

        out = [e for e in ret if ret.index(e) not in remove_i]
        if len(out) > 3:
            out = out[:3]
        return out

    def solve_detect_outlier(self):
        '''
        analyze outliers to certain class, find potential backdoor due to overfitting
        '''
        print('Detecting outliers.')

        tops = []  # outstanding neuron for each class

        for each_class in self.classes:
            self.current_class = each_class
            if self.verbose:
                print('current_class: {}'.format(each_class))

            # top_ = self.find_outstanding_neuron(each_class, prefix="all_")
            top_ = self.find_outstanding_neuron(each_class, prefix="")
            tops.append(top_)

        save_top = []
        for top in tops:
            save_top = [*save_top, *top]
        save_top = np.array(save_top)
        flag_list = self.outlier_detection(1 - save_top / max(save_top), 1)
        np.savetxt(RESULT_DIR + "outlier_count.txt", save_top, fmt="%s")

        base_class, target_class = self.find_target_class(flag_list)

        out = []
        for i in range(0, len(base_class)):
            if base_class[i] != target_class[i]:
                out.append([base_class[i], target_class[i]])

        # '''
        ret = []
        base_class = []
        target_class = []
        for i in range(0, len(out)):
            base_class.append(out[i][0])
            target_class.append(out[i][1])
            ret.append([base_class[i], target_class[i]])

        remove_i = []
        for i in range(0, len(base_class)):
            if base_class[i] in target_class:
                ii = target_class.index(base_class[i])
                if target_class[i] == base_class[ii]:
                    remove_i.append(i)

        out = [e for e in ret if ret.index(e) not in remove_i]
        if len(out) > 1:
            out = out[:1]
        return out

    def uap_causality_analysis(self, x_batch):
        # causality analysis on input
        self.mini_batch = self.MINI_BATCH
        do_logits_avg = []
        for idx in range(self.mini_batch):
            #X_batch, Y_batch = gen.next()
            X_batch = x_batch[:BATCH_SIZE]
            do_logtis = []

            for i in range (0, IMG_ROWS * IMG_COLS * IMG_COLOR):
                input = copy.deepcopy(X_batch.reshape(X_batch.shape[0], -1))
                input_do = np.zeros(shape=input[:, i].shape)
                input[:, i] = input_do
                do_logit = self.model.predict(input.reshape(X_batch.shape))
                ori_pre = self.model.predict(X_batch)
                do_logit = np.abs(do_logit - ori_pre)
                do_logit = np.mean(np.array(do_logit), axis=0)
                #print('do_logit shape:{}'.format(do_logit.shape))
                do_logtis.append(do_logit)
                del input

            do_logits_avg.append(do_logtis)
            print('do_logtis shape:{}'.format(np.array(do_logtis).shape))

        do_logits_avg = np.mean(np.array(do_logits_avg), axis=0)
        do_logits_avg = np.array(do_logits_avg)
        print('do_logits_avg shape:{}'.format(do_logits_avg.shape))

        np.save(RESULT_DIR + CA_FN, do_logits_avg)

        return do_logits_avg

    def uap_gen_trig(self, x_batch, neu_idx):
        '''
        generate trigger cmv's method
        '''
        weights = self.model.get_layer('dense_2').get_weights()

        self.model.get_input_shape_at(0)

        output_index = self.target_class
        reg = 0.8

        # compute the gradient of the input picture wrt this loss
        input_img = keras.layers.Input(shape=INPUT_SHAPE)

        model1 = keras.models.clone_model(self.model)
        model1.set_weights(self.model.get_weights())
        loss = K.mean(model1(input_img)[:, output_index]) - reg * K.mean(K.square(input_img))
        grads = K.gradients(loss, input_img)[0]
        # normalization trick: we normalize the gradient
        # grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])
        neu_idx = np.array(neu_idx).astype(int)
        opt_mask = np.ones(shape=(IMG_ROWS, IMG_COLS, IMG_COLOR))
        opt_mask = opt_mask.reshape(IMG_ROWS * IMG_COLS * IMG_COLOR)
        opt_mask[neu_idx] = 0
        opt_mask = opt_mask.reshape((IMG_ROWS, IMG_COLS, IMG_COLOR))
        # run gradient ascent for self.steps steps
        input_img_data = x_batch[:BATCH_SIZE]
        trig_mask = np.zeros(shape=(IMG_ROWS, IMG_COLS, IMG_COLOR))
        for batch in range (0, self.steps):
            # BATCH_SIZE = 50
            loss_value, grads_value = iterate([input_img_data])
            grads_value = np.mean(grads_value, axis=0)
            trig_mask += grads_value * opt_mask
            input_img_data = input_img_data + grads_value
            #print(loss_value / BATCH_SIZE)

        predict = self.model.predict(input_img_data)
        predict = np.argmax(predict, axis=1)
        print("prediction: {}".format(predict))

        # print(loss_value)
        '''
        img = input_img_data[0].copy()
        img = self.deprocess_image(img)

        utils_backdoor.dump_image(self.deprocess_image(ori_img),
                                  RESULT_DIR + 'cmv_ori_' + str(base_class) + '_' + str(target_class) + '_' + str(idx) + ".png",
                                  'png')

        utils_backdoor.dump_image(img,
                                  RESULT_DIR + 'cmv' + str(base_class) + '_' + str(target_class) + '_' + str(idx) + ".png",
                                  'png')
        del img
        del ori_img

        np.savetxt(RESULT_DIR + "cmv"+ str(base_class) + '_' + str(target_class) + '_' + str(idx) + ".txt", input_img_data[0].reshape(28*28*1), fmt="%s")

        img = np.loadtxt(RESULT_DIR + "cmv" + str(idx) + ".txt")
        img = img.reshape(((28,28,1)))

        predict = self.model.predict(img.reshape(1,28,28,1))
        predict = np.argmax(predict, axis=1)
        print("prediction: {}".format(predict))
        '''
        del model1
        np.save(RESULT_DIR + "uap_trig.npy", trig_mask)
        return

    def uap_find_neuron(self):
        '''
        find input neuron to optimize
        '''
        #load ca
        do_logits_avg = np.load(RESULT_DIR + CA_FN)
        idx = np.expand_dims(np.arange(0, len(do_logits_avg)),axis=0).transpose()
        target_logtis = np.expand_dims(do_logits_avg[:, self.target_class], axis=0).transpose()
        target_logtis = np.concatenate([idx, target_logtis], axis=1)

        #sort
        ind = np.argsort(target_logtis[:, 1])[::-1]
        target_logtis = target_logtis[ind]

        top_cnt = len(self.outlier_detection(target_logtis[:, 1], max(target_logtis[:, 1]), verbose=False))
        top_idx = list(target_logtis[0: (top_cnt - 1)][:, 0])

        return top_idx

    def uap_test(self, x_batch):
        '''
        test uap trigger generated
        '''
        #load trig np.save(RESULT_DIR + "uap_trig.npy", trig_mask)
        trig_mask = np.load(RESULT_DIR + "uap_trig.npy")

        utils_backdoor.dump_image(self.deprocess_image(trig_mask * 255),
                                  RESULT_DIR + "trig_mask.png",
                                  'png')

        success = 0
        total = len(x_batch) // BATCH_SIZE
        for i in range (self.MINI_BATCH, total):
            input_batch = x_batch[BATCH_SIZE * i:BATCH_SIZE * (i + 1)]
            input_batch = input_batch + trig_mask
            predict = self.model.predict(input_batch)
            predict = np.argmax(predict, axis=1)
            print("prediction: {}".format(predict))
            success = success + np.sum(predict == self.target_class)
        input_batch = x_batch[0]

        utils_backdoor.dump_image(self.deprocess_image(input_batch * 255),
                                  RESULT_DIR + "input.png",
                                  'png')
        utils_backdoor.dump_image(self.deprocess_image((input_batch + trig_mask) * 255),
                                  RESULT_DIR + "patched_input.png",
                                  'png')
        predict = self.model.predict(np.reshape((input_batch + trig_mask), INPUT_SHAPE_EXT))
        predict = np.argmax(predict, axis=1)
        print("prediction: {}".format(predict))
        return success / ((total - 1) * BATCH_SIZE)

    # util function to convert a tensor into a valid image
    def deprocess_image(self, x):
        # normalize tensor: center on 0., ensure std is 0.1
        # '''
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255

        x = np.clip(x, 0, 255).astype('uint8')
        '''
        x = np.clip(x, 0, 1)
        '''
        return x

    def analyze_eachclass_expand(self, cur_class):
        '''
        use samples from base class, find important neurons
        '''
        ana_start_t = time.time()
        self.verbose = False
        x_class, y_class = load_dataset_class(cur_class=cur_class)
        class_gen = build_data_loader(x_class, y_class)

        hidden_test = self.hidden_permutation_test_all(class_gen, cur_class)

        hidden_test_all = []
        hidden_test_name = []

        for this_class in self.classes:
            hidden_test_all_ = []
            for i in range(0, len(self.layer)):
                temp = hidden_test[i][:, [0, (this_class + 1)]]
                hidden_test_all_.append(temp)

            hidden_test_all.append(hidden_test_all_)

            hidden_test_name.append('class' + str(this_class))

        if self.plot:
            self.plot_multiple(hidden_test_all, hidden_test_name, save_n="test")

        pass

    def plot_eachclass_expand(self, cur_class, prefix=""):
        hidden_test = []
        for cur_layer in self.layer:
            hidden_test_ = np.loadtxt(
                RESULT_DIR + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
            hidden_test.append(hidden_test_)
        hidden_test = np.array(hidden_test)

        hidden_test_all = []
        hidden_test_name = []

        for this_class in self.classes:
            hidden_test_all_ = []
            for i in range(0, len(self.layer)):
                temp = hidden_test[i][:, [0, (this_class + 1)]]
                hidden_test_all_.append(temp)

            hidden_test_all.append(hidden_test_all_)

            hidden_test_name.append('class' + str(this_class))

        self.plot_multiple(hidden_test_all, hidden_test_name, save_n=prefix + "test")
        pass

    def detect_eachclass_all_layer(self, cur_class):
        hidden_test = []
        for cur_layer in self.layer:
            hidden_test_ = np.loadtxt(
                RESULT_DIR + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
            # l = np.ones(len(hidden_test_)) * cur_layer
            hidden_test_ = np.insert(np.array(hidden_test_), 0, cur_layer, axis=1)
            hidden_test = hidden_test + list(hidden_test_)

        hidden_test = np.array(hidden_test)

        # check common important neuron
        temp = hidden_test[:, [0, 1, (cur_class + 2)]]
        ind = np.argsort(temp[:, 2])[::-1]
        temp = temp[ind]

        # find outlier hidden neurons
        top_num = len(self.outlier_detection(temp[:, 2], max(temp[:, 2]), verbose=False))
        num_neuron = top_num
        if self.verbose:
            print('significant neuron: {}'.format(num_neuron))
        cur_top = list(temp[0: (num_neuron - 1)][:, [0, 1]])

        top_list = []
        top_neuron = []
        # compare with all other classes
        for cmp_class in self.classes:
            if cmp_class == cur_class:
                top_list.append(0)
                top_neuron.append(np.array([0] * num_neuron))
                continue
            temp = hidden_test[:, [0, 1, (cmp_class + 2)]]
            ind = np.argsort(temp[:, 2])[::-1]
            temp = temp[ind]
            cmp_top = list(temp[0: (num_neuron - 1)][:, [0, 1]])
            temp = np.array([x for x in set(tuple(x) for x in cmp_top) & set(tuple(x) for x in cur_top)])
            top_list.append(len(temp))
            top_neuron.append(temp)

        # top_list x10
        # find outlier
        # flag_list = self.outlier_detection(top_list, top_num, cur_class)

        # top_list: number of intersected neurons (10,)
        # top_neuron: layer and index of intersected neurons    ((2, n) x 10)
        return list(np.array(top_list) / top_num), top_neuron

        pass

    def find_outstanding_neuron(self, cur_class, prefix=""):
        '''
        find outstanding neurons for cur_class
        '''
        '''
        hidden_test = []
        for cur_layer in self.layer:
            #hidden_test_ = np.loadtxt(RESULT_DIR + prefix + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
            hidden_test_ = np.loadtxt(RESULT_DIR + prefix + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
            #l = np.ones(len(hidden_test_)) * cur_layer
            hidden_test_ = np.insert(np.array(hidden_test_), 0, cur_layer, axis=1)
            hidden_test = hidden_test + list(hidden_test_)
        '''
        hidden_test = np.loadtxt(RESULT_DIR + prefix + "test_pre0_" + "c" + str(cur_class) + "_layer_13" + ".txt")
        # '''
        hidden_test = np.array(hidden_test)

        # find outlier hidden neurons for all class embedding
        top_num = []
        # compare with all other classes
        for cmp_class in self.classes:
            temp = hidden_test[:, [0, (cmp_class + 1)]]
            ind = np.argsort(temp[:, 1])[::-1]
            temp = temp[ind]
            cmp_top = self.outlier_detection_overfit(temp[:, (1)], max(temp[:, (1)]), verbose=False)
            top_num.append((cmp_top))

        return top_num

    def locate_candidate_neuron(self, base_class, target_class):
        '''
        find outstanding neurons for target class
        '''
        hidden_test = []
        for cur_layer in self.layer:
            # hidden_test_ = np.loadtxt(RESULT_DIR + prefix + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
            hidden_test_ = np.loadtxt(
                RESULT_DIR + "test_pre0_" + "c" + str(base_class) + "_layer_" + str(cur_layer) + ".txt")
            # l = np.ones(len(hidden_test_)) * cur_layer
            hidden_test_ = np.insert(np.array(hidden_test_), 0, cur_layer, axis=1)
            hidden_test = hidden_test + list(hidden_test_)
        hidden_test = np.array(hidden_test)

        # find outlier hidden neurons for target class embedding
        temp = hidden_test[:, [0, 1, (target_class + 2)]]
        ind = np.argsort(temp[:, 2])[::-1]
        temp = temp[ind]
        top = self.outlier_detection(temp[:, 2], max(temp[:, 2]), verbose=False)
        ret = temp[0: (len(top) - 1)][:, [0, 1]]
        return ret

    def detect_common_outstanding_neuron(self, tops):
        '''
        find common important neurons for each class with samples from current class
        @param tops: list of outstanding neurons for each class
        '''
        top_list = []
        top_neuron = []
        # compare with all other classes
        for base_class in self.classes:
            for cur_class in self.classes:
                if cur_class <= base_class:
                    continue
                temp = np.array(
                    [x for x in set(tuple(x) for x in tops[base_class]) & set(tuple(x) for x in tops[cur_class])])
                top_list.append(len(temp))
                top_neuron.append(temp)

        flag_list = self.outlier_detection(top_list, max(top_list))

        # top_list: number of intersected neurons (10,)
        # top_neuron: layer and index of intersected neurons    ((2, n) x 10)

        return flag_list

    def find_common_neuron(self, cmv_top, tops):
        '''
        find common important neurons for cmv top and base_top
        @param tops: activated neurons @base class sample
               cmv_top: important neurons for this attack from base to target
        '''

        temp = np.array([x for x in set(tuple(x) for x in tops) & set(tuple(x) for x in cmv_top)])
        return temp

    def hidden_permutation_test_all(self, gen, pre_class, prefix=''):
        # calculate the importance of each hidden neuron
        out = []
        for cur_layer in self.layer:
            model_copy = keras.models.clone_model(self.model)
            model_copy.set_weights(self.model.get_weights())

            # split to current layer
            partial_model1, partial_model2 = self.split_keras_model(model_copy, cur_layer + 1)

            self.mini_batch = self.MINI_BATCH
            perm_predict_avg = []
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                out_hidden = partial_model1.predict(X_batch)  # 32 x 16 x 16 x 32
                ori_pre = partial_model2.predict(out_hidden)  # 32 x 10

                predict = self.model.predict(X_batch)  # 32 x 10

                out_hidden_ = copy.deepcopy(out_hidden.reshape(out_hidden.shape[0], -1))

                # randomize each hidden
                perm_predict = []
                for i in range(0, len(out_hidden_[0])):
                    perm_predict_neu = []
                    out_hidden_ = out_hidden.reshape(out_hidden.shape[0], -1).copy()
                    for j in range(0, self.random_sample):
                        # hidden_random = np.random.uniform(low=min[i], high=max[i], size=len(out_hidden)).transpose()
                        hidden_do = np.zeros(shape=out_hidden_[:, i].shape)
                        out_hidden_[:, i] = hidden_do
                        sample_pre = partial_model2.predict(out_hidden_.reshape(out_hidden.shape))  # 8k x 32
                        perm_predict_neu.append(sample_pre)

                    perm_predict_neu = np.mean(np.array(perm_predict_neu), axis=0)
                    perm_predict_neu = np.abs(ori_pre - perm_predict_neu)
                    perm_predict_neu = np.mean(np.array(perm_predict_neu), axis=0)
                    to_add = []
                    to_add.append(int(i))
                    for class_n in self.classes:
                        to_add.append(perm_predict_neu[class_n])
                    perm_predict.append(np.array(to_add))
                perm_predict_avg.append(perm_predict)
            # average of all baches
            perm_predict_avg = np.mean(np.array(perm_predict_avg), axis=0)

            # now perm_predict contains predic value of all permutated hidden neuron at current layer
            perm_predict_avg = np.array(perm_predict_avg)
            out.append(perm_predict_avg)
            # ind = np.argsort(perm_predict_avg[:,1])[::-1]
            # perm_predict_avg = perm_predict_avg[ind]
            np.savetxt(RESULT_DIR + prefix + "test_pre0_" + "c" + str(pre_class) + "_layer_" + str(cur_layer) + ".txt",
                       perm_predict_avg, fmt="%s")
            # out.append(perm_predict_avg)

        return np.array(out)

    def outlier_detection(self, cmp_list, max_val, verbose=False):
        cmp_list = list(np.array(cmp_list) / max_val)
        consistency_constant = 1.4826  # if normal distribution
        median = np.median(cmp_list)
        mad = consistency_constant * np.median(np.abs(cmp_list - median))  # median of the deviation
        min_mad = np.abs(np.min(cmp_list) - median) / mad

        # print('median: %f, MAD: %f' % (median, mad))
        # print('anomaly index: %f' % min_mad)

        flag_list = []
        i = 0
        for cmp in cmp_list:
            if cmp_list[i] < median:
                i = i + 1
                continue
            if np.abs(cmp_list[i] - median) / mad > 2:
                flag_list.append((i, cmp_list[i]))
            i = i + 1

        if len(flag_list) > 0:
            flag_list = sorted(flag_list, key=lambda x: x[1])
            if verbose:
                print('flagged label list: %s' %
                      ', '.join(['%d: %2f' % (idx, val)
                                 for idx, val in flag_list]))
        return flag_list
        pass

    def outlier_detection_overfit(self, cmp_list, max_val, verbose=True):
        flag_list = self.outlier_detection(cmp_list, max_val)
        return len(flag_list)

    def plot_multiple(self, _rank, name, normalise=False, save_n=""):
        # plot the permutation of cmv img and test imgs
        plt_row = len(_rank)

        rank = []
        for _rank_i in _rank:
            rank.append(copy.deepcopy(_rank_i))

        plt_col = len(self.layer)
        fig, ax = plt.subplots(plt_row, plt_col, figsize=(7 * plt_col, 5 * plt_row), sharex=False, sharey=True)

        col = 0
        for do_layer in self.layer:
            for row in range(0, plt_row):
                # plot ACE
                if row == 0:
                    ax[row, col].set_title('Layer_' + str(do_layer))
                    # ax[row, col].set_xlabel('neuron index')
                    # ax[row, col].set_ylabel('delta y')

                if row == (plt_row - 1):
                    # ax[row, col].set_title('Layer_' + str(do_layer))
                    ax[row, col].set_xlabel('neuron index')

                ax[row, col].set_ylabel(name[row])

                # Baseline is np.mean(expectation_do_x)
                if normalise:
                    rank[row][col][:, 1] = rank[row][col][:, 1] / np.max(rank[row][col][:, 1])

                ax[row, col].scatter(rank[row][col][:, 0].astype(int), rank[row][col][:, 1],
                                     label=str(do_layer) + '_cmv', color='b')
                ax[row, col].legend()

            col = col + 1
        if normalise:
            plt.savefig(RESULT_DIR + "plt_n_c" + str(self.current_class) + save_n + ".png")
        else:
            plt.savefig(RESULT_DIR + "plt_c" + str(self.current_class) + save_n + ".png")
        # plt.show()


def load_dataset_class_all(data_file=('%s/%s' % (DATA_DIR, DATA_FILE)), cur_class=0):
    if not os.path.exists(data_file):
        print(
            "The data file does not exist. Please download the file and put in data/ directory")
        exit(1)

    dataset = utils_backdoor.load_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

    X_train = dataset['X_train']
    Y_train = dataset['Y_train']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    # Scale images to the [0, 1] range
    x_train = X_train.astype("float32") / 255
    x_test = X_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    # x_train = np.expand_dims(x_train, -1)
    # x_test = np.expand_dims(x_test, -1)
    # print("x_train shape:", x_train.shape)
    # print(x_train.shape[0], "train samples")
    # print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = tensorflow.keras.utils.to_categorical(Y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)

    x_out = []
    y_out = []
    for i in range(0, len(x_test)):
        if np.argmax(y_test[i], axis=0) == cur_class:
            x_out.append(x_test[i])
            y_out.append(y_test[i])

    return np.array(x_out), np.array(y_out)


def load_dataset_class(data_file=('%s/%s' % (DATA_DIR, DATA_FILE)), cur_class=0):
    if not os.path.exists(data_file):
        print(
            "The data file does not exist. Please download the file and put in data/ directory")
        exit(1)

    dataset = utils_backdoor.load_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

    X_train = dataset['X_train']
    Y_train = dataset['Y_train']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    # Scale images to the [0, 1] range
    x_train = X_train.astype("float32") / 255
    x_test = X_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    # x_train = np.expand_dims(x_train, -1)
    # x_test = np.expand_dims(x_test, -1)
    # print("x_train shape:", x_train.shape)
    # print(x_train.shape[0], "train samples")
    # print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = tensorflow.keras.utils.to_categorical(Y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)

    x_test = np.delete(x_test, CREEN_TST, axis=0)
    y_test = np.delete(y_test, CREEN_TST, axis=0)

    x_out = []
    y_out = []
    for i in range(0, len(x_test)):
        if np.argmax(y_test[i], axis=0) == cur_class:
            x_out.append(x_test[i])
            y_out.append(y_test[i])

    # randomize the sample
    x_out = np.array(x_out)
    y_out = np.array(y_out)
    idx = np.arange(len(x_out))
    np.random.shuffle(idx)
    # print(idx)
    x_out = x_out[idx, :]
    y_out = y_out[idx, :]

    return np.array(x_out), np.array(y_out)


def build_data_loader(X, Y):
    datagen = ImageDataGenerator()
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE, shuffle=True)

    return generator


def build_data_loader_aug(X, Y):
    datagen = ImageDataGenerator(rotation_range=20, horizontal_flip=True)
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE)

    return generator
