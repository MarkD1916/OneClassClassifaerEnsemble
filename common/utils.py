# coding=utf-8
import os
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from collections import Counter, OrderedDict
import json
from glob import glob

def form_classif_result(actual, predicted, lb, line=True, table=False):
    try: # при распознавании единичного предъявления расчет точности невозможен
        num_classes = len(lb.classes_)
        num_examps = Counter(actual)#np.sum(actual, axis=0)
        # auc = roc_auc_score(actual.reshape(-1), lb.transform(np.array(predicted)).reshape(-1))
        #categs = le.inverse_transform(le.classes_)
        #lb = LabelBinarizer().fit(categs)
        auc = roc_auc_score(lb.transform(np.array(actual)).reshape(-1), lb.transform(np.array(predicted)).reshape(-1))
        # if num_classes > 2:
        #     actual = np.argmax(actual, axis=1)
        #     #predicted = np.argmax(predicted, axis=1)
        # else:
        #     actual = actual.flatten()
        cm = confusion_matrix(actual, predicted, labels=lb.classes_)
        if line:
            acc_str = '{:.1f}('.format(100.*accuracy_score(actual, predicted)) # ave acc
            acc_lst = [] # acc by categs
            for i in range(num_classes): # acc for each categ
                acc_str += '{:.1f} '.format(100.*cm[i,i]/num_examps[i])
                acc_lst.append(100.*cm[i,i]/num_examps[i])
            # if num_classes > 2:
            #     for i in range(num_classes): # acc for each categ
            #         acc_str += '{:.1f} '.format(100.*cm[i,i]/num_examps[i])
            #         acc_lst.append(100.*cm[i,i]/num_examps[i])
            # else:
            #     acc_str += '{:.1f} '.format(100.*cm[0,0]/(len(actual) - num_examps[0]))
            #     acc_str += '{:.1f} '.format(100.*cm[1,1]/num_examps[0])
            #     acc_lst.append(100.*cm[0,0]/(len(actual) - num_examps[0]))
            #     acc_lst.append(100.*cm[1,1]/num_examps[0])
            acc_str = acc_str[:-1]+')' # вместо последнего пробела
        else:
            acc_lst = None
            acc_str = ''
        if table: # in acc_str add table of classification
            if line: acc_str += '\n'
            acc_lst = []
            for i in range(num_classes): # head
                acc_str += '{:5d} '.format(lb.classes_[i])
            acc_str += '\n'
            for i in range(num_classes):
                acc_lst.append([100.*acc/num_examps[i] for acc in cm[i]]) # значения точностей для i-го класса
                acc_str += '{:d} '.format(lb.classes_[i])
                for j in range(num_classes):
                    acc_str += '{:3.1f} '.format(100.*cm[i,j]/num_examps[i])
                if i != (num_classes-1): acc_str += '\n'
    except:
        auc = 0.0;   acc_str = '';   acc_lst = []
    return auc, acc_str, acc_lst

def print_log(log_, log, verbose=True):
    if verbose: print(log_)
    log += log_ + '\n'
    return log

# посчитаем количество примеров каждого класса
def count_examples(data_xy, categs, verbose=True): #noday=True,
    def count(log, tab, subj, day=''):
        #tab = '\t' if not noday else ''
        categs_id_line = ''
        examps_count_line = ''
        tot_num_values = 0
        for idx in range(num_categs):
            categ = categs[idx]
            num_values = np.sum(data_xy[subj][day][series][1] == categ)#(idx+1)) #np.sum(data_xy[subj][series][1] == (idx+1)) if noday else
            categs_id_line += '%s\t' % categ
            examps_count_line += '%d\t' % num_values
            tot_num_values += num_values
        log += tab+'\tCategs:\t%s\tTotal\n' % categs_id_line
        log += tab+'\tCount:\t%s\t%d\n' % (examps_count_line, tot_num_values)
        return log

    num_categs = len(categs)
    log = ''
    tab = ''
    for subj in sorted(data_xy):
        if subj != '':
            log += 'Subj:\t"%s"\n' % subj
            tab = '\t'
        for day in sorted(data_xy[subj]):
            if day != '':
                log += tab+'Day:\t"%s"\n' % day
                tab += '\t'
            for series in sorted(data_xy[subj][day]):
                log += tab + 'Series:\t%s\n' % str(series)
                log = count(log, tab, subj, day)
    if verbose:
        print(log)
    return log

# сохранение настроек и результатов точности и длительности вычислительного эксперимента
# добавляется новая точка при каждом вызове batch
def save_batch_acc_dur_result(save_file_name, log_file_name, accs, elapsed, loaded_settings): #mode, take_models, cv_space, take_freqs, diff, augment, committee):
    if not os.path.exists(save_file_name): # создаем пустой файл
        with open(save_file_name, 'w') as f:
            json.dump(OrderedDict(zip(["timestamps","accuracies","durations","settings"],[[],[],[],[]])), f, indent=2)#("timestamps"=[],"accuracies"=[],"durations"=[],"settings"={}), f, indent=2)
    with open(save_file_name) as f:
        experiments = json.load(f)
    # добавляем точку
    timestamp = log_file_name #'%d-%02d-%02d_%02d%02d%02d' % (time.localtime()[:6])
    timestamps = list(experiments["timestamps"])
    timestamps.append(timestamp)
    accuracies = list(experiments["accuracies"])
    accuracies.append(np.mean(accs))
    durations = list(experiments["durations"])
    durations.append(elapsed)
    settings = OrderedDict(experiments["settings"])#OrderedDict()
    #augment_name, augment_range, augment_window, augment_step = augment
    settings[timestamp] = loaded_settings
        # {"mode":mode, "take_models":take_models,
        # "cv":{"space":{"num_components":cv_space[-1], "spatial_filters": cv_space[0], "second_spatial_filters":cv_space[1], #.keys()
        #                "classifiers":cv_space[2]}},#"reguls":cv_space['space']['reguls'],
        # "fe":{"take_freqs":take_freqs, "diff":diff},
        # "augment":{"name":augment_name, "range":augment_range, "window":augment_window, "step":augment_step},
        # "committee":committee}
    # запоминаем настройки эксперимента
    with open(save_file_name, 'w') as f:
        json.dump(OrderedDict(zip(["timestamps","accuracies","durations","settings"],[timestamps,accuracies,durations,settings])), f, indent=2)#{"timestamps": timestamps, "accuracies": accuracies, "durations": durations, "settings": settings}, f, indent=2)
    return timestamps, accuracies, durations



