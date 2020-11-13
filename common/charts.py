# coding=utf-8
import os
import numpy as np
import itertools
import matplotlib.pyplot as plt


def draw_param_grid(file_name, pics_dir, param_grid, cv_results, sign):

    def one_plot(axis_names, axis_lens, axis_values, scores):

        plt.imshow(scores, interpolation='nearest', cmap=plt.get_cmap('gist_rainbow'))  # , norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
        thresh = (scores.max() - scores.min()) / 2. + scores.min()
        for i, j in itertools.product(range(scores.shape[0]), range(scores.shape[1])):
            plt.text(j, i, '%.2f' % (sign * scores[i, j]), size=8, horizontalalignment="center", verticalalignment="center", color="black")  # "white" if scores[i, j] > thresh else "black"
        plt.xlabel(axis_names[1], fontsize=8)
        plt.ylabel(axis_names[0], fontsize=8)
        # plt.colorbar()
        plt.xticks(np.arange(axis_lens[1]), param_grid[axis_names[1]], fontsize=7, rotation=45)
        plt.yticks(np.arange(axis_lens[0]), param_grid[axis_names[0]], fontsize=8)
        # plt.title('Search grid')

    # сортируем оси массива результатов cv так, чтобы в 2D: [1 1 1 2 2 2],  [(10,) (20,) (30,) (10,) (20,) (30,)],
    # в 3D [1 1 1 1 2 2 2 2], [0.1 0.1 10.0 10.0 0.1 0.1 10.0 10.0], [(10,) (20,) (10,) (20,) (10,) (20,) (10,) (20,)]
    # param_names = param_grid.keys()
    # отсортируем имена по количеству значений, вначале должны идти с меньшим количеством, чтобы в ситуации 1 1 1 2 2 2 не разбить этот массив на 3 части
    sorted_param_names = sorted([str(len(param_grid[key])) + '-' + key for key in param_grid.keys()])  # sorted({str(len(param_grid[key]))+'_'+key:(key, param_grid[key]) for key in param_grid.keys()})
    param_names = [param_name.split('-')[1] for param_name in sorted_param_names]  # отрежем теперь ненужную приставку-номер
    axis_names = []
    len_arr = len(cv_results['params'])  # начальный размер
    while len(axis_names) < len(param_names):
        for param_name in param_names:
            if not (param_name in axis_names):
                arr = cv_results['param_' + param_name].data
                if (len_arr % len(param_grid[param_name])) != 0:
                    continue
                len_piece_arr = len_arr / len(param_grid[param_name])  # длину разделить на количество возможных значений
                piece_arr = arr[:len_piece_arr]
                if len(np.unique(piece_arr)) == 1:
                    axis_names.append(param_name)
                    len_arr = len_piece_arr  # текущий размер
    axis_lens = [len(param_grid[axis_name]) for axis_name in axis_names]
    scores = cv_results['mean_test_score'].reshape(axis_lens)  # трансформируем в многомерный массив и исправляем отрицательные ошибки чтобы искать минимизацию
    # print(scores)

    # plt.set_printoptions(precision=2)
    if not os.path.exists(pics_dir):
        os.makedirs(pics_dir)
    scale = 0.7
    max_idx = np.unravel_index(np.argmax(scores), scores.shape)  # -> (1,1,2) определяем, где минимум, чтобы делать 2D срезы относительно этого минимума
    if scores.ndim == 1:
        pass
    elif scores.ndim == 2:
        plt.figure(figsize=(max(scale * max(axis_lens), 2), max(scale * max(axis_lens), 2)))  # plt.figure(figsize=(0.5*axis_lens[1], 0.5*axis_lens[0]))
        # plt.subplots_adjust(left=.2, right=0.95, bottom=0.25, top=0.95)
        one_plot(axis_names, axis_lens, [param_grid[axis_names[0]], param_grid[axis_names[1]]], scores)
    elif scores.ndim == 3:
        plt.figure(figsize=(max(scale * max(axis_lens) * 3, 2), max(scale * max(axis_lens), 2)))  # 0.5*max(axis_lens)
        # plt.subplots_adjust(left=.04, right=.99, bottom=.05, top=.95, hspace = 0.25)

        # a[1,:,:] - индексы 1 2
        plt.subplot(1, scores.ndim, 1)
        one_plot([axis_names[1], axis_names[2]], [axis_lens[1], axis_lens[2]],
                 [param_grid[axis_names[1]], param_grid[axis_names[2]]], scores[max_idx[0], :, :])

        # a[:,1,:] = - индексы 0 2
        plt.subplot(1, scores.ndim, 2)
        one_plot([axis_names[0], axis_names[2]], [axis_lens[0], axis_lens[2]],
                 [param_grid[axis_names[0]], param_grid[axis_names[2]]], scores[:, max_idx[1], :])

        # a[:,:,2]
        plt.subplot(1, scores.ndim, 3)
        one_plot([axis_names[0], axis_names[1]], [axis_lens[0], axis_lens[1]],
                 [param_grid[axis_names[0]], param_grid[axis_names[1]]], scores[:, :, max_idx[2]])
    elif scores.ndim == 4:
        plt.figure(figsize=(max(scale * max(axis_lens) * 5, 2), max(scale * max(axis_lens), 2)))
        # plt.subplots_adjust(left=.04, right=.99, bottom=.05, top=.95, hspace = 0.25)

        # a[1,1,:,:] - индексы 2 3
        plt.subplot(1, scores.ndim, 1)
        idx1, idx2 = 2, 3
        one_plot([axis_names[idx1], axis_names[idx2]], [axis_lens[idx1], axis_lens[idx2]],
                 [param_grid[axis_names[idx1]], param_grid[axis_names[idx2]]],
                 scores[max_idx[0], max_idx[1], :, :])

        # a[1,:,1,:] - индексы 1 3
        plt.subplot(1, scores.ndim, 2)
        idx1, idx2 = 1, 3
        one_plot([axis_names[idx1], axis_names[idx2]], [axis_lens[idx1], axis_lens[idx2]],
                 [param_grid[axis_names[idx1]], param_grid[axis_names[idx2]]],
                 scores[max_idx[0], :, max_idx[2], :])

        # a[1,:,:,1] - индексы 1 2
        plt.subplot(1, scores.ndim, 3)
        idx1, idx2 = 1, 2
        one_plot([axis_names[idx1], axis_names[idx2]], [axis_lens[idx1], axis_lens[idx2]],
                 [param_grid[axis_names[idx1]], param_grid[axis_names[idx2]]],
                 scores[max_idx[0], :, :, max_idx[3]])

        # a[:,1,:,1] - индексы 0 2
        plt.subplot(1, scores.ndim, 4)
        idx1, idx2 = 0, 2
        one_plot([axis_names[idx1], axis_names[idx2]], [axis_lens[idx1], axis_lens[idx2]],
                 [param_grid[axis_names[idx1]], param_grid[axis_names[idx2]]],
                 scores[:, max_idx[1], :, max_idx[3]])

        # a[:,:,1,1] - индексы 0 1
        plt.subplot(1, scores.ndim, 5)
        idx1, idx2 = 0, 1
        one_plot([axis_names[idx1], axis_names[idx2]], [axis_lens[idx1], axis_lens[idx2]],
                 [param_grid[axis_names[idx1]], param_grid[axis_names[idx2]]],
                 scores[:, :, max_idx[2], max_idx[3]])

    elif scores.ndim == 5:
        pass
    plt.tight_layout()
    plt.savefig(pics_dir + '/' + file_name + '.png')  # plt.savefig(path_dir + '/' + file_name[:-7] + '_'.join(('_Odor%d'%odor, 'FreqWidth%d'%freq_width, 'TrainLen%s'%train_len, 'TestLen%s'%test_len)) + '.png')

'''
def test_param_grid():
    a = np.array([[[1,2,3],
                   [4,5,6],
                   [7,8,9]],
                  [[10,20,30],
                   [40,50,0],
                   [70,80,90]]])
    print(np.argmin(a), np.unravel_index(np.argmin(a), a.shape)) # 14 -> (1,1,2)
    #a[1,:,:]
    #a[:,1,:]
    #a[:,:,2]
test_param_grid()
'''
"""
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
"""
