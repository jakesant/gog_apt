import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Confusion Matrix maps out the predicted label given to the data with the actual label
# Helps us check the rate of true/false positives and true/false negatives
# parameters are the true labels, and the predicted labels

# https://sklearn.org/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Visualizations/' + title + '.png')

def pie(csv_file, activity_encode, max):
    data = pd.read_csv(csv_file)
    data['Label'] = data['Label'].map(activity_encode)
    data, _ = [x for _, x in data.groupby(data['Label'] > max)]

    temp = data["Label"].value_counts()
    df = pd.DataFrame({'labels': list(activity_encode.keys())[:(max+1)],
                       'values': temp.values
                      })

    labels = df['labels']
    sizes = df['values']
    patches, texts = plt.pie(sizes, shadow=True, startangle=90, pctdistance=1.1, labeldistance=1.2)
    plt.legend(patches, labels, loc="best")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()