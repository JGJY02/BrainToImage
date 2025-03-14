from matplotlib import pyplot as plt


def training_curves(history,model_name,save_name,epochs=None):
    # visualize training
    acc = history['accuracy']
    val_acc = history['val_accuracy']

    loss = history['loss']
    val_loss = history['val_loss']

    if epochs:
        epochs_range = range(epochs)
    else:
        epochs_range = range(len(history['loss'])) #(epochs)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'Training and Validation Accuracy\n{model_name}')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'Training and Validation Loss\n{model_name}')

    plt.savefig(f"{save_name}",bbox_inches='tight')
    plt.show()