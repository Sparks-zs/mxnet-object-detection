from matplotlib import pyplot as plt


# 训练验证的细节画图
def draw_details(savefig, title, xlabel, ylabel, epochs, train,
                 val=None, train_label='train_label', val_label='val_label'):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    epoch = range(epochs)
    plt.plot(epoch, train, '.-', label=train_label)
    if val is not None:
        plt.plot(epoch, val, '--', label=val_label)
    plt.legend()

    plt.savefig(savefig)
    # plt.show()
