import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from data_loader import GetLoader
import numpy as np
from sklearn.manifold import TSNE
from torchvision import datasets
import matplotlib.pyplot as plt
from matplotlib import cm

def test(dataset_name, epoch):
    assert dataset_name in ['source', 'target']

    model_root = 'models'
    image_root = os.path.join('/root/Data', dataset_name)

    cuda = True
    cudnn.benchmark = True
    batch_size = 128
    image_size = 28
    alpha = 0

    """load data"""

    img_transform_source = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    img_transform_target = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # if dataset_name == 'mnist_m':
    #     test_list = os.path.join(image_root, 'mnist_m_test_labels.txt')
    #
    #     dataset = GetLoader(
    #         data_root=os.path.join(image_root, 'mnist_m_test'),
    #         data_list=test_list,
    #         transform=img_transform_target
    #     )
    # else:
    #     dataset = datasets.MNIST(
    #         root='dataset',
    #         train=False,
    #         transform=img_transform_source,
    #     )

    target_list = os.path.join(image_root, 'image_label.txt')

    dataset = GetLoader(
        data_root=image_root,
        data_list=target_list,
        transform=img_transform_target
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    """ training """

    my_net = torch.load(os.path.join(
        model_root, 'mnist_mnistm_model_epoch_' + str(epoch) + '.pth'
    ))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    num_class = 15

    acc_class = [0 for _ in range(num_class)]
    count_class = [0 for _ in range(num_class)]

    tsne_results = np.array([])
    tsne_labels = np.array([])

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        class_label.resize_as_(t_label).copy_(t_label)

        class_output, _ = my_net(input_data=input_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        pred1 = class_output.data.max(1)[1]
        n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1
        index_temp = pred1.eq(t_label.data)

        for acc_index in range(batch_size):
            temp_label_index = t_label.data[acc_index]
            count_class[temp_label_index] += 1
            if index_temp[acc_index]:
                acc_class[temp_label_index] += 1

        if len(tsne_labels)==0:
            tsne_results = class_output.cpu().data.numpy()
            tsne_labels = t_label.cpu().numpy()
        else:
            tsne_results = np.concatenate((tsne_results, class_output.cpu().data.numpy()))
            tsne_labels = np.concatenate((tsne_labels, t_label.cpu().numpy()))

    plot_only = 1000
    tsne_model = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    tsne_transformed = tsne_model.fit_transform(tsne_results[:plot_only, :])
    tsne_labels = tsne_labels[:plot_only]

    # colors = cm.rainbow(np.linspace(0, 1, num_class))
    for x, y, s in zip(tsne_transformed[:, 0], tsne_transformed[:, 1], tsne_labels):
        c = cm.rainbow(int(255 * s / num_class))
        plt.scatter(x, y, c=c)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('output1.png')

    for print_index in range(len(acc_class)):
        print('Class:{}, Accuracy:{:.2f}%'.format(
            print_index,
            100. * acc_class[print_index] / count_class[print_index]))

    accu = n_correct.data.numpy() * 1.0 / n_total

    print ('epoch: %d, accuracy of the %s dataset: %f' % (epoch, dataset_name, accu))
    torch.save(accu, '/root/Data/dann_result/dann_ep_' + str(epoch) + '_' + dataset_name + '_' + str(accu) + '.pt')
