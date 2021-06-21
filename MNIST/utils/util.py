import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import math

irange = range


def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """

    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new_full((3, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding) \
                .narrow(2, x * width + padding, width - padding) \
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=3, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    #im = im.convert('L')
    im.save(filename, quality=100)


# reference : https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py
def calc_gradient_penalty(netD, real_data, fake_data, use_gpu = True, dec_output=1):
    alpha = torch.rand(real_data.shape[0], 1)
    if len(real_data.shape) == 4:
        alpha = alpha.unsqueeze(2).unsqueeze(3)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_gpu else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_gpu:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    if dec_output==2:
        disc_interpolates,_ = netD(interpolates)
    elif dec_output == 3:
        disc_interpolates,_,_ = netD(interpolates)
    else:
        disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                    grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_gpu else torch.ones(
                                  disc_interpolates.size()),
                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def ae_loss(recon_x, x, mu, logvar, rec_loss):
    loss = rec_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return loss, KLD


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0]) + int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并
    # 将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0 - total1) ** 2).sum(2)
    # 调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    # 高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # 得到最终的核矩阵
    return sum(kernel_val)  # /len(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # 根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss  # 因为一般都是n==m，所以L矩阵一般不加入计算

def tensorToVar(tensor):
    """
    Convert a tensor to Variable
    If cuda is avaible, move to GPU
    """
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


def subtract_mean_batch(batch, img_type='color', grey_mean_shift=0):
    """
    Convert image batch to BGR and subtract imagenet mean
    Batch Size: (B, C, H, W), RGB
    Convert BGR to gray by: [0.114, 0.587, 0.299]
    """
    vgg_mean_bgr = np.array([103.939, 116.779, 123.680])  # inputNormalizedBGR
    grey_mean = np.array([np.dot(vgg_mean_bgr, np.array([0.114, 0.587, 0.299]))]*3)
    if img_type == 'color':
        mean_bgr = vgg_mean_bgr
    elif img_type == 'grey':
        mean_bgr = grey_mean + grey_mean_shift

    #batch = batch[:, [2, 1, 0], :, :]
    #batch = batch - tensorToVar(torch.Tensor(mean_bgr)).view(1, 3, 1, 1)
    batch = (batch[:, [0], :, :] + batch[:, [1], :, :] + batch[:, [2], :, :])/3
    return batch


def levels_loss(x, y, vgg_model, cnt_switch, weight):

    if cnt_switch == 0:
        x = subtract_mean_batch(x, img_type='color')
        y = subtract_mean_batch(y, img_type='color')
    else:
        x = subtract_mean_batch(x, img_type='grey')
        y = subtract_mean_batch(y, img_type='grey')

    loss_all = []
    loss = nn.MSELoss()(x, y)
    loss_all.append(loss)

    layer = ['r11', 'r31', 'r51']
    x_feat = vgg_model(x, layer)
    y_feat = vgg_model(y, layer)
    for n in range(len(x_feat)):
        loss = nn.MSELoss()(x_feat[n], y_feat[n])
        loss_all.append(loss)

    loss_sum = torch.as_tensor(0).cuda()
    weight_out = torch.zeros(4).cuda()
    for n in range(len(loss_all)):
        loss_sum = loss_sum + loss_all[n] * weight[n]
        weight_out[n] = loss_all[n]/sum(loss_all)
    return loss_sum, weight_out

def levels_loss_3(x, y, vgg_model, cnt_switch, weight):

    if cnt_switch == 0:
        x = subtract_mean_batch(x, img_type='color')
        y = subtract_mean_batch(y, img_type='color')
    else:
        x = subtract_mean_batch(x, img_type='grey')
        y = subtract_mean_batch(y, img_type='grey')

    loss_all = []
    #loss = nn.MSELoss()(x, y)
    #loss_all.append(loss)

    layer = ['r11', 'r31', 'r51']
    x_feat = vgg_model(x, layer)
    y_feat = vgg_model(y, layer)
    for n in range(len(x_feat)):
        loss = nn.MSELoss()(x_feat[n], y_feat[n])
        loss_all.append(loss)

    loss_sum = torch.as_tensor(0).cuda()
    weight_out = torch.zeros(3).cuda()
    for n in range(len(loss_all)):
        loss_sum = loss_sum + loss_all[n] * weight[n]
        weight_out[n] = loss_all[n]/sum(loss_all)
    return loss_sum, weight_out
