import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class U_Network(nn.Module):
    def __init__(self, dim, enc_nf, dec_nf, bn=None, full_size=True):
        super(U_Network, self).__init__()
        # 是否使用批量归一化
        self.bn = bn
        # 图像的维度
        self.dim = dim
        # 编码器的通道数
        self.enc_nf = enc_nf
        # 用于指示是否使用全尺寸解码器
        self.full_size = full_size
        # 根据dec_nf列表的长度来判断是否使用voxelmorph 2
        self.vm2 = len(dec_nf) == 7
        """
        创建了一个编码器网络,其中包含了多个卷积层，
        每个卷积层都使用不同的通道数，通道数由 enc_nf 列表中的值确定。
        循环中的每次迭代都创建一个新的卷积层,
        通道数从 prev_nf 变为 enc_nf[i]，
        并将其添加到编码器网络中。
        编码器用于图像特征提取。
        """
        self.enc = nn.ModuleList()
        # 遍历编码器层的数量
        for i in range(len(enc_nf)):
            # [2, ]
            prev_nf = 2 if i == 0 else enc_nf[i - 1]
            """self.conv_block()
            用于创建一个卷积块convolutional block
            将前一个编码器层的通道数 prev_nf 作为输入通道数,
            当前编码器层的通道数 enc_nf[i] 作为输出通道数，
            卷积核大小为4x4，步幅（stride）为2,
            并根据 batchnorm 参数来选择是否添加批量归一化层。
            """
            self.enc.append(self.conv_block(dim, prev_nf, enc_nf[i], 4, 2, batchnorm=bn))
        """
        创建解码器网络，其中包含了多个卷积层（卷积块），
        每个卷积层用于将特征图从编码器层的通道数转换为不同的通道数。
        解码器的作用是还原特征图的空间尺寸，以便生成输出。
        解码器用于图像生成。
        """
        self.dec = nn.ModuleList()
        # 使用了编码器最后一层的通道数作为输入通道数
        self.dec.append(self.conv_block(dim, enc_nf[-1], dec_nf[0], batchnorm=bn))  # 1
        self.dec.append(self.conv_block(dim, dec_nf[0] * 2, dec_nf[1], batchnorm=bn))  # 2
        self.dec.append(self.conv_block(dim, dec_nf[1] * 2, dec_nf[2], batchnorm=bn))  # 3
        self.dec.append(self.conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3], batchnorm=bn))  # 4
        self.dec.append(self.conv_block(dim, dec_nf[3], dec_nf[4], batchnorm=bn))  # 5

        if self.full_size:
            self.dec.append(self.conv_block(dim, dec_nf[4] + 2, dec_nf[5], batchnorm=bn))
        if self.vm2:
            self.vm2_conv = self.conv_block(dim, dec_nf[5], dec_nf[6], batchnorm=bn)
        # 上采样层，用于将解码器的特征图上采样，将其尺寸扩大2倍。
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        """
        定义了一个卷积层用于生成流场，并对卷积层的权重和偏置进行了初始化。
        此外，还创建了一个批量归一化层以规范化输入数据。
        """
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        # 创建了一个卷积层（Conv层），用于生成流场
        """
        dec_nf[-1] 是前一个解码器层的输出通道数，作为卷积层的输入通道数
        dim 是图像维度，用于确定创建2D还是3D卷积层
        kernel_size=3 指定卷积核的大小为3x3
        padding=1 指定卷积的填充大小为1，以保持输入和输出的尺寸一致
        """
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)
        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        # 创建了一个批量归一化（Batch Normalization）层，用于规范化输入数据
        self.batch_norm = getattr(nn, "BatchNorm{0}d".format(dim))(3)

    def conv_block(self, dim, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batchnorm=False):
        # 动态构造了卷积层和批量归一化层的类名
        # import torch.nn as nn
        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        bn_fn = getattr(nn, "BatchNorm{0}d".format(dim))
        """
        如果 batchnorm 为真，创建一个包含卷积、批量归一化和LeakyReLU激活函数的层
        如果 batchnorm 为假，创建一个只包含卷积和LeakyReLU激活函数的层
        """
        if batchnorm:
            layer = nn.Sequential(
                # conv_fn是构建的类名
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                bn_fn(out_channels),
                nn.LeakyReLU(0.2))
        else:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.LeakyReLU(0.2))
        return layer

    """
    定义了神经网络模型的前向传播方法 forward，
    该方法用于定义输入数据如何通过模型进行前向计算以生成输出。
    这个前向传播方法定义了模型如何从输入数据中提取特征，
    通过解码器生成流场，并对流场进行必要的处理。
    """
    """
    1.将输入数据 src 和 tgt 进行拼接，形成一个新的输入张量x,这个操作通常用于将两个输入数据合并在一起
    2.通过循环遍历编码器层，逐渐减小图像的尺寸并提取特征
        循环中的每一步都对输入数据进行一次卷积操作，并将结果存储在 x_enc 列表中，以供后续的解码器使用。
    3.解码器部分根据编码器的特征逐步生成输出
        a.前三个解码器层（self.dec[0]、self.dec[1]、self.dec[2]）执行卷积操作
        b.使用上采样（self.upsample）将特征图的尺寸扩大
        c.将其与相应编码器层的特征图进行拼接
        d.后两个解码器层（self.dec[3]、self.dec[4]）用于在全尺寸的一半分辨率上进行特征处理
    4.通过卷积层 self.flow 对最终的特征图进行卷积操作，以生成流场（flow field）
    """
    def forward(self, src, tgt):
        # 将输入数据 src 和 tgt 沿着维度1（通道维度）进行拼接
        x = torch.cat([src, tgt], dim=1)
        # 将拼接后的输入 x 添加到列表中
        x_enc = [x]
        """
        x = l(x_enc[-1])：对于每个编码器层 l，使用前一个编码器层的激活 
        x_enc[-1] 作为输入，计算当前编码器层的输出 x
        x_enc.append(x)：将当前编码器层的输出 x 添加到 x_enc 列表中，以便后续的解码器使用
        """
        for i, l in enumerate(self.enc):
            x = l(x_enc[-1])
            x_enc.append(x)
        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
            # 卷积操作
            y = self.dec[i](y)
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)
        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)
        # Upsample to full res, concatenate and conv
        if self.full_size:
            y = self.upsample(y)
            y = torch.cat([y, x_enc[0]], dim=1)
            y = self.dec[5](y)
        # Extra conv for vm2
        if self.vm2:
            y = self.vm2_conv(y)
        flow = self.flow(y)
        if self.bn:
            flow = self.batch_norm(flow)
        return flow


"""
SpatialTransformer 模块可以接受源图像和流场作为输入，
然后根据流场将源图像进行空间变换，并返回变换后的图像。
"""
class SpatialTransformer(nn.Module):
    """
    构造函数用于初始化空间变换器,接受两个参数：
    size：一个表示输出尺寸的元组或列表，通常包含图像的高度和宽度（2D图像）或高度、宽度和深度（3D图像）
    mode：插值模式，用于指定如何在变换时对输入数据进行插值，默认为双线性插值（'bilinear'）
    """
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode
    """
    前向传播方法接受两个输入：
    src：源图像，即需要进行变换的输入数据
    flow：流场（flow field），用于描述如何对输入数据进行变换
    """
    def forward(self, src, flow):
        # 通过将流场（flow field）添加到采样网格上，计算新的位置坐标
        new_locs = self.grid + flow
        # 标准化坐标,获取流场的空间维度
        shape = flow.shape[2:]

        # 对 new_locs 中的每个坐标维度，将其值标准化到 [-1, 1] 的范围内，以适应PyTorch的插值操作要求
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        """
        使用 F.grid_sample 函数对源图像 src 进行采样，根据新的位置坐标进行变换,
        采样时使用了在构造函数中指定的插值模式（self.mode）
        """
        return F.grid_sample(src, new_locs, mode=self.mode)
