# python imports
import os
import glob
import warnings
# external imports
import torch
import numpy as np
import SimpleITK as sitk
from torch.optim import Adam
import torch.utils.data as Data
# internal imports
from Model import losses
from Model.config import args
from Model.datagenerators import Dataset
from Model.model import U_Network, SpatialTransformer

"""
计算一个PyTorch模型中可训练参数的总数量
model:一个PyTorch模型的实例
model.parameters():返回模型中的所有参数，包括权重和偏置
"""
def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


"""
创建目录,它会检查指定的目录路径是否存在，如果不存在则创建这些目录
"""
def make_dirs():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


"""
用于将PyTorch张量表示的图像数据保存为NIfTI格式的文件，
并确保保存的图像具有正确的元数据信息
"""
def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.result_dir, name))

"""
实现了一个图像配准的训练循环,训练了一个包含UNet和STN的模型，
该模型用于将移动图像配准到固定图像。
在训练过程中，损失函数包括相似性损失和梯度损失，用于指导模型学习正确的配准变换。
"""
def train():
    # 创建所需的文件夹，包括模型保存目录、日志目录和结果目录
    make_dirs()
    # 指定要使用的计算设备（GPU或CPU）
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    # 日志文件
    log_name = str(args.n_iter) + "_" + str(args.lr) + "_" + str(args.alpha)
    print("log_name: ", log_name)
    f = open(os.path.join(args.log_dir, log_name + ".txt"), "w")

    # 读入fixed图像
    f_img = sitk.ReadImage(args.atlas_file)
    # 转换为NumPy数组
    # 模型通常期望输入数据的维度是[batch_size, num_channels, depth, width, height]
    input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    # 用于获取固定图像的空间尺寸
    vol_size = input_fixed.shape[2:]
    # [B, C, D, W, H]
    input_fixed = np.repeat(input_fixed, args.batch_size, axis=0)
    # 将NumPy数组转换为PyTorch张量，这个张量用作训练中的固定图像输入。
    input_fixed = torch.from_numpy(input_fixed).to(device).float()

    # 创建配准网络（UNet）和STN
    nf_enc = [16, 32, 32, 32]
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    UNet = U_Network(len(vol_size), nf_enc, nf_dec).to(device)
    STN = SpatialTransformer(vol_size).to(device)
    UNet.train()
    STN.train()
    # 模型参数个数
    print("UNet: ", count_parameters(UNet))
    print("STN: ", count_parameters(STN))

    # 设置优化器和损失
    opt = Adam(UNet.parameters(), lr=args.lr)
    # 损失函数包括相似性损失（similarity loss）和梯度损失（gradient loss）
    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss

    # 获取所有训练数据的名称
    train_files = glob.glob(os.path.join(args.train_dir, '*.nii.gz'))
    DS = Dataset(files=train_files)
    print("Number of training images: ", len(DS))
    # 数据加载器
    DL = Data.DataLoader(DS, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    """
    图像配准训练
    每次迭代都会更新模型的权重以优化图像配准任务
    """
    for i in range(1, args.n_iter + 1):
        # 通过数据加载器DL获取数据
        input_moving = iter(DL).next()
        # [B, C, D, W, H]，将 input_moving 转换为PyTorch张量
        input_moving = input_moving.to(device).float()

        # 通过模型运行数据以产生扭曲和流场
        # 将 input_moving 和固定图像 input_fixed 作为输入，运行数据通过模型以生成变换（warp）和流场（flow field）
        flow_m2f = UNet(input_moving, input_fixed)
        m2f = STN(input_moving, flow_m2f)

        """
        sim_loss 表示变换后的移动图像 m2f 与固定图像 input_fixed 之间的相似性损失，用来衡量它们的差异
        grad_loss 表示流场 flow_m2f 的梯度损失，用来惩罚流场的不光滑性
        将这两个损失加权相加，得到总的损失 loss
        """
        sim_loss = sim_loss_fn(m2f, input_fixed)
        grad_loss = grad_loss_fn(flow_m2f)
        loss = sim_loss + args.alpha * grad_loss
        print("i: %d  loss: %f  sim: %f  grad: %f" % (i, loss.item(), sim_loss.item(), grad_loss.item()), flush=True)
        print("%d, %f, %f, %f" % (i, loss.item(), sim_loss.item(), grad_loss.item()), file=f)

        # 反向传播和优化
        # 使用反向传播算法计算梯度，然后使用优化器 opt 更新模型的权重以最小化损失
        opt.zero_grad()
        loss.backward()
        opt.step()
        # 定期保存模型和图像
        if i % args.n_save_iter == 0:
            # Save model checkpoint
            save_file_name = os.path.join(args.model_dir, '%d.pth' % i)
            torch.save(UNet.state_dict(), save_file_name)
            # Save images
            m_name = str(i) + "_m.nii.gz"
            m2f_name = str(i) + "_m2f.nii.gz"
            save_image(input_moving, f_img, m_name)
            save_image(m2f, f_img, m2f_name)
            print("warped images have saved.")
    f.close()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()
