import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet
import logging
from torch.utils.tensorboard import SummaryWriter


'''这串代码设置了一个显示日志的东西 可以显示跟训练有关的内容'''
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    '''在diffusion中我们首先要知道 加入噪声的时间步数，β的起始到结束是多少，要生成图像的size'''
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps  #加噪的时间步数
        self.beta_start = beta_start  #β起始
        self.beta_end = beta_end    #β终点
        self.img_size = img_size    #图像的size
        self.device = device    #在什么设备上跑

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta   #获得α
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) #获得α_hat 这个函数的作用是累乘

    def prepare_noise_schedule(self):
        '''这个代码定义了一个线性增长β的过程
        输入:β的起始和终点
        返回:为noise_step的一维数组'''
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        '''这个函数定义了diffusion的前向过程 根据公式直接一步把噪声加到图片上
        输入:要加噪声的图片和要加噪声的步数
        输出:加上噪声后的图片和加的噪声
        '''
        # sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_alpha_hat=torch.sqrt(self.alpha_hat[t])[:,None:,None,None]     #后面的[]就是调整他的维度调整后的维度是[1,1,1,1]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)  #生成噪声
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        '''这段代码定义了一个随机生成加噪/去噪的步数 范围为1-noise_steps
        输入:想要多少个时间步n
        输出:一个长度为n的tiem_steps
        '''
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        '''这段代码是在训练过程中对模型进行采样的代码
        输入:一个去噪的模型model和你想生成的图片数量n
        输出:一张去噪后的图片
        '''
        logging.info(f"Sampling {n} new images....")   #显示采样的进程
        model.eval()    #将模型设为评估模式
        #将模型设为不需要计算梯度
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device) #生成一个batch为n 的原始噪声
            '''这里的tqdm设置了一个进度调 它展示了遍历 tqdm里面的’容器‘的进程 position 表示将进度条放在顶端'''
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                #生成当前对应的时间步
                t = (torch.ones(n) * i).long().to(self.device)
                #通过model去除噪声
                predicted_noise = model(x, t)
                '''获取对应的alpha，beta 这里的t由于有.long（）所有可以作为下标传入'''
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                '''基于预测噪声去除噪声的关键步骤，去噪过程加入随机噪声目的是引入扰动，这样可以产生随机性'''
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        #这一步是首先将输出的x中的元素值转换为-1--1的区间，然后再将其转换到0--1的区间
        x = (x.clamp(-1, 1) + 1) / 2
        #输出图像格式
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    #设置日志
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args) #得到数据集
    model = UNet().to(device)  #实例化unet
    optimizer = optim.AdamW(model.parameters(), lr=args.lr) #定义好学习率
    mse = nn.MSELoss()  #定义损失
    diffusion = Diffusion(img_size=args.image_size, device=device)  #实例化diffusion过程
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)  #获取batch_size个时间步
            x_t, noise = diffusion.noise_images(images, t) #获取加噪声后的图片，和所加的噪声
            predicted_noise = model(x_t, t)   #预测噪声
            loss = mse(noise, predicted_noise) #计算损失

            optimizer.zero_grad()   #梯度清零
            loss.backward()         #计算梯度
            optimizer.step()        #更新参数

            # pbar.set_postfix(MSE=loss.item())#这行代码用于更新进度条 pbar 的后缀信息。
            pbar.set_postfix(MSE=loss.item())
            #这行代码用于向 TensorBoard 日志中添加一个标量值
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))


def launch():
    import argparse
    '''这行代码创建了一个 ArgumentParser 对象，它会保存所有需要处理的命令行参数信息。'''
    parser = argparse.ArgumentParser()
    '''这行代码解析命令行参数。在这个例子中，由于我们没有在 ArgumentParser 对象中添加任何参数，所以 parse_args() 会返回一个空的 Namespace 对象'''
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 500
    args.batch_size = 1
    args.image_size = 64
    args.dataset_path ='./data'
    args.device = "cuda"
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet().to(device)
    # ckpt = torch.load("./working/orig/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # x = diffusion.sample(model, 8)
    # print(x.shape)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in x.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    # plt.show()
