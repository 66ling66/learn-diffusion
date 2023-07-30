
import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    '''这个函数定义了一个自注意力操作
    输入:要处理的注意力数据
    输出：处理后的数据
    维度：输入前与输入后维度不变
    '''
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True) #输入注意力的channel数，注意力的头数，以及输入的第一个维度是batch
        self.ln = nn.LayerNorm([channels])  #表明只对channels维度进行归一化
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2) #先改变维度到[batch_size,h*w,channels]
        #[batch_size,h*w,channels]
        x_ln = self.ln(x)         #在channels这个维度进行归一化
        #[batch_size,h*w,channels]
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)  #这个函数会返回一个注意力值和注意力权重（softmax（q*v）） 输入是kqv这里kqv是一样的
        #[batch_size,h*w,channels]
        attention_value = attention_value + x  #残差
        attention_value = self.ff_self(attention_value) + attention_value  #残差
        #[batch_size,h*w,channels]
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    '''这个函数定义了卷积操作具体流程是：卷积 分组归一化 激活函数 卷积 分组归一化 残差连接可选
    输入:待处理的图像 中间的channels
    输出:卷积操作后的特征图
    '''
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),  #第一个参数表示分组的数量 第二个参数表示输入的通道数
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    '''这个函数定义了下采样，同时将时间步长编码加入到图像中,流程：最大池化，
    双卷积同时进行残差连接，对时间步长编码进行一个线性变化然后升维加到卷积操作后的数据中
    输入:待下采样的特征图，时间步长编码
    输出：卷积后与时间步长融合的特征图
    '''
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), #最大池化 池化窗口是2 所以高宽都会缩小到原来的两倍
            DoubleConv(in_channels, in_channels, residual=True), #做卷积操作
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        #t[batch_size,256]
        x = self.maxpool_conv(x)
        '''将时间步长编码加入到卷积操作之后的数据里'''
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        '''这个函数做了一个上采样的操作，流程是：先做双线性插值进行上采样，然后做双卷积，对时间步长进行线性变化加到卷积后的数据中
        输入：待卷积的特征图，与之对应的下采样的卷积后的特征图，时间步长编码
        输出：卷积操作后与时间步融合的特征图
        '''
        super().__init__()
        '''这里进行了一个上采样操作，使用双线性插值法 放大因子是2说明高宽会被放大到原来的两倍 角对其说明其会通过对其角上的元素避免边缘效应'''
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(   #做一个双卷积
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)  #与对应的下采样进行残差连接
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):  #定义了进入模型的通道数和输出模型的通道数
        super().__init__()
        self.device = device
        self.time_dim = time_dim   #时间的嵌入维度
        self.inc = DoubleConv(c_in, 64)      #卷积
        self.down1 = Down(64, 128)           #下采样
        self.sa1 = SelfAttention(128, 32)    #自注意力机制
        self.down2 = Down(128, 256)          #下采样
        self.sa2 = SelfAttention(256, 16)    #自注意力机制
        self.down3 = Down(256, 256)          #上采样
        self.sa3 = SelfAttention(256, 8)     #自注意力机制

        self.bot1 = DoubleConv(256, 512)     #中间的卷积层
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)             #上采样
        self.sa4 = SelfAttention(128, 16)   #自注意力机制
        self.up2 = Up(256, 64)              #上采样
        self.sa5 = SelfAttention(64, 32)    #自注意力机制
        self.up3 = Up(128, 64)              #上采样
        self.sa6 = SelfAttention(64, 64)    #自注意力机制
        self.outc = nn.Conv2d(64, c_out, kernel_size=1) #卷积出来

    def pos_encoding(self, t, channels):
        '''这个函数是用来进行对时间步长进行编码的
        输入:维度是[batch_size,timestep]和通道数 ,一次性计算所有时间步长的位置编码
        输出:一个时间位置编码
        '''
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)  #升一个维度
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output






class UNet_conditional(nn.Module):   #改进的代码 num_classes是类别
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:        #如果num_classes不是None 那么对类别进行一个embedding 使其维度与timeembedding类似
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):  #unet前向传播的过程需要加入类别数据 与时间embedding一起作为引导
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:   #引导时直接与时间步长相加
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


if __name__ == '__main__':
    net = UNet(device="cpu")
    # net = UNet_conditional(num_classes=10, device="cpu")
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 3, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()  #条件
    print(net(x, t).shape)
