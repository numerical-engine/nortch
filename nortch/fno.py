import torch
import torch.nn as nn
import torch.nn.functional as F


class FNO1D(nn.Module):
    """1次元座標に対するFNO処理

    Attributes:
        * pad_size (int): パディングサイズ。非周期性対策のために、特徴量ベクトルと位置座標を結合する前にパディングを行う。
        * conv_lift (nn.Conv1d): 1D畳み込み層。特徴量ベクトルを変換する。
        * conv_wave (nn.Conv1d): 1D畳み込み層。フーリエ空間での畳み込みを行う。
        * num_modes (int): フーリエ空間での畳み込みに使用するモード数。0の場合は全てのモードを使用する。
    
    Methods:
        * pad(z:torch.Tensor, r:torch.Tensor)->torch.Tensor: 非周期性対策のために、特徴量ベクトルと位置座標を結合する前にパディングを行う。
    Note
        * 位置座標は等間隔かつr[:,0,0] < r[:,0,1] < ... < r[:,0,-1]であることを仮定している。
        * 各特徴量は同じ位置座標に対応していることを仮定している。
    """
    def __init__(self, in_features:int, out_features:int, pad_size:int, num_modes:int = 0)->None:
        super().__init__()
        self.pad_size = pad_size
        self.num_modes = num_modes
        self.conv_lift = nn.Conv1d(in_features + 1, out_features, kernel_size=1)
        self.conv_wave = nn.Conv1d(in_features + 1, out_features, kernel_size=1, dtype = torch.cfloat)
    def forward(self, z:torch.Tensor, r:torch.Tensor)->torch.Tensor:
        """forward処理

        Args:
            z (torch.Tensor): 特徴量ベクトル。shapeは(batch_size, num_features, sequence_length)。
            r (torch.Tensor): 位置座標。shapeは(batch_size, 1, sequence_length)。
        Returns:
            torch.Tensor: FNOの出力。shapeは(batch_size, num_features, sequence_length)。
        """
        z = self.pad(z, r)  # パディングを行う

        z_space = self.conv_lift(z)  # 1D畳み込みで特徴量を変換

        z_wave = torch.fft.fft(z, dim = -1)  # フーリエ変換
        if self.num_modes != 0:
            z_low_pass_filter = torch.zeros_like(z_wave)
            z_low_pass_filter[:, :, :self.num_modes] = z_wave[:, :, :self.num_modes]
            z_low_pass_filter[:, :, -self.num_modes:] = z_wave[:, :, -self.num_modes:]

            z_wave = self.conv_wave(z_low_pass_filter)  # フーリエ空間での畳み込み
        else:
            z_wave = self.conv_wave(z_wave)  # フーリエ空間での畳み込み
        z_wave = torch.fft.ifft(z_wave, dim=-1).real  # 逆フーリエ変換

        z = z_space + z_wave  # 空間特徴量とフーリエ特徴量を結合

        if self.pad_size > 0:
            z = z[:,:,self.pad_size:-self.pad_size]  # パディングを除去

        return z

    def pad(self, z:torch.Tensor, r:torch.Tensor)->torch.Tensor:
        """非周期性対策のために、特徴量ベクトルと位置座標を結合する前にパディングを行う。

        特徴量については端点の値でパディングを行い、位置座標についてはdxずつ減少(左側)もしくは増加(右側)するようにパディング。

        Args:
            z (torch.Tensor): 特徴量ベクトル。shapeは(batch_size, num_features, sequence_length)。
            r (torch.Tensor): 位置座標。shapeは(batch_size, 1, sequence_length)。

        Returns:
            torch.Tensor: padding済み特徴量。shapeは(batch_size, num_features + 1, sequence_length + 2 * pad_size)。
        """
        if self.pad_size == 0:
            return torch.cat([z, r], dim=1)

        dx = (r[:,:,1] - r[:,:,0]).unsqueeze(2)  # 位置座標の差分を計算。等間隔であることを仮定。

        left_z = z[:,:,0].unsqueeze(2).repeat(1, 1, self.pad_size)  # (batch_size, num_features, pad_size)に変換
        right_z = z[:,:,-1].unsqueeze(2).repeat(1, 1, self.pad_size)  # (batch_size, num_features, pad_size)に変換
        z_padded = torch.cat([left_z, z, right_z], dim=2)

        left_steps = torch.arange(self.pad_size, 0, -1, device=r.device, dtype=r.dtype).view(1, 1, -1)
        right_steps = torch.arange(1, self.pad_size + 1, device=r.device, dtype=r.dtype).view(1, 1, -1)
        left_r = r[:,:,:1] - dx * left_steps
        right_r = r[:,:,-1:] + dx * right_steps
        r_padded = torch.cat([left_r, r, right_r], dim=2)

        return torch.cat([z_padded, r_padded], dim=1)


class FNO2D(nn.Module):
    """2次元座標に対するFNO処理

    Attributes:
        * pad_size (int): パディングサイズ。非周期性対策のために、特徴量ベクトルと位置座標を結合する前にパディングを行う。
        * conv_lift (nn.Conv2d): 2D畳み込み層。特徴量ベクトルを変換する。
        * conv_wave (nn.Conv2d): 2D畳み込み層。フーリエ空間での畳み込みを行う。
        * num_modes (int|tuple[int,int]): フーリエ空間での畳み込みに使用するモード数。0の場合は全てのモードを使用する。
    
    Methods:
        * pad(z:torch.Tensor, r:torch.Tensor)->torch.Tensor: 非周期性対策のために、特徴量ベクトルと位置座標を結合する前にパディングを行う。
    Note
        * 位置座標は等間隔かつr[:,0,0,:] < r[:,0,1,:] < ... < r[:,0,-1,:]ならびにr[:,1,:,0] < r[:,1,:,1] < ... < r[:,1,:,-1]であることを仮定している。
        * 各特徴量は同じ位置座標に対応していることを仮定している。
    """
    def __init__(self, in_features:int, out_features:int, pad_size:int, num_modes:tuple[int,int] = (0,0))->None:
        super().__init__()
        self.pad_size = pad_size
        self.num_modes = num_modes
        self.conv_lift = nn.Conv2d(in_features + 2, out_features, kernel_size=1)
        self.conv_wave = nn.Conv2d(in_features + 2, out_features, kernel_size=1, dtype = torch.cfloat)
    def forward(self, z:torch.Tensor, r:torch.Tensor)->torch.Tensor:
        """forward処理

        Args:
            z (torch.Tensor): 特徴量ベクトル。shapeは(batch_size, num_features, width, height)。
            r (torch.Tensor): 位置座標。shapeは(batch_size, 2, width, height)。
        Returns:
            torch.Tensor: FNOの出力。shapeは(batch_size, num_features, width, height)。
        """
        z = self.pad(z, r)  # パディングを行う

        z_space = self.conv_lift(z)
        z_wave = torch.fft.fft2(z)
        if self.num_modes != (0, 0):
            z_low_pass_filter = torch.zeros_like(z_wave)

            if self.num_modes[0] != 0:
                z_low_pass_filter[:, :, :self.num_modes[0],:] = z_wave[:, :, :self.num_modes[0],:]
                z_low_pass_filter[:, :, -self.num_modes[0]:,:] = z_wave[:, :, -self.num_modes[0]:,:]
            if self.num_modes[1] != 0:
                z_low_pass_filter[:, :, :, :self.num_modes[1]] = z_wave[:, :, :, :self.num_modes[1]]
                z_low_pass_filter[:, :, :, -self.num_modes[1]:] = z_wave[:, :, :, -self.num_modes[1]:]
            z_wave = self.conv_wave(z_low_pass_filter)
        else:
            z_wave = self.conv_wave(z_wave)
        z_wave = torch.fft.ifft2(z_wave).real

        z = z_space + z_wave
        if self.pad_size > 0:
            z = z[:, :, self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]

        return z

    def pad(self, z:torch.Tensor, r:torch.Tensor)->torch.Tensor:
        """非周期性対策のために、特徴量ベクトルと位置座標を結合する前にパディングを行う。

        特徴量については端点の値でパディングを行い、位置座標についてはdxずつ減少(左側)もしくは増加(右側)するようにパディング。

        Args:
            z (torch.Tensor): 特徴量ベクトル。shapeは(batch_size, num_features, width, height)。
            r (torch.Tensor): 位置座標。shapeは(batch_size, 2, width, height)。

        Returns:
            torch.Tensor: padding済み特徴量。shapeは(batch_size, num_features + 2, width + 2 * pad_size, height + 2 * pad_size)。
        """
        if self.pad_size == 0:
            return torch.cat([z, r], dim=1)

        # zは境界値を複製して2次元パディング
        z_padded = F.pad(z, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), mode='replicate')

        # rは2チャネル([x, y])を想定
        rx = r[:, 0:1, :, :]
        ry = r[:, 1:2, :, :]

        dx = (rx[:, :, 1, 0] - rx[:, :, 0, 0]).view(-1, 1, 1, 1)
        dy = (ry[:, :, 0, 1] - ry[:, :, 0, 0]).view(-1, 1, 1, 1)

        # x座標: width方向を外挿し、height方向は境界値を複製
        left_w_steps = torch.arange(self.pad_size, 0, -1, device=r.device, dtype=r.dtype).view(1, 1, -1, 1)
        right_w_steps = torch.arange(1, self.pad_size + 1, device=r.device, dtype=r.dtype).view(1, 1, -1, 1)
        left_rx = rx[:, :, :1, :] - dx * left_w_steps
        right_rx = rx[:, :, -1:, :] + dx * right_w_steps
        rx_w = torch.cat([left_rx, rx, right_rx], dim=2)

        top_rx = rx_w[:, :, :, :1].repeat(1, 1, 1, self.pad_size)
        bottom_rx = rx_w[:, :, :, -1:].repeat(1, 1, 1, self.pad_size)
        rx_padded = torch.cat([top_rx, rx_w, bottom_rx], dim=3)

        # y座標: width方向は境界値を複製し、height方向を外挿
        left_ry = ry[:, :, :1, :].repeat(1, 1, self.pad_size, 1)
        right_ry = ry[:, :, -1:, :].repeat(1, 1, self.pad_size, 1)
        ry_w = torch.cat([left_ry, ry, right_ry], dim=2)

        top_h_steps = torch.arange(self.pad_size, 0, -1, device=r.device, dtype=r.dtype).view(1, 1, 1, -1)
        bottom_h_steps = torch.arange(1, self.pad_size + 1, device=r.device, dtype=r.dtype).view(1, 1, 1, -1)
        top_ry = ry_w[:, :, :, :1] - dy * top_h_steps
        bottom_ry = ry_w[:, :, :, -1:] + dy * bottom_h_steps
        ry_padded = torch.cat([top_ry, ry_w, bottom_ry], dim=3)

        r_padded = torch.cat([rx_padded, ry_padded], dim=1)

        return torch.cat([z_padded, r_padded], dim=1)


class FNO3D(nn.Module):
    """3次元座標に対するFNO処理

    Attributes:
        * pad_size (int): パディングサイズ。非周期性対策のために、特徴量ベクトルと位置座標を結合する前にパディングを行う。
        * conv_lift (nn.Conv3d): 3D畳み込み層。特徴量ベクトルを変換する。
        * conv_wave (nn.Conv3d): 3D畳み込み層。フーリエ空間での畳み込みを行う。
        * num_modes (int|tuple[int,int,int]): フーリエ空間での畳み込みに使用するモード数。0の場合は全てのモードを使用する。
    
    Methods:
        * pad(z:torch.Tensor, r:torch.Tensor)->torch.Tensor: 非周期性対策のために、特徴量ベクトルと位置座標を結合する前にパディングを行う。
    Note
        * 位置座標は等間隔かつr[:,0,0,:,:] < r[:,0,1,:,:] < ... < r[:,0,-1,:,:]ならびにr[:,1,:,0,:] < r[:,1,:,1,:] < ... < r[:,1,:,-1,:]ならびにr[:,2,:,:,0] < r[:,2,:,:,1] < ... < r[:,2,:,:,-1]であることを仮定している。
        * 各特徴量は同じ位置座標に対応していることを仮定している。
    """
    def __init__(self, in_features:int, out_features:int, pad_size:int, num_modes:tuple[int,int,int] = (0,0,0))->None:
        super().__init__()
        self.pad_size = pad_size
        self.conv_lift = nn.Conv3d(in_features + 3, out_features, kernel_size=1)
        self.conv_wave = nn.Conv3d(in_features + 3, out_features, kernel_size=1, dtype = torch.cfloat)

    def forward(self, z:torch.Tensor, r:torch.Tensor)->torch.Tensor:
        """forward処理

        Args:
            z (torch.Tensor): 特徴量ベクトル。shapeは(batch_size, num_features, width, height, depth)。
            r (torch.Tensor): 位置座標。shapeは(batch_size, 3, width, height, depth)。
        Returns:
            torch.Tensor: FNOの出力。shapeは(batch_size, num_features, width, height, depth)。
        """
        z = self.pad(z, r)  # パディングを行う

        z_space = self.conv_lift(z)

        z_wave = torch.fft.fftn(z, dim=(-3, -2, -1))
        if self.num_modes != (0, 0, 0):
            z_low_pass_filter = torch.zeros_like(z_wave)
            if self.num_modes[0] != 0:
                z_low_pass_filter[:, :, :self.num_modes[0], :, :] = z_wave[:, :, :self.num_modes[0], :, :]
                z_low_pass_filter[:, :, -self.num_modes[0]:, :, :] = z_wave[:, :, -self.num_modes[0]:, :, :]
            if self.num_modes[1] != 0:
                z_low_pass_filter[:, :, :, :self.num_modes[1], :] = z_wave[:, :, :, :self.num_modes[1], :]
                z_low_pass_filter[:, :, :, -self.num_modes[1]:, :] = z_wave[:, :, :, -self.num_modes[1]:, :]
            if self.num_modes[2] != 0:
                z_low_pass_filter[:, :, :, :, :self.num_modes[2]] = z_wave[:, :, :, :, :self.num_modes[2]]
                z_low_pass_filter[:, :, :, :, -self.num_modes[2]:] = z_wave[:, :, :, :, -self.num_modes[2]:]
            z_wave = self.conv_wave(z_low_pass_filter)
        else:
            z_wave = self.conv_wave(z_wave)
        z_wave = torch.fft.ifftn(z_wave, dim=(-3, -2, -1)).real

        z = z_space + z_wave
        if self.pad_size > 0:
            z = z[:, :, self.pad_size:-self.pad_size, self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]

        return z

    def pad(self, z:torch.Tensor, r:torch.Tensor)->torch.Tensor:
        """非周期性対策のために、特徴量ベクトルと位置座標を結合する前にパディングを行う。

        特徴量については端点の値でパディングを行い、位置座標についてはdxずつ減少(左側)もしくは増加(右側)するようにパディング。

        Args:
            z (torch.Tensor): 特徴量ベクトル。shapeは(batch_size, num_features, width, height, depth)。
            r (torch.Tensor): 位置座標。shapeは(batch_size, 3, width, height, depth)。

        Returns:
            torch.Tensor: padding済み特徴量。shapeは(batch_size, num_features + 3, width + 2 * pad_size, height + 2 * pad_size, depth + 2 * pad_size)。
        """
        if self.pad_size == 0:
            return torch.cat([z, r], dim=1)

        # zは境界値を複製して3次元パディング
        z_padded = F.pad(
            z,
            (self.pad_size, self.pad_size, self.pad_size, self.pad_size, self.pad_size, self.pad_size),
            mode='replicate'
        )

        # rは3チャネル([x, y, z])を想定
        rx = r[:, 0:1, :, :, :]
        ry = r[:, 1:2, :, :, :]
        rz = r[:, 2:3, :, :, :]

        dx = (rx[:, :, 1, 0, 0] - rx[:, :, 0, 0, 0]).view(-1, 1, 1, 1, 1)
        dy = (ry[:, :, 0, 1, 0] - ry[:, :, 0, 0, 0]).view(-1, 1, 1, 1, 1)
        dz = (rz[:, :, 0, 0, 1] - rz[:, :, 0, 0, 0]).view(-1, 1, 1, 1, 1)

        # x座標: width方向を外挿し、height/depth方向は境界値を複製
        left_w_steps = torch.arange(self.pad_size, 0, -1, device=r.device, dtype=r.dtype).view(1, 1, -1, 1, 1)
        right_w_steps = torch.arange(1, self.pad_size + 1, device=r.device, dtype=r.dtype).view(1, 1, -1, 1, 1)
        left_rx = rx[:, :, :1, :, :] - dx * left_w_steps
        right_rx = rx[:, :, -1:, :, :] + dx * right_w_steps
        rx_w = torch.cat([left_rx, rx, right_rx], dim=2)

        top_rx = rx_w[:, :, :, :1, :].repeat(1, 1, 1, self.pad_size, 1)
        bottom_rx = rx_w[:, :, :, -1:, :].repeat(1, 1, 1, self.pad_size, 1)
        rx_wh = torch.cat([top_rx, rx_w, bottom_rx], dim=3)

        front_rx = rx_wh[:, :, :, :, :1].repeat(1, 1, 1, 1, self.pad_size)
        back_rx = rx_wh[:, :, :, :, -1:].repeat(1, 1, 1, 1, self.pad_size)
        rx_padded = torch.cat([front_rx, rx_wh, back_rx], dim=4)

        # y座標: height方向を外挿し、width/depth方向は境界値を複製
        left_ry = ry[:, :, :1, :, :].repeat(1, 1, self.pad_size, 1, 1)
        right_ry = ry[:, :, -1:, :, :].repeat(1, 1, self.pad_size, 1, 1)
        ry_w = torch.cat([left_ry, ry, right_ry], dim=2)

        top_h_steps = torch.arange(self.pad_size, 0, -1, device=r.device, dtype=r.dtype).view(1, 1, 1, -1, 1)
        bottom_h_steps = torch.arange(1, self.pad_size + 1, device=r.device, dtype=r.dtype).view(1, 1, 1, -1, 1)
        top_ry = ry_w[:, :, :, :1, :] - dy * top_h_steps
        bottom_ry = ry_w[:, :, :, -1:, :] + dy * bottom_h_steps
        ry_wh = torch.cat([top_ry, ry_w, bottom_ry], dim=3)

        front_ry = ry_wh[:, :, :, :, :1].repeat(1, 1, 1, 1, self.pad_size)
        back_ry = ry_wh[:, :, :, :, -1:].repeat(1, 1, 1, 1, self.pad_size)
        ry_padded = torch.cat([front_ry, ry_wh, back_ry], dim=4)

        # z座標: depth方向を外挿し、width/height方向は境界値を複製
        left_rz = rz[:, :, :1, :, :].repeat(1, 1, self.pad_size, 1, 1)
        right_rz = rz[:, :, -1:, :, :].repeat(1, 1, self.pad_size, 1, 1)
        rz_w = torch.cat([left_rz, rz, right_rz], dim=2)

        top_rz = rz_w[:, :, :, :1, :].repeat(1, 1, 1, self.pad_size, 1)
        bottom_rz = rz_w[:, :, :, -1:, :].repeat(1, 1, 1, self.pad_size, 1)
        rz_wh = torch.cat([top_rz, rz_w, bottom_rz], dim=3)

        front_d_steps = torch.arange(self.pad_size, 0, -1, device=r.device, dtype=r.dtype).view(1, 1, 1, 1, -1)
        back_d_steps = torch.arange(1, self.pad_size + 1, device=r.device, dtype=r.dtype).view(1, 1, 1, 1, -1)
        front_rz = rz_wh[:, :, :, :, :1] - dz * front_d_steps
        back_rz = rz_wh[:, :, :, :, -1:] + dz * back_d_steps
        rz_padded = torch.cat([front_rz, rz_wh, back_rz], dim=4)

        r_padded = torch.cat([rx_padded, ry_padded, rz_padded], dim=1)

        return torch.cat([z_padded, r_padded], dim=1)