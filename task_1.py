import numpy as np
import zhplot
import matplotlib.pyplot as plt
from scipy.special import jv  # 贝塞尔函数
from scipy.fft import fft, fftshift, fftfreq
import matplotlib.font_manager as fm
import os
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class MultipathRayleighChannel:
    """多径瑞利衰落信道模型类"""
    
    def __init__(self, num_paths=6, max_delay=5e-6, fc=2e9, v_mobile=30, 
                 fs=1e6, duration=0.01):
        """
        初始化信道参数
        
        参数:
        num_paths: 多径数量
        max_delay: 最大时延扩展 (秒)
        fc: 载波频率 (Hz)
        v_mobile: 移动速度 (m/s)
        fs: 采样频率 (Hz)
        duration: 仿真时长 (秒)
        """
        self.num_paths = num_paths
        self.max_delay = max_delay
        self.fc = fc
        self.v_mobile = v_mobile
        self.fs = fs
        self.duration = duration
        
        # 计算最大多普勒频移
        self.fd_max = self.v_mobile * self.fc / 3e8
        
        # 时间和频率网格
        self.t = np.arange(0, duration, 1/fs)
        self.N = len(self.t)
        
        # 生成路径参数
        self._generate_path_parameters()
        
    def _generate_path_parameters(self):
        """生成多径参数"""
        # 设置随机种子以确保可重复性
        np.random.seed(42)
        
        # 路径时延 (指数分布)
        tau_rms = self.max_delay / 3  # 均方根时延扩展
        self.delays = np.sort(np.random.exponential(tau_rms, self.num_paths))
        self.delays[0] = 0  # 第一径时延为0
        
        # 确保最大时延不超过设定值
        self.delays = self.delays * self.max_delay / np.max(self.delays)
        
        # 路径功率 (指数衰减)
        self.path_powers = np.exp(-self.delays / tau_rms)
        self.path_powers = self.path_powers / np.sum(self.path_powers)  # 归一化
        
        # 路径到达角 (均匀分布)
        self.arrival_angles = np.random.uniform(0, 2*np.pi, self.num_paths)
        
    def generate_rayleigh_fading(self, path_idx):
        """生成单径的瑞利衰落"""
        # 多普勒频移
        fd = self.fd_max * np.cos(self.arrival_angles[path_idx])
        
        # 生成复高斯噪声
        variance = self.path_powers[path_idx] / 2
        h_i = np.random.normal(0, np.sqrt(variance), self.N)
        h_q = np.random.normal(0, np.sqrt(variance), self.N)
        
        # 应用多普勒频移
        doppler_phase = 2 * np.pi * fd * self.t
        h_complex = (h_i + 1j * h_q) * np.exp(1j * doppler_phase)
        
        return h_complex
    
    def generate_channel_response(self):
        """生成完整的信道冲激响应"""
        # 时延样本数
        delay_samples = np.round(self.delays * self.fs).astype(int)
        max_delay_samples = np.max(delay_samples)
        
        # 初始化信道矩阵
        h_matrix = np.zeros((self.N, max_delay_samples + 1), dtype=complex)
        
        # 为每个路径生成衰落
        self.path_fadimg = []
        for i in range(self.num_paths):
            fading = self.generate_rayleigh_fading(i)
            self.path_fadimg.append(fading)
            h_matrix[:, delay_samples[i]] += fading
            
        return h_matrix
    
    def compute_doppler_psd(self):
        """计算多普勒功率谱密度"""
        # 理论Jakes谱
        f = np.linspace(-2*self.fd_max, 2*self.fd_max, 1000)
        jakes_psd = np.zeros_like(f)
        
        # 计算Jakes功率谱密度
        mask = np.abs(f) <= self.fd_max
        jakes_psd[mask] = 1 / (np.pi * self.fd_max * 
                              np.sqrt(1 - (f[mask]/self.fd_max)**2))
        
        # 仿真功率谱密度
        sim_psd_total = np.zeros(len(f))
        
        for i in range(self.num_paths):
            # 对每径衰落进行FFT
            H = fft(self.path_fadimg[i])
            freq = fftfreq(self.N, 1/self.fs)
            psd = np.abs(H)**2 / (self.N * self.fs)
            
            # 插值到目标频率网格
            psd_interp = np.interp(f, freq, fftshift(psd))
            sim_psd_total += self.path_powers[i] * psd_interp
            
        return f, jakes_psd, sim_psd_total
    
    def plot_comprehensive_results(self, h_matrix, f, theoretical_psd, simulated_psd, save_path=None):
        """绘制综合结果图（三个子图）"""
        fig = plt.figure(figsize=(16, 12))
        
        # 子图1: 信道冲激响应幅度
        ax1 = plt.subplot(3, 1, 1)
        time_ms = self.t * 1000  # 转换为毫秒
        N_display = min(300, self.N)
        
        plt.plot(time_ms[:N_display], 
                20*np.log10(np.abs(h_matrix[:N_display, 0]) + 1e-10),
                'b-', linewidth=1.8, label='第一径衰落')
        plt.xlabel('时间 (ms)', fontsize=14)
        plt.ylabel('幅度 (dB)', fontsize=14)
        plt.title(f'信道冲激响应幅度 (fc={self.fc/1e9:.1f}GHz, v={self.v_mobile}m/s)', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.7)
        plt.legend(fontsize=12)
        plt.xlim([0, time_ms[N_display-1]])
        
        # 子图2: 功率时延谱
        ax2 = plt.subplot(3, 1, 2)
        delay_us = self.delays * 1e6  # 转换为微秒
        markerline, stemlines, baseline = plt.stem(delay_us, 10*np.log10(self.path_powers + 1e-10), 
                                                  basefmt=' ', linefmt='r-', markerfmt='ro')
        markerline.set_markersize(10)
        stemlines.set_linewidth(2.5)
        
        plt.xlabel('时延 (μs)', fontsize=14)
        plt.ylabel('相对功率 (dB)', fontsize=14)
        plt.title(f'功率时延谱 ({self.num_paths}径, 最大时延={self.max_delay*1e6:.1f}μs)', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.7)
        plt.xlim([0, self.max_delay*1e6*1.1])
        
        # 子图3: 多普勒功率谱密度
        ax3 = plt.subplot(3, 1, 3)
        plt.plot(f, 10*np.log10(theoretical_psd + 1e-10), 'r-', 
                linewidth=3, label='理论Jakes谱', alpha=0.9)
        plt.plot(f, 10*np.log10(simulated_psd + 1e-10), 'b--', 
                linewidth=2.5, label='仿真谱', alpha=0.8)
        
        plt.xlabel('频率偏移 (Hz)', fontsize=14)
        plt.ylabel('功率谱密度 (dB)', fontsize=14)
        plt.title(f'多普勒功率谱密度 (fd_max = {self.fd_max:.2f} Hz)', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.7)
        plt.xlim([-2*self.fd_max, 2*self.fd_max])
        
        # 添加参数文本框
        textstr = f'载波频率: {self.fc/1e9:.1f} GHz\n移动速度: {self.v_mobile} m/s\n多径数: {self.num_paths}\n时延扩展: {self.max_delay*1e6:.1f} μs'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        plt.text(0.02, 0.98, textstr, transform=ax3.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout(pad=3.0)
        
        if save_path:
            plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
            print(f"综合仿真结果图已保存: {save_path}")
        
        plt.close()
    
    def analyze_channel_statistics(self):
        """分析信道统计特性"""
        print("=== 信道参数 ===")
        print(f"载波频率: {self.fc/1e9:.1f} GHz")
        print(f"移动速度: {self.v_mobile} m/s")
        print(f"最大多普勒频移: {self.fd_max:.2f} Hz")
        print(f"多径数量: {self.num_paths}")
        print(f"最大时延扩展: {self.max_delay*1e6:.2f} μs")
        
        print("\n=== 各径参数 ===")
        for i in range(self.num_paths):
            print(f"径 {i+1}: 时延 = {self.delays[i]*1e6:.2f} μs, "
                  f"功率 = {10*np.log10(self.path_powers[i]):.1f} dB")
        
        # 计算均方根时延扩展
        mean_delay = np.sum(self.path_powers * self.delays)
        mean_delay_sq = np.sum(self.path_powers * self.delays**2)
        rms_delay = np.sqrt(mean_delay_sq - mean_delay**2)
        
        print(f"\n平均时延: {mean_delay*1e6:.2f} μs")
        print(f"均方根时延扩展: {rms_delay*1e6:.2f} μs")
        print(f"相干时间: {1/(2*self.fd_max)*1000:.2f} ms")

def create_output_directory():
    """创建输出文件夹"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"channel_simulation_results_{timestamp}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出文件夹: {output_dir}")
    
    return output_dir

def run_simulation_with_parameters(params, output_dir, case_name):
    """运行指定参数的仿真"""
    print(f"\n{'='*60}")
    print(f"运行仿真案例: {case_name}")
    print(f"{'='*60}")
    
    # 创建信道对象
    channel = MultipathRayleighChannel(**params)
    
    # 分析信道参数
    channel.analyze_channel_statistics()
    
    # 生成信道响应
    print("正在生成信道冲激响应...")
    h_matrix = channel.generate_channel_response()
    
    # 计算多普勒功率谱密度
    print("正在计算多普勒功率谱密度...")
    f, theoretical_psd, simulated_psd = channel.compute_doppler_psd()
    
    # 定义保存路径
    save_path = os.path.join(output_dir, f"{case_name}_综合仿真结果.png")
    
    # 绘制并保存综合结果
    print("正在生成和保存综合图表...")
    channel.plot_comprehensive_results(h_matrix, f, theoretical_psd, simulated_psd, save_path=save_path)
    
    return channel

def main():
    """主仿真程序"""
    print("多径瑞利衰落信道建模与仿真")
    print("="*60)
    
    # 创建输出文件夹
    output_dir = create_output_directory()
    
    # 定义三组仿真参数
    simulation_cases = [
        {
            'name': '案例1_城市环境_中速移动',
            'params': {
                'num_paths': 6,
                'max_delay': 5e-6,     # 5μs
                'fc': 2.4e9,           # 2.4GHz
                'v_mobile': 30,        # 30m/s (108km/h)
                'fs': 1e6,
                'duration': 0.02
            }
        },
        {
            'name': '案例2_高速公路_高速移动',
            'params': {
                'num_paths': 4,
                'max_delay': 8e-6,     # 8μs
                'fc': 1.8e9,           # 1.8GHz
                'v_mobile': 50,        # 50m/s (180km/h)
                'fs': 1e6,
                'duration': 0.02
            }
        },
        {
            'name': '案例3_5G毫米波_中速移动',
            'params': {
                'num_paths': 8,
                'max_delay': 3e-6,     # 3μs
                'fc': 28e9,            # 28GHz (5G毫米波)
                'v_mobile': 25,        # 25m/s (90km/h)
                'fs': 1e6,
                'duration': 0.02
            }
        }
    ]
    
    # 运行所有仿真案例
    channels = []
    for case in simulation_cases:
        channel = run_simulation_with_parameters(
            case['params'], 
            output_dir, 
            case['name']
        )
        channels.append(channel)
    
    print(f"\n{'='*60}")
    print("所有仿真案例完成！")
    print(f"结果已保存到文件夹: {output_dir}")
    print(f"共生成 3 个高清PNG图表文件 (600 DPI)")
    print("="*60)
    
    # 生成仿真总结报告
    summary_path = os.path.join(output_dir, "仿真参数总结.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("多径瑞利衰落信道仿真参数总结\n")
        f.write("="*50 + "\n\n")
        
        for i, case in enumerate(simulation_cases):
            channel = channels[i]
            f.write(f"【{case['name']}】\n")
            f.write(f"载波频率: {channel.fc/1e9:.1f} GHz\n")
            f.write(f"移动速度: {channel.v_mobile} m/s ({channel.v_mobile*3.6:.0f} km/h)\n")
            f.write(f"最大多普勒频移: {channel.fd_max:.2f} Hz\n")
            f.write(f"多径数量: {channel.num_paths}\n")
            f.write(f"最大时延扩展: {channel.max_delay*1e6:.2f} μs\n")
            f.write(f"相干时间: {1/(2*channel.fd_max)*1000:.2f} ms\n")
            
            # 计算均方根时延扩展
            mean_delay = np.sum(channel.path_powers * channel.delays)
            mean_delay_sq = np.sum(channel.path_powers * channel.delays**2)
            rms_delay = np.sqrt(mean_delay_sq - mean_delay**2)
            f.write(f"均方根时延扩展: {rms_delay*1e6:.2f} μs\n")
            f.write("-"*40 + "\n\n")
    
    print(f"仿真参数总结已保存: {summary_path}")

if __name__ == "__main__":
    main()