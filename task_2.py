import numpy as np
import zhplot
import matplotlib.pyplot as plt
from scipy import signal
from scipy.special import erfc
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建结果保存文件夹
results_dir = 'simulation_results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

class MultipathChannel:
    """多径衰落信道模型"""
    
    def __init__(self, delays, gains, doppler_freq, fs):
        self.delays = np.array(delays)  # 时延 (秒)
        self.gains = np.array(gains)    # 增益 (dB)
        self.doppler_freq = doppler_freq  # 多普勒频移 (Hz)
        self.fs = fs  # 采样频率 (Hz)
        
    def generate_response(self, length):
        """生成信道冲激响应"""
        # 将时延转换为采样点
        delay_samples = np.round(self.delays * self.fs).astype(int)
        max_delay = np.max(delay_samples)
        
        # 初始化信道响应
        h = np.zeros(length + max_delay, dtype=complex)
        
        # 添加每条路径
        for i, (delay, gain_db) in enumerate(zip(delay_samples, self.gains)):
            gain_linear = 10**(gain_db/20)
            
            # 生成瑞利衰落系数
            if i == 0:  # 第一条路径可能有直射分量(莱斯分布)
                rice_factor = 3  # K因子 (dB)
                k_linear = 10**(rice_factor/10)
                rayleigh_var = gain_linear**2 / (2*(k_linear + 1))
                los_amp = gain_linear * np.sqrt(k_linear/(k_linear + 1))
                
                # 莱斯衰落
                real_part = np.random.normal(los_amp, np.sqrt(rayleigh_var), length)
                imag_part = np.random.normal(0, np.sqrt(rayleigh_var), length)
            else:
                # 瑞利衰落
                rayleigh_var = gain_linear**2 / 2
                real_part = np.random.normal(0, np.sqrt(rayleigh_var), length)
                imag_part = np.random.normal(0, np.sqrt(rayleigh_var), length)
            
            fade_coeff = real_part + 1j * imag_part
            
            # 添加多普勒频移
            t = np.arange(length) / self.fs
            doppler_shift = np.exp(1j * 2 * np.pi * self.doppler_freq * t)
            fade_coeff *= doppler_shift
            
            # 添加到信道响应中
            h[delay:delay+length] += fade_coeff
            
        return h[:length]

class DigitalModulation:
    """数字调制类"""
    
    @staticmethod
    def qpsk_modulate(bits):
        """QPSK调制"""
        # 重新整形为符号对
        bits_reshaped = bits.reshape(-1, 2)
        symbols = np.zeros(len(bits_reshaped), dtype=complex)
        
        # QPSK星座映射
        constellation = {
            (0, 0): 1+1j,
            (0, 1): -1+1j,
            (1, 0): 1-1j,
            (1, 1): -1-1j
        }
        
        for i, bit_pair in enumerate(bits_reshaped):
            symbols[i] = constellation[tuple(bit_pair)]
        
        return symbols / np.sqrt(2)  # 归一化能量
    
    @staticmethod
    def qpsk_demodulate(symbols):
        """QPSK解调"""
        bits = np.zeros(len(symbols) * 2, dtype=int)
        
        for i, symbol in enumerate(symbols):
            real_bit = 0 if symbol.real > 0 else 1
            imag_bit = 0 if symbol.imag > 0 else 1
            bits[2*i] = real_bit
            bits[2*i+1] = imag_bit
            
        return bits
    
    @staticmethod
    def qam16_modulate(bits):
        """16QAM调制"""
        bits_reshaped = bits.reshape(-1, 4)
        symbols = np.zeros(len(bits_reshaped), dtype=complex)
        
        # 16QAM星座映射
        constellation = {}
        levels = [-3, -1, 1, 3]
        bit_patterns = [(i//2, i%2, j//2, j%2) for i in range(4) for j in range(4)]
        
        idx = 0
        for i in levels:
            for j in levels:
                constellation[bit_patterns[idx]] = i + 1j*j
                idx += 1
        
        for i, bit_quad in enumerate(bits_reshaped):
            symbols[i] = constellation[tuple(bit_quad)]
        
        return symbols / np.sqrt(10)  # 归一化能量
    
    @staticmethod
    def qam16_demodulate(symbols):
        """16QAM解调"""
        bits = np.zeros(len(symbols) * 4, dtype=int)
        
        for i, symbol in enumerate(symbols):
            # 硬判决解调
            real_part = symbol.real * np.sqrt(10)
            imag_part = symbol.imag * np.sqrt(10)
            
            # 判决边界
            if real_part > 2:
                real_bits = [1, 1]
            elif real_part > 0:
                real_bits = [1, 0]
            elif real_part > -2:
                real_bits = [0, 0]
            else:
                real_bits = [0, 1]
                
            if imag_part > 2:
                imag_bits = [1, 1]
            elif imag_part > 0:
                imag_bits = [1, 0]
            elif imag_part > -2:
                imag_bits = [0, 0]
            else:
                imag_bits = [0, 1]
            
            bits[4*i:4*i+2] = real_bits
            bits[4*i+2:4*i+4] = imag_bits
            
        return bits

def add_awgn(signal, snr_db):
    """添加AWGN噪声"""
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
    return signal + noise

def add_frequency_offset(signal, freq_offset, fs):
    """添加载波频偏"""
    t = np.arange(len(signal)) / fs
    return signal * np.exp(1j * 2 * np.pi * freq_offset * t)

def theoretical_ber_qpsk(snr_db):
    """QPSK理论误码率"""
    snr_linear = 10**(snr_db/10)
    return 0.5 * erfc(np.sqrt(snr_linear))

def theoretical_ber_qam16(snr_db):
    """16QAM理论误码率"""
    snr_linear = 10**(snr_db/10)
    return 0.375 * erfc(np.sqrt(0.4 * snr_linear))

def simulate_ber_performance():
    """BER性能仿真"""
    print("开始BER性能仿真...")
    
    # 仿真参数
    snr_range = np.arange(0, 16, 2)  # SNR范围 (dB)
    num_bits = 100000  # 比特数
    
    # 存储结果
    ber_qpsk_sim = []
    ber_qam16_sim = []
    ber_qpsk_theory = []
    ber_qam16_theory = []
    
    for snr in snr_range:
        print(f"  仿真SNR = {snr} dB...")
        
        # 生成随机比特
        bits_qpsk = np.random.randint(0, 2, num_bits)
        bits_qam16 = np.random.randint(0, 2, num_bits)
        
        # QPSK仿真
        qpsk_symbols = DigitalModulation.qpsk_modulate(bits_qpsk)
        qpsk_noisy = add_awgn(qpsk_symbols, snr)
        qpsk_demod = DigitalModulation.qpsk_demodulate(qpsk_noisy)
        ber_qpsk_sim.append(np.mean(bits_qpsk != qpsk_demod))
        
        # 16QAM仿真
        qam16_symbols = DigitalModulation.qam16_modulate(bits_qam16)
        qam16_noisy = add_awgn(qam16_symbols, snr)
        qam16_demod = DigitalModulation.qam16_demodulate(qam16_noisy)
        ber_qam16_sim.append(np.mean(bits_qam16 != qam16_demod))
        
        # 理论值
        ber_qpsk_theory.append(theoretical_ber_qpsk(snr))
        ber_qam16_theory.append(theoretical_ber_qam16(snr))
    
    # 绘制BER曲线
    plt.figure(figsize=(10, 8))
    plt.semilogy(snr_range, ber_qpsk_sim, 'ro-', label='QPSK仿真', markersize=8)
    plt.semilogy(snr_range, ber_qpsk_theory, 'r--', label='QPSK理论', linewidth=2)
    plt.semilogy(snr_range, ber_qam16_sim, 'bs-', label='16QAM仿真', markersize=8)
    plt.semilogy(snr_range, ber_qam16_theory, 'b--', label='16QAM理论', linewidth=2)
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('信噪比 (dB)', fontsize=14)
    plt.ylabel('误码率 (BER)', fontsize=14)
    plt.title('QPSK和16QAM调制系统BER性能对比', fontsize=16)
    plt.legend(fontsize=12)
    plt.ylim(1e-5, 1)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/ber_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def simulate_multipath_channel():
    """多径信道仿真"""
    print("开始多径信道仿真...")
    
    # 信道参数
    delays = [0, 1e-6, 2e-6, 3e-6]  # 时延 (秒)
    gains = [0, -3, -6, -9]  # 增益 (dB)
    doppler_freq = 100  # 多普勒频移 (Hz)
    fs = 1e6  # 采样频率 (Hz)
    
    # 创建信道
    channel = MultipathChannel(delays, gains, doppler_freq, fs)
    
    # 生成信道响应
    channel_length = 1000
    h = channel.generate_response(channel_length)
    
    # 绘制信道幅度响应
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    t = np.arange(len(h)) / fs * 1e6  # 转换为微秒
    plt.plot(t, 20*np.log10(np.abs(h)), 'b-', linewidth=1.5)
    plt.xlabel('时间 (μs)', fontsize=12)
    plt.ylabel('幅度 (dB)', fontsize=12)
    plt.title('多径信道时域响应', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 计算并绘制功率延迟分布
    plt.subplot(3, 1, 2)
    power_delay = np.zeros(len(delays))
    for i, (delay, gain) in enumerate(zip(delays, gains)):
        power_delay[i] = 10**(gain/10)
    
    delay_us = np.array(delays) * 1e6
    plt.stem(delay_us, power_delay, basefmt=' ')
    plt.xlabel('时延 (μs)', fontsize=12)
    plt.ylabel('功率 (线性)', fontsize=12)
    plt.title('功率延迟分布', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 频率响应
    plt.subplot(3, 1, 3)
    H = np.fft.fft(h, 2048)
    freq = np.fft.fftfreq(2048, 1/fs) / 1e3  # 转换为kHz
    plt.plot(freq[:1024], 20*np.log10(np.abs(H[:1024])), 'r-', linewidth=1.5)
    plt.xlabel('频率 (kHz)', fontsize=12)
    plt.ylabel('幅度 (dB)', fontsize=12)
    plt.title('多径信道频率响应', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/multipath_channel.png', dpi=300, bbox_inches='tight')
    plt.show()

def simulate_frequency_offset_effect():
    """频偏影响仿真"""
    print("开始频偏影响仿真...")
    
    # 仿真参数
    freq_offsets = np.linspace(0, 0.1, 11)  # 归一化频偏 (相对于符号速率)
    snr_db = 10  # 固定SNR
    num_bits = 50000
    
    ber_qpsk_freq = []
    ber_qam16_freq = []
    
    for freq_offset in freq_offsets:
        print(f"  仿真频偏 = {freq_offset:.2f} * Rs...")
        
        # 生成随机比特
        bits_qpsk = np.random.randint(0, 2, num_bits)
        bits_qam16 = np.random.randint(0, 2, num_bits)
        
        # QPSK with frequency offset
        qpsk_symbols = DigitalModulation.qpsk_modulate(bits_qpsk)
        qpsk_offset = add_frequency_offset(qpsk_symbols, freq_offset, 1.0)  # 归一化采样率
        qpsk_noisy = add_awgn(qpsk_offset, snr_db)
        qpsk_demod = DigitalModulation.qpsk_demodulate(qpsk_noisy)
        ber_qpsk_freq.append(np.mean(bits_qpsk != qpsk_demod))
        
        # 16QAM with frequency offset
        qam16_symbols = DigitalModulation.qam16_modulate(bits_qam16)
        qam16_offset = add_frequency_offset(qam16_symbols, freq_offset, 1.0)
        qam16_noisy = add_awgn(qam16_offset, snr_db)
        qam16_demod = DigitalModulation.qam16_demodulate(qam16_noisy)
        ber_qam16_freq.append(np.mean(bits_qam16 != qam16_demod))
    
    # 绘制频偏影响
    plt.figure(figsize=(10, 8))
    plt.semilogy(freq_offsets, ber_qpsk_freq, 'ro-', label='QPSK', markersize=8, linewidth=2)
    plt.semilogy(freq_offsets, ber_qam16_freq, 'bs-', label='16QAM', markersize=8, linewidth=2)
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('归一化频偏 (Δf/Rs)', fontsize=14)
    plt.ylabel('误码率 (BER)', fontsize=14)
    plt.title(f'载波频偏对误码率的影响 (SNR = {snr_db} dB)', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/frequency_offset_effect.png', dpi=300, bbox_inches='tight')
    plt.show()

def simulate_constellation_diagrams():
    """星座图仿真"""
    print("生成星座图...")
    
    num_symbols = 1000
    snr_values = [20, 10, 5]  # 不同SNR下的星座图
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, snr in enumerate(snr_values):
        # QPSK星座图
        bits_qpsk = np.random.randint(0, 2, num_symbols * 2)
        qpsk_symbols = DigitalModulation.qpsk_modulate(bits_qpsk)
        qpsk_noisy = add_awgn(qpsk_symbols, snr)
        
        axes[0, i].scatter(qpsk_noisy.real, qpsk_noisy.imag, alpha=0.6, s=20)
        axes[0, i].set_title(f'QPSK (SNR = {snr} dB)', fontsize=12)
        axes[0, i].set_xlabel('同相分量', fontsize=10)
        axes[0, i].set_ylabel('正交分量', fontsize=10)
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].axis('equal')
        
        # 16QAM星座图
        bits_qam16 = np.random.randint(0, 2, num_symbols * 4)
        qam16_symbols = DigitalModulation.qam16_modulate(bits_qam16)
        qam16_noisy = add_awgn(qam16_symbols, snr)
        
        axes[1, i].scatter(qam16_noisy.real, qam16_noisy.imag, alpha=0.6, s=20)
        axes[1, i].set_title(f'16QAM (SNR = {snr} dB)', fontsize=12)
        axes[1, i].set_xlabel('同相分量', fontsize=10)
        axes[1, i].set_ylabel('正交分量', fontsize=10)
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].axis('equal')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/constellation_diagrams.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("="*60)
    print("移动通信信道建模与数字调制系统仿真")
    print("="*60)
    
    # 运行各种仿真
    simulate_ber_performance()
    simulate_multipath_channel()
    simulate_frequency_offset_effect()
    simulate_constellation_diagrams()
    
    print("\n仿真完成！所有结果已保存到 'simulation_results' 文件夹")
    print("生成的图像文件:")
    print("- ber_performance.png: BER性能对比")
    print("- multipath_channel.png: 多径信道特性")
    print("- frequency_offset_effect.png: 频偏影响")
    print("- constellation_diagrams.png: 星座图")

if __name__ == "__main__":
    main()