import numpy as np
import zhplot
import matplotlib.pyplot as plt
from scipy import signal
from scipy.special import erfc
import os

# 创建结果保存文件夹
results_folder = 'mobile_comm_figures'
os.makedirs(results_folder, exist_ok=True)


plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = [10, 6]  # 默认图片尺寸


class MultipathChannel:
    """多径衰落信道模型"""
    def __init__(self, delays, gains, doppler_shifts=None):
        self.delays = np.array(delays)
        self.gains = np.array(gains)
        self.doppler_shifts = doppler_shifts if doppler_shifts is not None else np.zeros(len(delays))
    
    def apply_channel(self, signal, fs, t_max):
        """应用多径信道"""
        t = np.linspace(0, t_max, len(signal))
        output = np.zeros_like(signal, dtype=complex)
        
        for delay, gain, fd in zip(self.delays, self.gains, self.doppler_shifts):
            # 时延
            delay_samples = int(delay * fs)
            delayed_signal = np.zeros_like(signal)
            
            # 确保时延不超过信号长度
            if delay_samples > 0 and delay_samples < len(signal):
                delayed_signal[delay_samples:] = signal[:-delay_samples]
            elif delay_samples == 0:
                delayed_signal = signal.copy()
            # 如果时延过大，该路径贡献为零
            
            # 多普勒频移和衰落
            doppler_effect = gain * np.exp(1j * 2 * np.pi * fd * t)
            output += delayed_signal * doppler_effect
        
        return output

class DigitalModulation:
    """数字调制类"""
    
    @staticmethod
    def qpsk_modulate(bits):
        """QPSK调制"""
        # 将比特分组为2比特符号
        symbols = []
        for i in range(0, len(bits), 2):
            if i+1 < len(bits):
                bit_pair = (bits[i], bits[i+1])
                if bit_pair == (0, 0):
                    symbols.append(1 + 1j)
                elif bit_pair == (0, 1):
                    symbols.append(-1 + 1j)
                elif bit_pair == (1, 0):
                    symbols.append(1 - 1j)
                else:  # (1, 1)
                    symbols.append(-1 - 1j)
        return np.array(symbols) / np.sqrt(2)
    
    @staticmethod
    def qpsk_demodulate(symbols, soft_decision=False):
        """QPSK解调"""
        if soft_decision:
            # 软判决解调 (更好的性能，但这里简化为硬判决)
            bits = []
            for symbol in symbols:
                real_bit = 0 if symbol.real > 0 else 1
                imag_bit = 0 if symbol.imag > 0 else 1
                bits.extend([real_bit, imag_bit])
        else:
            # 硬判决解调
            bits = []
            for symbol in symbols:
                real_bit = 0 if symbol.real > 0 else 1
                imag_bit = 0 if symbol.imag > 0 else 1
                bits.extend([real_bit, imag_bit])
        return np.array(bits)
    
    @staticmethod
    def qam16_modulate(bits):
        """16QAM调制"""
        constellation = np.array([
            -3-3j, -3-1j, -3+3j, -3+1j,
            -1-3j, -1-1j, -1+3j, -1+1j,
            3-3j, 3-1j, 3+3j, 3+1j,
            1-3j, 1-1j, 1+3j, 1+1j
        ]) / np.sqrt(10)
        
        symbols = []
        for i in range(0, len(bits), 4):
            if i+3 < len(bits):
                # 格雷码映射
                index = bits[i]*8 + bits[i+1]*4 + bits[i+2]*2 + bits[i+3]
                symbols.append(constellation[index])
        return np.array(symbols)
    
    @staticmethod
    def qam16_demodulate(symbols):
        """16QAM解调"""
        constellation = np.array([
            -3-3j, -3-1j, -3+3j, -3+1j,
            -1-3j, -1-1j, -1+3j, -1+1j,
            3-3j, 3-1j, 3+3j, 3+1j,
            1-3j, 1-1j, 1+3j, 1+1j
        ]) / np.sqrt(10)
        
        bits = []
        for symbol in symbols:
            # 最小距离判决
            distances = np.abs(symbol - constellation)
            min_index = np.argmin(distances)
            
            # 转换为4比特
            bit3 = (min_index >> 3) & 1
            bit2 = (min_index >> 2) & 1
            bit1 = (min_index >> 1) & 1
            bit0 = min_index & 1
            bits.extend([bit3, bit2, bit1, bit0])
        
        return np.array(bits)

def calculate_ber(tx_bits, rx_bits):
    """计算误码率"""
    min_len = min(len(tx_bits), len(rx_bits))
    errors = np.sum(tx_bits[:min_len] != rx_bits[:min_len])
    return errors / min_len

def add_awgn(signal, snr_db):
    """添加高斯白噪声"""
    signal_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db/10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
    return signal + noise

def frequency_offset_effect(signal, fs, freq_offset):
    """添加载波频偏"""
    t = np.arange(len(signal)) / fs
    return signal * np.exp(1j * 2 * np.pi * freq_offset * t)

# 仿真参数设置
num_bits = 10000
snr_range = np.arange(0, 16, 2)
fs = 1e6  # 采样频率
freq_offsets = np.array([0, 100, 500, 1000])  # 频偏值 (Hz)

print(f"仿真参数配置:")
print(f"- 比特数: {num_bits}")
print(f"- 信噪比范围: {snr_range[0]}dB 到 {snr_range[-1]}dB")
print(f"- 采样频率: {fs/1e6:.1f} MHz")
print(f"- 频偏测试值: {freq_offsets} Hz")
print("-" * 50)

# 1. 多径信道特性分析
print("正在进行多径信道特性分析...")
delays = [0, 1e-6, 2e-6]  # 时延：0, 1μs, 2μs
gains = [1, 0.5, 0.3]     # 功率增益
doppler_shifts = [0, 50, -30]  # 多普勒频移

channel = MultipathChannel(delays, gains, doppler_shifts)

# 生成测试脉冲响应
impulse = np.zeros(1000)
impulse[0] = 1
h_response = channel.apply_channel(impulse, fs, 1000/fs)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(np.arange(len(h_response))/fs*1e6, np.abs(h_response))
plt.title('多径信道冲激响应幅度')
plt.xlabel('时间 (μs)')
plt.ylabel('幅度')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(np.arange(len(h_response))/fs*1e6, np.angle(h_response))
plt.title('多径信道冲激响应相位')
plt.xlabel('时间 (μs)')
plt.ylabel('相位 (rad)')
plt.grid(True)

# 频域响应
H_freq = np.fft.fft(h_response, 1024)
freq_axis = np.fft.fftfreq(1024, 1/fs)

plt.subplot(2, 2, 3)
plt.plot(freq_axis[:512]/1e3, 20*np.log10(np.abs(H_freq[:512])))
plt.title('多径信道频域响应')
plt.xlabel('频率 (kHz)')
plt.ylabel('幅度 (dB)')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(freq_axis[:512]/1e3, np.angle(H_freq[:512]))
plt.title('多径信道相位响应')
plt.xlabel('频率 (kHz)')
plt.ylabel('相位 (rad)')
plt.grid(True)

plt.tight_layout()
plt.savefig(f'{results_folder}/multipath_channel_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. QPSK和16QAM星座图
print("正在生成调制星座图...")
mod = DigitalModulation()

# 生成随机比特
test_bits = np.random.randint(0, 2, 1000)

# QPSK调制
qpsk_symbols = mod.qpsk_modulate(test_bits)
# 16QAM调制
qam16_symbols = mod.qam16_modulate(test_bits)

plt.figure(figsize=(15, 5))

# 理想QPSK星座图
plt.subplot(1, 3, 1)
plt.scatter(qpsk_symbols.real, qpsk_symbols.imag, alpha=0.6, s=20)
plt.title('QPSK星座图')
plt.xlabel('同相分量')
plt.ylabel('正交分量')
plt.grid(True)
plt.axis('equal')

# 理想16QAM星座图
plt.subplot(1, 3, 2)
plt.scatter(qam16_symbols.real, qam16_symbols.imag, alpha=0.6, s=20)
plt.title('16QAM星座图')
plt.xlabel('同相分量')
plt.ylabel('正交分量')
plt.grid(True)
plt.axis('equal')

# 添加噪声后的16QAM星座图
noisy_qam16 = add_awgn(qam16_symbols, 10)  # 10dB SNR
plt.subplot(1, 3, 3)
plt.scatter(noisy_qam16.real, noisy_qam16.imag, alpha=0.6, s=20)
plt.title('16QAM星座图 (10dB SNR)')
plt.xlabel('同相分量')
plt.ylabel('正交分量')
plt.grid(True)
plt.axis('equal')

plt.tight_layout()
plt.savefig(f'{results_folder}/constellation_diagrams.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. BER性能分析
print("正在进行BER性能分析...")
qpsk_ber_sim = []
qam16_ber_sim = []
qpsk_ber_theory = []
qam16_ber_theory = []

for snr in snr_range:
    # 生成随机比特
    tx_bits = np.random.randint(0, 2, num_bits)
    
    # QPSK仿真
    qpsk_symbols = mod.qpsk_modulate(tx_bits)
    qpsk_noisy = add_awgn(qpsk_symbols, snr)
    qpsk_rx_bits = mod.qpsk_demodulate(qpsk_noisy)
    qpsk_ber_sim.append(calculate_ber(tx_bits, qpsk_rx_bits))
    
    # 16QAM仿真
    qam16_symbols = mod.qam16_modulate(tx_bits)
    qam16_noisy = add_awgn(qam16_symbols, snr)
    qam16_rx_bits = mod.qam16_demodulate(qam16_noisy)
    qam16_ber_sim.append(calculate_ber(tx_bits, qam16_rx_bits))
    
    # 理论BER
    eb_n0 = 10**(snr/10)
    qpsk_ber_theory.append(0.5 * erfc(np.sqrt(eb_n0)))
    qam16_ber_theory.append(3/8 * erfc(np.sqrt(eb_n0/5)))

plt.figure(figsize=(10, 6))
plt.semilogy(snr_range, qpsk_ber_sim, 'bo-', label='QPSK仿真', markersize=6)
plt.semilogy(snr_range, qpsk_ber_theory, 'b--', label='QPSK理论', linewidth=2)
plt.semilogy(snr_range, qam16_ber_sim, 'ro-', label='16QAM仿真', markersize=6)
plt.semilogy(snr_range, qam16_ber_theory, 'r--', label='16QAM理论', linewidth=2)
plt.xlabel('信噪比 (dB)')
plt.ylabel('误码率')
plt.title('QPSK和16QAM的BER性能比较')
plt.legend()
plt.grid(True)
plt.ylim([1e-5, 1])
plt.savefig(f'{results_folder}/ber_performance.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. 载波频偏影响分析
print("正在分析载波频偏影响...")
freq_offset_ber_qpsk = []
freq_offset_ber_qam16 = []

for freq_offset in freq_offsets:
    # 生成随机比特
    tx_bits = np.random.randint(0, 2, num_bits)
    
    # QPSK
    qpsk_symbols = mod.qpsk_modulate(tx_bits)
    qpsk_offset = frequency_offset_effect(qpsk_symbols, fs, freq_offset)
    qpsk_noisy = add_awgn(qpsk_offset, 10)  # 10dB SNR
    qpsk_rx_bits = mod.qpsk_demodulate(qpsk_noisy)
    freq_offset_ber_qpsk.append(calculate_ber(tx_bits, qpsk_rx_bits))
    
    # 16QAM
    qam16_symbols = mod.qam16_modulate(tx_bits)
    qam16_offset = frequency_offset_effect(qam16_symbols, fs, freq_offset)
    qam16_noisy = add_awgn(qam16_offset, 10)  # 10dB SNR
    qam16_rx_bits = mod.qam16_demodulate(qam16_noisy)
    freq_offset_ber_qam16.append(calculate_ber(tx_bits, qam16_rx_bits))

plt.figure(figsize=(10, 6))
plt.semilogy(freq_offsets, freq_offset_ber_qpsk, 'bo-', label='QPSK', markersize=8, linewidth=2)
plt.semilogy(freq_offsets, freq_offset_ber_qam16, 'ro-', label='16QAM', markersize=8, linewidth=2)
plt.xlabel('载波频偏 (Hz)')
plt.ylabel('误码率')
plt.title('载波频偏对调制性能的影响 (SNR=10dB)')
plt.legend()
plt.grid(True)
plt.savefig(f'{results_folder}/frequency_offset_effect.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. 多径信道下的性能分析
print("正在分析多径信道下的性能...")
multipath_ber_qpsk = []
multipath_ber_qam16 = []

for snr in snr_range:
    # 生成随机比特
    tx_bits = np.random.randint(0, 2, num_bits)
    
    # QPSK在多径信道
    qpsk_symbols = mod.qpsk_modulate(tx_bits)
    qpsk_multipath = channel.apply_channel(qpsk_symbols, fs, len(qpsk_symbols)/fs)
    qpsk_noisy = add_awgn(qpsk_multipath, snr)
    qpsk_rx_bits = mod.qpsk_demodulate(qpsk_noisy)
    multipath_ber_qpsk.append(calculate_ber(tx_bits, qpsk_rx_bits))
    
    # 16QAM在多径信道
    qam16_symbols = mod.qam16_modulate(tx_bits)
    qam16_multipath = channel.apply_channel(qam16_symbols, fs, len(qam16_symbols)/fs)
    qam16_noisy = add_awgn(qam16_multipath, snr)
    qam16_rx_bits = mod.qam16_demodulate(qam16_noisy)
    multipath_ber_qam16.append(calculate_ber(tx_bits, qam16_rx_bits))

plt.figure(figsize=(10, 6))
plt.semilogy(snr_range, qpsk_ber_sim, 'b--', label='QPSK (AWGN)', linewidth=2)
plt.semilogy(snr_range, multipath_ber_qpsk, 'bo-', label='QPSK (多径)', markersize=6)
plt.semilogy(snr_range, qam16_ber_sim, 'r--', label='16QAM (AWGN)', linewidth=2)
plt.semilogy(snr_range, multipath_ber_qam16, 'ro-', label='16QAM (多径)', markersize=6)
plt.xlabel('信噪比 (dB)')
plt.ylabel('误码率')
plt.title('多径信道与AWGN信道性能比较')
plt.legend()
plt.grid(True)
plt.ylim([1e-4, 1])
plt.savefig(f'{results_folder}/multipath_performance.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. OFDM系统简单实现
print("正在进行OFDM系统仿真...")

class OFDMSystem:
    def __init__(self, N=64, cp_len=16):
        self.N = N  # 子载波数
        self.cp_len = cp_len  # 循环前缀长度
    
    def modulate(self, data):
        """OFDM调制"""
        # IFFT
        ifft_out = np.fft.ifft(data, self.N)
        # 添加循环前缀
        cp = ifft_out[-self.cp_len:]
        return np.concatenate([cp, ifft_out])
    
    def demodulate(self, signal):
        """OFDM解调"""
        # 移除循环前缀
        signal_no_cp = signal[self.cp_len:]
        # FFT
        return np.fft.fft(signal_no_cp, self.N)

ofdm = OFDMSystem()

# 生成OFDM符号数据
ofdm_data = np.random.randn(64) + 1j * np.random.randn(64)
ofdm_signal = ofdm.modulate(ofdm_data)

# 分析不同频偏下的OFDM性能
freq_offsets_ofdm = np.linspace(0, 0.5, 21)  # 归一化频偏
ofdm_mse = []

for freq_offset in freq_offsets_ofdm:
    # 添加频偏
    t = np.arange(len(ofdm_signal))
    offset_signal = ofdm_signal * np.exp(1j * 2 * np.pi * freq_offset * t / len(ofdm_signal))
    
    # 添加噪声
    noisy_signal = add_awgn(offset_signal, 20)
    
    # 解调
    rx_data = ofdm.demodulate(noisy_signal)
    
    # 计算MSE
    mse = np.mean(np.abs(ofdm_data - rx_data)**2)
    ofdm_mse.append(mse)

plt.figure(figsize=(10, 6))
plt.plot(freq_offsets_ofdm, 10*np.log10(ofdm_mse), 'bo-', markersize=6, linewidth=2)
plt.xlabel('归一化载波频偏')
plt.ylabel('MSE (dB)')
plt.title('OFDM系统载波频偏敏感性分析')
plt.grid(True)
plt.savefig(f'{results_folder}/ofdm_frequency_sensitivity.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"仿真完成！所有结果图片已保存到 '{results_folder}' 文件夹中。")
print("=" * 60)
print("生成的图片包括:")
print("1. multipath_channel_analysis.png - 多径信道特性分析")
print("2. constellation_diagrams.png - 调制星座图")
print("3. ber_performance.png - BER性能比较") 
print("4. frequency_offset_effect.png - 载波频偏影响")
print("5. multipath_performance.png - 多径信道性能分析")
print("6. ofdm_frequency_sensitivity.png - OFDM频偏敏感性")
print("=" * 60)

# 保存仿真参数和结果摘要
summary_text = f"""移动通信系统仿真结果摘要
================================

仿真参数:
- 比特数: {num_bits}
- 信噪比范围: {snr_range[0]}dB 到 {snr_range[-1]}dB
- 采样频率: {fs/1e6:.1f} MHz
- 多径时延: {delays}
- 多径增益: {gains}
- 多普勒频移: {doppler_shifts} Hz

关键结果:
- QPSK在15dB SNR下的BER: {qpsk_ber_sim[-1]:.2e}
- 16QAM在15dB SNR下的BER: {qam16_ber_sim[-1]:.2e}
- 1kHz频偏对QPSK的影响: BER增加到 {freq_offset_ber_qpsk[-1]:.2e}
- 1kHz频偏对16QAM的影响: BER增加到 {freq_offset_ber_qam16[-1]:.2e}

图片文件:
1. multipath_channel_analysis.png - 多径信道特性分析
2. constellation_diagrams.png - 调制星座图
3. ber_performance.png - BER性能比较
4. frequency_offset_effect.png - 载波频偏影响
5. multipath_performance.png - 多径信道性能分析
6. ofdm_frequency_sensitivity.png - OFDM频偏敏感性

生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open(f'{results_folder}/simulation_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary_text)

print(f"仿真摘要已保存到 '{results_folder}/simulation_summary.txt'")
print("所有文件均以300dpi高分辨率保存，适合论文和报告使用。")