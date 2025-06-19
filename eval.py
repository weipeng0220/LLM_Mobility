import numpy as np
import torch
import pickle
import os
import sys
import time
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

# 添加项目路径
sys.path.append(os.path.abspath('.'))

from train.llama_flash_attn_replace import *
from utils.args import *
from model.Decoder_CausalLLM import *
from model.VAE_CausalLLM import *
from utils.data_collator import *
from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig
from peft import PeftModel


class MobilityEvaluator:
    def __init__(self, model_path: str, params, device: str = 'cuda'):
        """
        初始化评估器
        
        Args:
            model_path: 模型保存路径
            params: 模型参数
            device: 设备类型
        """
        self.model_path = model_path
        self.params = params
        self.device = device
        
        # 加载模型和tokenizer
        self._load_model()
        
    def _load_model(self):
        """加载训练好的模型"""
        print("Loading model...")
        
        # 替换注意力机制
        replace_llama_attn(inference=True)
        
        # 加载配置
        config = AutoConfig.from_pretrained(
            self.model_path,
            z_latent_size=self.params.z_latent_size
        )
        
        # 加载解码器模型
        self.model_decoder = DecoderLlamaForCausalLM.from_pretrained(
            self.model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
            device_map=self.device
        )
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            model_max_length=self.params.traj_length,
            padding_side="right",
            use_fast=True,
        )
        self.model_decoder.get_tokenizer(self.tokenizer)
        
        # 加载VAE模型
        self.model = VAE_CausalLLM(self.params, self.model_decoder)
        
        # 加载训练好的权重
        if os.path.exists(os.path.join(self.model_path, 'model.pth')):
            state_dict = torch.load(os.path.join(self.model_path, 'model.pth'), 
                                  map_location=self.device)
            self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
        
    def evaluate_reconstruction_loss(self, test_data: List[Tuple]) -> Dict:
        """
        评估重建损失
        
        Args:
            test_data: 测试数据列表
            
        Returns:
            重建损失统计信息
        """
        print("Evaluating reconstruction loss...")
        
        total_loss = 0.0
        total_samples = 0
        losses = []
        
        with torch.no_grad():
            for trajs_tensor, home_ids_tensor in tqdm(collate_batch_data(
                self.uid_traj, test_data, self.params), desc="Reconstruction"):
                
                trajs_tensor = trajs_tensor.to(self.device)
                home_ids_tensor = home_ids_tensor.to(self.device)
                
                # 前向传播
                output = self.model(trajs_tensor, home_ids_tensor)
                loss = output.loss
                
                total_loss += loss.item()
                total_samples += trajs_tensor.size(0)
                losses.append(loss.item())
        
        avg_loss = total_loss / total_samples
        
        return {
            'avg_reconstruction_loss': avg_loss,
            'total_samples': total_samples,
            'loss_std': np.std(losses),
            'loss_min': np.min(losses),
            'loss_max': np.max(losses)
        }
    
    def evaluate_trajectory_generation(self, home_locations: List[int], 
                                     num_samples: int = 100) -> Dict:
        """
        评估轨迹生成质量
        
        Args:
            home_locations: 家庭位置列表
            num_samples: 每个位置生成的样本数
            
        Returns:
            生成质量评估结果
        """
        print("Evaluating trajectory generation...")
        
        generated_trajs = []
        home_ids_tensor = torch.LongTensor(home_locations).to(self.device)
        
        with torch.no_grad():
            for _ in tqdm(range(num_samples), desc="Generating trajectories"):
                # 生成轨迹
                output = self.model.generate(home_ids_tensor)
                generated_trajs.append(output.logits.cpu().numpy())
        
        # 计算生成轨迹的统计信息
        generated_trajs = np.array(generated_trajs)
        
        # 计算多样性指标
        diversity_scores = self._calculate_diversity(generated_trajs)
        
        # 计算合理性指标
        validity_scores = self._calculate_validity(generated_trajs)
        
        return {
            'generated_trajectories': generated_trajs,
            'diversity_scores': diversity_scores,
            'validity_scores': validity_scores,
            'num_generated': len(generated_trajs)
        }
    
    def _calculate_diversity(self, trajectories: np.ndarray) -> Dict:
        """计算轨迹多样性指标"""
        # 计算不同位置的数量
        unique_locations = []
        for traj in trajectories:
            unique_locations.append(len(np.unique(traj)))
        
        # 计算轨迹间的相似度
        similarities = []
        for i in range(len(trajectories)):
            for j in range(i+1, len(trajectories)):
                similarity = np.mean(trajectories[i] == trajectories[j])
                similarities.append(similarity)
        
        return {
            'avg_unique_locations': np.mean(unique_locations),
            'std_unique_locations': np.std(unique_locations),
            'avg_similarity': np.mean(similarities),
            'similarity_std': np.std(similarities)
        }
    
    def _calculate_validity(self, trajectories: np.ndarray) -> Dict:
        """计算轨迹合理性指标"""
        # 检查是否有重复的连续位置
        consecutive_repeats = []
        for traj in trajectories:
            repeats = 0
            for i in range(1, len(traj)):
                if traj[i] == traj[i-1]:
                    repeats += 1
            consecutive_repeats.append(repeats)
        
        # 检查轨迹长度
        valid_lengths = []
        for traj in trajectories:
            valid_lengths.append(len(traj) == self.params.traj_length)
        
        return {
            'avg_consecutive_repeats': np.mean(consecutive_repeats),
            'valid_length_ratio': np.mean(valid_lengths),
            'max_consecutive_repeats': np.max(consecutive_repeats)
        }
    
    def evaluate_location_prediction(self, test_data: List[Tuple]) -> Dict:
        """
        评估位置预测准确性
        
        Args:
            test_data: 测试数据
            
        Returns:
            位置预测评估结果
        """
        print("Evaluating location prediction...")
        
        predictions = []
        ground_truth = []
        
        with torch.no_grad():
            for trajs_tensor, home_ids_tensor in tqdm(collate_batch_data(
                self.uid_traj, test_data, self.params), desc="Location prediction"):
                
                trajs_tensor = trajs_tensor.to(self.device)
                home_ids_tensor = home_ids_tensor.to(self.device)
                
                # 获取预测
                output = self.model(trajs_tensor, home_ids_tensor)
                pred = torch.argmax(output.logits, dim=-1)
                
                predictions.extend(pred.cpu().numpy().flatten())
                ground_truth.extend(trajs_tensor.cpu().numpy().flatten())
        
        # 计算准确率
        accuracy = np.mean(np.array(predictions) == np.array(ground_truth))
        
        # 计算每个位置的准确率
        location_accuracy = defaultdict(list)
        for pred, true in zip(predictions, ground_truth):
            location_accuracy[true].append(pred == true)
        
        location_accuracies = {loc: np.mean(accs) for loc, accs in location_accuracy.items()}
        
        return {
            'overall_accuracy': accuracy,
            'location_accuracies': location_accuracies,
            'avg_location_accuracy': np.mean(list(location_accuracies.values())),
            'num_predictions': len(predictions)
        }
    
    def evaluate_temporal_patterns(self, test_data: List[Tuple]) -> Dict:
        """
        评估时间模式保持
        
        Args:
            test_data: 测试数据
            
        Returns:
            时间模式评估结果
        """
        print("Evaluating temporal patterns...")
        
        original_patterns = []
        reconstructed_patterns = []
        
        with torch.no_grad():
            for trajs_tensor, home_ids_tensor in tqdm(collate_batch_data(
                self.uid_traj, test_data, self.params), desc="Temporal patterns"):
                
                trajs_tensor = trajs_tensor.to(self.device)
                home_ids_tensor = home_ids_tensor.to(self.device)
                
                # 获取重建轨迹
                output = self.model(trajs_tensor, home_ids_tensor)
                pred = torch.argmax(output.logits, dim=-1)
                
                # 计算时间模式（例如：位置变化频率）
                for i in range(trajs_tensor.size(0)):
                    original_traj = trajs_tensor[i].cpu().numpy()
                    reconstructed_traj = pred[i].cpu().numpy()
                    
                    # 计算位置变化次数
                    original_changes = np.sum(original_traj[1:] != original_traj[:-1])
                    reconstructed_changes = np.sum(reconstructed_traj[1:] != reconstructed_traj[:-1])
                    
                    original_patterns.append(original_changes)
                    reconstructed_patterns.append(reconstructed_changes)
        
        # 计算模式保持的相关系数
        correlation = np.corrcoef(original_patterns, reconstructed_patterns)[0, 1]
        
        # 计算模式差异
        pattern_diff = np.array(original_patterns) - np.array(reconstructed_patterns)
        
        return {
            'pattern_correlation': correlation,
            'avg_pattern_difference': np.mean(np.abs(pattern_diff)),
            'pattern_difference_std': np.std(pattern_diff),
            'original_pattern_mean': np.mean(original_patterns),
            'reconstructed_pattern_mean': np.mean(reconstructed_patterns)
        }
    
    def generate_evaluation_report(self, test_data: List[Tuple], 
                                 output_path: str = './evaluation_results/') -> Dict:
        """
        生成完整的评估报告
        
        Args:
            test_data: 测试数据
            output_path: 输出路径
            
        Returns:
            完整的评估结果
        """
        print("Generating comprehensive evaluation report...")
        
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        # 加载测试数据
        self.uid_traj = pickle.load(open(self.params.path_traj, 'rb'))
        
        # 执行各项评估
        results = {}
        
        # 1. 重建损失评估
        results['reconstruction'] = self.evaluate_reconstruction_loss(test_data)
        
        # 2. 位置预测评估
        results['location_prediction'] = self.evaluate_location_prediction(test_data)
        
        # 3. 时间模式评估
        results['temporal_patterns'] = self.evaluate_temporal_patterns(test_data)
        
        # 4. 轨迹生成评估
        home_locations = [data[2] for data in test_data[:10]]  # 取前10个位置进行生成测试
        results['trajectory_generation'] = self.evaluate_trajectory_generation(home_locations)
        
        # 保存结果
        with open(os.path.join(output_path, 'evaluation_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        # 生成可视化报告
        self._generate_visualization_report(results, output_path)
        
        # 打印摘要
        self._print_summary(results)
        
        return results
    
    def _generate_visualization_report(self, results: Dict, output_path: str):
        """生成可视化报告"""
        print("Generating visualization report...")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('VAE-LLM Mobility Model Evaluation Report', fontsize=16)
        
        # 1. 重建损失分布
        if 'reconstruction' in results:
            axes[0, 0].hist([results['reconstruction']['avg_reconstruction_loss']], 
                          bins=20, alpha=0.7, color='blue')
            axes[0, 0].set_title('Reconstruction Loss Distribution')
            axes[0, 0].set_xlabel('Loss')
            axes[0, 0].set_ylabel('Frequency')
        
        # 2. 位置预测准确率
        if 'location_prediction' in results:
            axes[0, 1].bar(['Overall Accuracy'], 
                         [results['location_prediction']['overall_accuracy']], 
                         color='green', alpha=0.7)
            axes[0, 1].set_title('Location Prediction Accuracy')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_ylim(0, 1)
        
        # 3. 时间模式相关性
        if 'temporal_patterns' in results:
            axes[1, 0].scatter([results['temporal_patterns']['pattern_correlation']], 
                             [results['temporal_patterns']['avg_pattern_difference']], 
                             color='red', s=100)
            axes[1, 0].set_title('Temporal Pattern Correlation vs Difference')
            axes[1, 0].set_xlabel('Correlation')
            axes[1, 0].set_ylabel('Average Pattern Difference')
        
        # 4. 轨迹多样性
        if 'trajectory_generation' in results:
            diversity = results['trajectory_generation']['diversity_scores']
            axes[1, 1].bar(['Avg Unique Locations', 'Avg Similarity'], 
                         [diversity['avg_unique_locations'], diversity['avg_similarity']], 
                         color='orange', alpha=0.7)
            axes[1, 1].set_title('Trajectory Generation Diversity')
            axes[1, 1].set_ylabel('Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'evaluation_visualization.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {output_path}")
    
    def _print_summary(self, results: Dict):
        """打印评估摘要"""
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        if 'reconstruction' in results:
            rec = results['reconstruction']
            print(f"Reconstruction Loss: {rec['avg_reconstruction_loss']:.4f} ± {rec['loss_std']:.4f}")
        
        if 'location_prediction' in results:
            loc = results['location_prediction']
            print(f"Location Prediction Accuracy: {loc['overall_accuracy']:.4f}")
            print(f"Average Location Accuracy: {loc['avg_location_accuracy']:.4f}")
        
        if 'temporal_patterns' in results:
            temp = results['temporal_patterns']
            print(f"Temporal Pattern Correlation: {temp['pattern_correlation']:.4f}")
            print(f"Average Pattern Difference: {temp['avg_pattern_difference']:.4f}")
        
        if 'trajectory_generation' in results:
            gen = results['trajectory_generation']
            div = gen['diversity_scores']
            print(f"Trajectory Diversity - Avg Unique Locations: {div['avg_unique_locations']:.2f}")
            print(f"Trajectory Similarity: {div['avg_similarity']:.4f}")
        
        print("="*50)


def main():
    """主函数"""
    # 设置参数
    params, remaining_args = param_settings('SH')
    
    # 模型路径
    model_path = params.path_save
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist!")
        return
    
    # 创建评估器
    evaluator = MobilityEvaluator(model_path, params, device=params.device)
    
    # 加载测试数据
    uid_traj = pickle.load(open(params.path_traj, 'rb'))
    uid_mask_day = pickle.load(open(params.uid_mask_day, 'rb'))
    user_attr = pickle.load(open(params.path_attr, 'rb'))
    
    # 获取测试数据
    batch_all = get_batch_home_info(uid_traj, uid_mask_day, user_attr)
    test_data = batch_all['test'][:1000]  # 使用前1000个测试样本
    
    print(f"Testing on {len(test_data)} samples...")
    
    # 执行评估
    results = evaluator.generate_evaluation_report(
        test_data, 
        output_path=f'./evaluation_results/{time.strftime("%Y%m%d_%H%M%S")}/'
    )
    
    print("Evaluation completed successfully!")


if __name__ == '__main__':
    main() 