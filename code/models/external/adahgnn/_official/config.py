import os
import torch

# ===================================================================
# 6. 配置类 (Config Class)
# ===================================================================
class Config:
    def __init__(self, root, file, dataset_class, save_dir, total_sub, 
                 num_channels,       # 必要的EEG通道数
                 time_sample_num,    # 每个样本的时间点总数
                 patch_size,         # 时间分块的大小
                 batch_size=None, lr=None, epoch=None):
        
        # --- 数据与路径参数 ---
        self.root = root
        self.file: str = file
        self.subject = '1'
        self.total_sub = total_sub
        self.dataset_class = dataset_class
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # --- 训练超参数 ---
        self.batch_size = batch_size or 60
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_decay = 1e-3
        self.learning_rate = lr or 1e-3
        self.patience = 1
        self.factor = 0.9
        self.epochs = epoch or 100

        # --- 模型架构核心参数 ---
        self.num_channels = num_channels
        self.time_sample_num = time_sample_num 
        self.patch_size = patch_size           
        
        # --- 根据通道数，动态生成多尺度结构和池化映射 ---
        self._generate_model_architecture()
        self._generate_mappings()

    def _generate_model_architecture(self):
        """根据通道数，动态配置模型的超参数。"""
        if self.num_channels == 64:
            self.scale_node_nums = [64, 16, 4]
            self.hyper_num = [40, 20, 10]
        elif self.num_channels == 32:
            self.scale_node_nums = [32, 8, 2]
            self.hyper_num = [30, 15, 8]
        else:
            raise ValueError(f"不支持的通道数: {self.num_channels}。请在Config类中添加对它的支持。")
        
        # 通用模型参数
        self.d_model = 128
        self.k = 5
        self.num_classes = 2
        self.dropout = 0.5

    def _generate_mappings(self):
        """根据 scale_node_nums 动态创建池化映射关系。"""
        self.all_mappings = []
        for i in range(len(self.scale_node_nums) - 1):
            nodes_from = self.scale_node_nums[i]
            nodes_to = self.scale_node_nums[i+1]
            if nodes_from % nodes_to != 0:
                raise ValueError(f"池化层级不兼容: 无法从 {nodes_from} 个节点均匀池化到 {nodes_to} 个节点。")
            ratio = nodes_from // nodes_to
            mapping = [list(range(j, j + ratio)) for j in range(0, nodes_from, ratio)]
            self.all_mappings.append(mapping)
        print(f"为 {self.num_channels} 通道数据动态生成的池化映射层数: {len(self.all_mappings)}")

    @property
    def file_name(self) -> str:
        return self.file.format(**self.__dict__)

    def __str__(self):
        """返回格式化的配置信息字符串"""
        info = "\n" + "="*20 + " CONFIGURATION " + "="*20 + "\n"
        for key, value in self.__dict__.items():
            if key not in ['all_mappings', 'dataset_class']: # 过滤掉太长的或类对象
                info += f"{key:<20}: {value}\n"
        info += "="*55 + "\n"
        return info