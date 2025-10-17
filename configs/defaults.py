"""
Enhanced OWOD Configuration System
Defines all configuration parameters for the Open World Object Detection with Continual Learning
"""

from detectron2.config import CfgNode as CN

def add_owod_cl_config(cfg):
    """
    Add config for Enhanced OWOD with Continual Learning
    """
    
    # ============================================================================== #
    # Foundation Model Configuration
    # ============================================================================== #
    cfg.FOUNDATION_MODEL = CN()
    cfg.FOUNDATION_MODEL.ENABLED = True
    cfg.FOUNDATION_MODEL.TYPE = "clip"  # Options: clip, dinov2, sam, hybrid
    
    # CLIP Configuration
    cfg.FOUNDATION_MODEL.CLIP = CN()
    cfg.FOUNDATION_MODEL.CLIP.MODEL_NAME = "ViT-B/16"  # ViT-B/16, ViT-L/14, RN50
    cfg.FOUNDATION_MODEL.CLIP.PRETRAINED = "openai"
    cfg.FOUNDATION_MODEL.CLIP.FREEZE_VISION = True
    cfg.FOUNDATION_MODEL.CLIP.FREEZE_TEXT = True
    cfg.FOUNDATION_MODEL.CLIP.EMBED_DIM = 512
    
    # DINOv2 Configuration
    cfg.FOUNDATION_MODEL.DINOV2 = CN()
    cfg.FOUNDATION_MODEL.DINOV2.MODEL_NAME = "dinov2_vitb14"  # dinov2_vits14, dinov2_vitb14, dinov2_vitl14
    cfg.FOUNDATION_MODEL.DINOV2.PRETRAINED = True
    cfg.FOUNDATION_MODEL.DINOV2.FREEZE = True
    cfg.FOUNDATION_MODEL.DINOV2.USE_REGISTERS = True
    
    # SAM Configuration
    cfg.FOUNDATION_MODEL.SAM = CN()
    cfg.FOUNDATION_MODEL.SAM.ENABLED = False
    cfg.FOUNDATION_MODEL.SAM.CHECKPOINT = "sam_vit_h_4b8939.pth"
    cfg.FOUNDATION_MODEL.SAM.MODEL_TYPE = "vit_h"  # vit_h, vit_l, vit_b
    
    # ============================================================================== #
    # Parameter-Efficient Fine-Tuning (PEFT)
    # ============================================================================== #
    cfg.PEFT = CN()
    cfg.PEFT.ENABLED = True
    cfg.PEFT.METHOD = "lora"  # Options: lora, adapter, prefix, prompt_tuning, bitfit
    
    # LoRA Configuration
    cfg.PEFT.LORA = CN()
    cfg.PEFT.LORA.R = 8  # Rank of low-rank matrices
    cfg.PEFT.LORA.ALPHA = 16  # Scaling factor
    cfg.PEFT.LORA.DROPOUT = 0.1
    cfg.PEFT.LORA.TARGET_MODULES = ["q_proj", "v_proj"]  # Which layers to adapt
    cfg.PEFT.LORA.BIAS = "none"  # Options: none, all, lora_only
    cfg.PEFT.LORA.MERGE_WEIGHTS = False
    
    # Adapter Configuration
    cfg.PEFT.ADAPTER = CN()
    cfg.PEFT.ADAPTER.BOTTLENECK_SIZE = 64
    cfg.PEFT.ADAPTER.DROPOUT = 0.1
    cfg.PEFT.ADAPTER.INIT_SCALE = 1e-3
    cfg.PEFT.ADAPTER.RESIDUAL_BEFORE_LN = True
    
    # Prompt Tuning Configuration
    cfg.PEFT.PROMPT = CN()
    cfg.PEFT.PROMPT.NUM_TOKENS = 10
    cfg.PEFT.PROMPT.INIT_METHOD = "random"  # random, vocab, class_labels
    cfg.PEFT.PROMPT.DEEP = True  # Deep prompt tuning
    cfg.PEFT.PROMPT.DROPOUT = 0.0
    
    # Prefix Tuning Configuration
    cfg.PEFT.PREFIX = CN()
    cfg.PEFT.PREFIX.NUM_PREFIX = 20
    cfg.PEFT.PREFIX.PREFIX_PROJECTION = True
    cfg.PEFT.PREFIX.PROJECTION_HIDDEN_SIZE = 512
    
    # ============================================================================== #
    # Continual Learning Configuration
    # ============================================================================== #
    cfg.CONTINUAL_LEARNING = CN()
    cfg.CONTINUAL_LEARNING.ENABLED = True
    cfg.CONTINUAL_LEARNING.STRATEGY = "replay"  # replay, ewc, lwf, gem, agem, icarl, der
    cfg.CONTINUAL_LEARNING.NUM_TASKS = 4
    cfg.CONTINUAL_LEARNING.CURRENT_TASK = 1
    
    # Memory Replay Configuration
    cfg.CONTINUAL_LEARNING.REPLAY = CN()
    cfg.CONTINUAL_LEARNING.REPLAY.ENABLED = True
    cfg.CONTINUAL_LEARNING.REPLAY.MEMORY_SIZE = 2000  # Total exemplars
    cfg.CONTINUAL_LEARNING.REPLAY.SELECTION_STRATEGY = "herding"  # random, herding, entropy, forgetting
    cfg.CONTINUAL_LEARNING.REPLAY.BATCH_RATIO = 0.5  # Ratio of replay samples in batch
    cfg.CONTINUAL_LEARNING.REPLAY.UPDATE_STRATEGY = "reservoir"  # reservoir, ring_buffer
    
    # Knowledge Distillation Configuration
    cfg.CONTINUAL_LEARNING.DISTILLATION = CN()
    cfg.CONTINUAL_LEARNING.DISTILLATION.ENABLED = True
    cfg.CONTINUAL_LEARNING.DISTILLATION.TEMPERATURE = 2.0
    cfg.CONTINUAL_LEARNING.DISTILLATION.ALPHA = 0.5  # Weight for distillation loss
    cfg.CONTINUAL_LEARNING.DISTILLATION.TYPE = "logit"  # logit, feature, attention
    cfg.CONTINUAL_LEARNING.DISTILLATION.FEATURE_LAYERS = ["layer3", "layer4"]
    
    # Elastic Weight Consolidation (EWC)
    cfg.CONTINUAL_LEARNING.EWC = CN()
    cfg.CONTINUAL_LEARNING.EWC.ENABLED = False
    cfg.CONTINUAL_LEARNING.EWC.LAMBDA = 5000.0
    cfg.CONTINUAL_LEARNING.EWC.MODE = "online"  # online, separate
    cfg.CONTINUAL_LEARNING.EWC.GAMMA = 1.0  # Decay factor for online EWC
    
    # Dynamic Architecture Expansion
    cfg.CONTINUAL_LEARNING.EXPANSION = CN()
    cfg.CONTINUAL_LEARNING.EXPANSION.ENABLED = False
    cfg.CONTINUAL_LEARNING.EXPANSION.STRATEGY = "dynamic"  # dynamic, progressive, packnet
    cfg.CONTINUAL_LEARNING.EXPANSION.CAPACITY_INCREASE = 0.1
    
    # Progressive Neural Networks
    cfg.CONTINUAL_LEARNING.PROGRESSIVE = CN()
    cfg.CONTINUAL_LEARNING.PROGRESSIVE.ENABLED = False
    cfg.CONTINUAL_LEARNING.PROGRESSIVE.LATERAL_CONNECTIONS = True
    
    # ============================================================================== #
    # Multi-Modal Configuration
    # ============================================================================== #
    cfg.MULTIMODAL = CN()
    cfg.MULTIMODAL.ENABLED = True
    cfg.MULTIMODAL.FUSION_TYPE = "cross_attention"  # concat, add, cross_attention, film
    
    # Vision-Language Fusion
    cfg.MULTIMODAL.VISION_LANGUAGE = CN()
    cfg.MULTIMODAL.VISION_LANGUAGE.ENABLED = True
    cfg.MULTIMODAL.VISION_LANGUAGE.TEXT_ENCODER = "clip"  # clip, bert, roberta
    cfg.MULTIMODAL.VISION_LANGUAGE.FUSION_LAYERS = [3, 6, 9, 12]
    cfg.MULTIMODAL.VISION_LANGUAGE.ATTENTION_HEADS = 8
    cfg.MULTIMODAL.VISION_LANGUAGE.DROPOUT = 0.1
    
    # Cross-Modal Attention
    cfg.MULTIMODAL.CROSS_ATTENTION = CN()
    cfg.MULTIMODAL.CROSS_ATTENTION.NUM_HEADS = 8
    cfg.MULTIMODAL.CROSS_ATTENTION.HIDDEN_DIM = 512
    cfg.MULTIMODAL.CROSS_ATTENTION.NUM_LAYERS = 3
    cfg.MULTIMODAL.CROSS_ATTENTION.DROPOUT = 0.1
    
    # Text-Guided Detection
    cfg.MULTIMODAL.TEXT_GUIDED = CN()
    cfg.MULTIMODAL.TEXT_GUIDED.ENABLED = True
    cfg.MULTIMODAL.TEXT_GUIDED.USE_CLASS_PROMPTS = True
    cfg.MULTIMODAL.TEXT_GUIDED.PROMPT_TEMPLATE = "a photo of a {}"
    
    # ============================================================================== #
    # Open World Object Detection (OWOD) Specific
    # ============================================================================== #
    cfg.OWOD = CN()
    cfg.OWOD.ENABLED = True
    
    # Class Configuration
    cfg.OWOD.TOTAL_CLASSES = 81  # Total number of classes in dataset
    cfg.OWOD.PREV_INTRODUCED_CLS = 0  # Number of classes learned in previous tasks
    cfg.OWOD.CUR_INTRODUCED_CLS = 20  # Number of classes to learn in current task
    cfg.OWOD.UNKNOWN_CLASS_INDEX = 80  # Index for unknown class
    
    # Unknown Detection Configuration
    cfg.OWOD.UNKNOWN_DETECTION = CN()
    cfg.OWOD.UNKNOWN_DETECTION.ENABLED = True
    cfg.OWOD.UNKNOWN_DETECTION.METHOD = "energy"  # energy, max_logit, msp, odin
    cfg.OWOD.UNKNOWN_DETECTION.THRESHOLD = 0.5
    cfg.OWOD.UNKNOWN_DETECTION.TEMPERATURE = 1.5
    
    # Energy-Based Detection
    cfg.OWOD.ENERGY = CN()
    cfg.OWOD.ENERGY.ENABLED = True
    cfg.OWOD.ENERGY.COMPUTE_DIST = False
    cfg.OWOD.ENERGY.SAVE_PATH = "energy_dist.pkl"
    cfg.OWOD.ENERGY.MARGIN = 10.0
    
    # Contrastive Clustering
    cfg.OWOD.CLUSTERING = CN()
    cfg.OWOD.CLUSTERING.ENABLED = True
    cfg.OWOD.CLUSTERING.START_ITER = 500
    cfg.OWOD.CLUSTERING.UPDATE_MU_ITER = 1000
    cfg.OWOD.CLUSTERING.MOMENTUM = 0.9
    cfg.OWOD.CLUSTERING.Z_DIMENSION = 64
    cfg.OWOD.CLUSTERING.MARGIN = 10.0
    cfg.OWOD.CLUSTERING.ITEMS_PER_CLASS = 10
    cfg.OWOD.CLUSTERING.LOSS_WEIGHT = 0.1
    
    # Feature Store
    cfg.OWOD.FEATURE_STORE = CN()
    cfg.OWOD.FEATURE_STORE.ENABLED = True
    cfg.OWOD.FEATURE_STORE.SAVE_PATH = "feature_store.pkl"
    cfg.OWOD.FEATURE_STORE.UPDATE_INTERVAL = 100
    
    # Unknown Object Proposals
    cfg.OWOD.UNKNOWN_PROPOSALS = CN()
    cfg.OWOD.UNKNOWN_PROPOSALS.ENABLED = True
    cfg.OWOD.UNKNOWN_PROPOSALS.NUM_PER_IMAGE = 5
    cfg.OWOD.UNKNOWN_PROPOSALS.MIN_SCORE = 0.05
    
    # ============================================================================== #
    # Hardware-Aware Optimization
    # ============================================================================== #
    cfg.OPTIMIZATION = CN()
    
    # Mixed Precision Training
    cfg.OPTIMIZATION.MIXED_PRECISION = CN()
    cfg.OPTIMIZATION.MIXED_PRECISION.ENABLED = True
    cfg.OPTIMIZATION.MIXED_PRECISION.DTYPE = "fp16"  # fp16, bf16
    cfg.OPTIMIZATION.MIXED_PRECISION.LOSS_SCALE = "dynamic"
    cfg.OPTIMIZATION.MIXED_PRECISION.GROWTH_INTERVAL = 2000
    
    # Gradient Checkpointing
    cfg.OPTIMIZATION.GRADIENT_CHECKPOINTING = CN()
    cfg.OPTIMIZATION.GRADIENT_CHECKPOINTING.ENABLED = False
    cfg.OPTIMIZATION.GRADIENT_CHECKPOINTING.SEGMENTS = 4
    
    # Quantization
    cfg.OPTIMIZATION.QUANTIZATION = CN()
    cfg.OPTIMIZATION.QUANTIZATION.ENABLED = False
    cfg.OPTIMIZATION.QUANTIZATION.METHOD = "dynamic"  # dynamic, static, qat
    cfg.OPTIMIZATION.QUANTIZATION.BITS = 8
    cfg.OPTIMIZATION.QUANTIZATION.CALIBRATION_SAMPLES = 100
    
    # Pruning
    cfg.OPTIMIZATION.PRUNING = CN()
    cfg.OPTIMIZATION.PRUNING.ENABLED = False
    cfg.OPTIMIZATION.PRUNING.METHOD = "magnitude"  # magnitude, l1, movement
    cfg.OPTIMIZATION.PRUNING.AMOUNT = 0.3
    cfg.OPTIMIZATION.PRUNING.ITERATIVE = True
    cfg.OPTIMIZATION.PRUNING.SCHEDULE = "cubic"
    
    # Efficient Inference
    cfg.OPTIMIZATION.INFERENCE = CN()
    cfg.OPTIMIZATION.INFERENCE.TRT_ENABLED = False
    cfg.OPTIMIZATION.INFERENCE.ONNX_EXPORT = False
    cfg.OPTIMIZATION.INFERENCE.BATCH_SIZE = 1
    cfg.OPTIMIZATION.INFERENCE.NUM_WORKERS = 4
    
    # Memory Optimization
    cfg.OPTIMIZATION.MEMORY = CN()
    cfg.OPTIMIZATION.MEMORY.MAX_SPLIT_SIZE = 512  # MB
    cfg.OPTIMIZATION.MEMORY.EMPTY_CACHE_INTERVAL = 100
    
    # ============================================================================== #
    # Training Configuration
    # ============================================================================== #
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (15000, 18000)
    cfg.SOLVER.MAX_ITER = 20000
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000
    
    # Optimizer
    cfg.SOLVER.OPTIMIZER = "AdamW"  # SGD, Adam, AdamW
    cfg.SOLVER.WEIGHT_DECAY = 1e-4
    cfg.SOLVER.BIAS_LR_FACTOR = 1.0
    cfg.SOLVER.WEIGHT_DECAY_BIAS = 0.0
    
    # Learning Rate Scheduler
    cfg.SOLVER.LR_SCHEDULER = "WarmupMultiStepLR"  # WarmupMultiStepLR, CosineAnnealing
    cfg.SOLVER.COSINE_ANNEALING_T_MAX = 20000
    
    # Gradient Clipping
    cfg.SOLVER.CLIP_GRADIENTS = CN()
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"  # norm, value
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    
    # ============================================================================== #
    # Data Augmentation
    # ============================================================================== #
    cfg.INPUT.AUGMENTATION = CN()
    cfg.INPUT.AUGMENTATION.ENABLED = True
    cfg.INPUT.AUGMENTATION.METHODS = ["flip", "crop", "color_jitter"]
    
    # ============================================================================== #
    # Evaluation Configuration
    # ============================================================================== #
    cfg.TEST.EVAL_PERIOD = 2000
    cfg.TEST.UNKNOWN_EVAL = True
    cfg.TEST.COMPUTE_FORGETTING = True
    cfg.TEST.WILDERNESS_IMPACT = True
    
    # ============================================================================== #
    # Logging and Visualization
    # ============================================================================== #
    cfg.LOGGING = CN()
    cfg.LOGGING.WANDB_ENABLED = False
    cfg.LOGGING.WANDB_PROJECT = "enhanced-owod"
    cfg.LOGGING.WANDB_ENTITY = ""
    cfg.LOGGING.LOG_INTERVAL = 50
    cfg.LOGGING.SAVE_VISUALIZATIONS = True
    
    return cfg


def get_cfg():
    """
    Get a copy of the default config with OWOD CL additions
    """
    from detectron2.config import get_cfg as get_detectron2_cfg
    cfg = get_detectron2_cfg()
    cfg = add_owod_cl_config(cfg)
    return cfg
