from transformers import AutoImageProcessor, AutoModel
from peft import LoraConfig, get_peft_model
import torch

MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
backbone = AutoModel.from_pretrained(MODEL_NAME)

class DinoV3ClassifierLinearHead(torch.nn.Module):
    def __init__(
        self, 
        num_classes: int, 
        backbone_path: str = MODEL_NAME, 
        freeze_backbone: bool = True,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules=["q_proj", "v_proj", "k_proj"]
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_path)
        if use_lora:
            print("Converting backbone to LoRA...")
            # Define LoRA Config
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules, 
                lora_dropout=lora_dropout,
                bias="none",
                modules_to_save=[] # We only save the adapters
            )
            
            # Wrap the backbone
            self.backbone = get_peft_model(self.backbone, peft_config)
            self.backbone.print_trainable_parameters()
            
        elif freeze_backbone:
            print("Freezing backbone...")
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        
        hidden_size = getattr(backbone.config, "hidden_size", None)
        self.head = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        last_hidden = outputs.last_hidden_state
        cls = last_hidden[:, 0]
        logits = self.head(cls)
        return logits