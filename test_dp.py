import torch
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

model_name = "knowledgator/gliclass-modern-base-v2.0-init"
model = GLiClassModel.from_pretrained(model_name)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
pipeline = ZeroShotClassificationPipeline(
    model, tokenizer,
    classification_type='multi-label',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

texts = ["This is a test document."] * 10
topics = ["science", "sports"]
res = pipeline(texts, topics, batch_size=10)
print(res)
