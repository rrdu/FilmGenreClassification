import torch
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassAccuracy

# Import training definitions
from utils.module import MoE_LightningModule, IMDBDataset
from layers.encoder import SBERT_MoE_Model
from train import k, num_experts, expert_hidden_dim, encoder_kwargs, DATA_DIR


if __name__ == "__main__":

    # ==========================================
    # 1. CONFIG
    # ==========================================
    CHECKPOINT_PATH = "checkpoints/moe-film-epoch=06-val_loss=0.63.ckpt"  # update if needed
    TEST_FILE = "imdb_arh_test.csv"  # name of your test CSV
    BATCH_SIZE = 64

    # ==========================================
    # 2. CLASS DISCOVERY + DATASET
    # ==========================================
    CLASS_NAMES = IMDBDataset.discover_classes(DATA_DIR, 'imdb_arh_train.csv')
    print(f"Loaded {len(CLASS_NAMES)} classes: {CLASS_NAMES}")

    tr_ds = IMDBDataset(data_dir_path=DATA_DIR, filename='imdb_arh_train.csv', class_names=CLASS_NAMES)
    te_ds = IMDBDataset(data_dir_path=DATA_DIR, filename=TEST_FILE, class_names=CLASS_NAMES)
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ==========================================
    # 3. MODEL LOADING
    # ==========================================
    num_classes = len(CLASS_NAMES)

    # Initialize backbone with same hyperparameters as training
    backbone_skeleton = SBERT_MoE_Model(
        num_classes=num_classes,
        num_experts=num_experts,
        expert_hidden_dim=expert_hidden_dim,
        vocab_texts=tr_ds.texts,
        encoder_kwargs=encoder_kwargs,
        top_k=k
    )

    # Load checkpoint
    print(f"Loading model from checkpoint: {CHECKPOINT_PATH}")
    loaded_model = MoE_LightningModule.load_from_checkpoint(
        CHECKPOINT_PATH,
        model=backbone_skeleton,
        num_classes=num_classes,
        num_experts=num_experts,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model.to(device)
    loaded_model.eval()

    # ==========================================
    # 4. METRICS
    # ==========================================
    metrics = MetricCollection({
        "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro"),   # keep accuracy as-is
        "f1_macro": MulticlassF1Score(num_classes=num_classes, average="macro"),
        "precision_macro": MulticlassPrecision(num_classes=num_classes, average="macro"),
        "recall_macro": MulticlassRecall(num_classes=num_classes, average="macro"),
        "f1_weighted": MulticlassF1Score(num_classes=num_classes, average="weighted"),
    }).to(device)

    def map_logits_to_labels_multiclass(logits, class_names):
        probs = torch.softmax(logits, dim=1).cpu()
        pred_indices = torch.argmax(probs, dim=1)
        batch_results = [class_names[idx.item()] for idx in pred_indices]
        return batch_results, probs


    # ==========================================
    # 5. INFERENCE LOOP
    # ==========================================
    print("\n--- Running Inference on Test Loader (Multi-Class) ---")

    with torch.no_grad():
        for batch_idx, batch in enumerate(te_loader):
            texts, true_label_indices = batch
            true_label_indices = true_label_indices.to(device)

            logits, _ = loaded_model(texts)
            metrics.update(logits, true_label_indices)

            if batch_idx < 3:
                preds, probs = map_logits_to_labels_multiclass(logits, CLASS_NAMES)
                for i, (text, pred, p) in enumerate(zip(texts, preds, probs)):
                    actual_lbl = CLASS_NAMES[true_label_indices[i].item()]
                    sig_probs = {CLASS_NAMES[j]: round(p[j].item(), 3) for j in range(len(CLASS_NAMES)) if p[j] > 0.1}
                    print(f"\n[Batch {batch_idx} - Sample {i}]")
                    print(f"Text: {text[:100]}...")
                    print(f"Predicted: {pred}")
                    print(f"Actual: {actual_lbl}")
                    print(f"High probs: {sig_probs}")

    # ==========================================
    # 6. FINAL METRICS
    # ==========================================
    print("\n" + "="*30)
    print("FINAL EVALUATION REPORT (MULTICLASS)")
    print("="*30)

    final_results = metrics.compute()
    for metric_name, value in final_results.items():
        print(f"{metric_name.capitalize()}: {value.item():.4f}")

    metrics.reset()
