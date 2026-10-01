import sys, config
from src.model_utils import train_and_save
if __name__ == "__main__":
    res, best, imp = train_and_save(sys.argv[1] if len(sys.argv) > 1 else config.DATASET)
    for n, m in res.items(): print(f"{n:20s} test_acc={m['accuracy']:.3f} test_prec={m['precision']:.3f} test_rec={m['recall']:.3f} test_f1={m['f1']:.3f}")
    print("Selected by validation F1; saved best model:", best)
