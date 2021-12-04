### Instructions for part 2

1. Train model on given `txt` file:
   ```bash
   python train.py --txt_file assets/book_EN_grimms_fairy_tails.txt
   ```
   Results (plots) will be saved to `results/book_EN_grimms_fairy_tails_loss.png`.
   Checkpoints will be saved to `checkpoints/book_EN_grimms_fairy_tails/`.
2. Generate sample text
   ```bash
   python generate.py --ckpt_path ./checkpoints/book_EN_grimms_fairy_tails/ckpt_epoch_best.pt.tar
   ```
   Logs will be saved to `logs/book_EN_grimms_fairy_tails/`.