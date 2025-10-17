import subprocess
import itertools
import time
import os
import tensorflow as tf

PROGRESS_FILE = "completed_runsv2.txt"

class NeuronComboRunner:
    def __init__(self):
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, "r") as f:
                self.completed = set(int(line.strip()) for line in f if line.strip())
        else:
            self.completed = set()

        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Detected {len(gpus)} available GPU(s)")
        print(f"Resuming, already completed runs: {sorted(self.completed)}")

    def _clear_gpu_memory(self):
        tf.keras.backend.clear_session()

    def generate_layer_combos(self):
        sizes = [32, 64, 128, 256]
        for combo in itertools.product(sizes, repeat=3):  # Only 3-layer combinations
            yield list(combo)

    def generate_commands(self):
        optimizers  = ['adam', 'nadam', 'SGD']
        r_values    = [1e-4, 1e-3, 1e-2]
        lr_values   = [1e-4, 5e-4, 1e-3]
        batch_size  = 32
        momentum    = 0.8  # Fixed momentum

        for optimizer in optimizers:
            for layers in self.generate_layer_combos():
                hidden_str = ",".join(str(n) for n in layers)
                for r in r_values:
                    for lr in lr_values:
                        cmd = [
                            "python3", "pre_processing.py",
                            "--optimizer",     optimizer,
                            "--momentum",      str(momentum),
                            "--lr",            str(lr),
                            "--epochs",        "1100",
                            "--b_size",        str(batch_size),
                            "--more_layers",   "True",
                            "--hidden_layers", hidden_str,
                            "--use_l2",        "True",  # Only L2
                            "--r",             str(r),
                        ]
                        yield cmd

    def run_experiment(self, cmd, run_id):
        self._clear_gpu_memory()
        cmd_with_id = cmd + ["--run_id", str(run_id)]
        print(f"\n=== Run {run_id} ===")
        print("Running:", " ".join(cmd_with_id))
        start = time.time()
        try:
            subprocess.run(cmd_with_id, check=True, text=True, timeout=14400)
            print(f"✓ Completed in {time.time() - start:.1f}s")
            return True
        except subprocess.TimeoutExpired:
            print(f"⚠ Timeout after {time.time() - start:.1f}s")
            return False
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed (exit code {e.returncode})")
            return False

    def run_all(self):
        total = success = 0
        for run_id, cmd in enumerate(self.generate_commands(), start=1):
            if run_id in self.completed:
                continue

            total += 1
            try:
                if self.run_experiment(cmd, run_id):
                    success += 1
                    with open(PROGRESS_FILE, "a") as f:
                        f.write(f"{run_id}\n")
                    self.completed.add(run_id)
                time.sleep(1)
            except KeyboardInterrupt:
                print("\nInterrupted by user; exiting. Run again to resume.")
                break

        print(f"\nTotal new runs executed: {success}, skipped: {len(self.completed)}")

if __name__ == "__main__":
    NeuronComboRunner().run_all()
