# Updated run_experiments.py with batch size option

import subprocess
import itertools
import time
import tensorflow as tf

class NeuronComboRunner:
    def __init__(self):
        # Enable GPU memory growth
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Detected {len(gpus)} available GPU(s)")

    def _clear_gpu_memory(self):
        tf.keras.backend.clear_session()

    def generate_layer_combos(self):
        # only 2‑ and 3‑layer configurations
        sizes = [32, 64, 128, 256]
        for combo in itertools.product(sizes, repeat=2):
            yield list(combo)
        for combo in itertools.product(sizes, repeat=3):
            yield list(combo)

    def generate_commands(self):
        optimizers    = ['SGD', 'adam', 'nadam']
        reg_methods   = [('none', False, False), ('l1', True, False), ('l2', False, True)]
        r_values      = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        lr_values     = [1e-4, 1e-3, 1e-2]
        dropout_rates = [0.0, 0.1, 0.2, 0.3]
        batch_sizes   = [32, 64]  # new batch size options

        for optimizer in optimizers:
            # For SGD test two momentum values, for others momentum=0
            if optimizer == 'SGD':
                momentums = [0.2, 0.6]
            else:
                momentums = [0.0]

            for momentum in momentums:
                for layers in self.generate_layer_combos():
                    hidden_str = ",".join(str(n) for n in layers)
                    for method, use_l1, use_l2 in reg_methods:
                        for r in r_values:
                            # only allow r=0 for 'none', and r>0 for l1/l2
                            if (method == 'none' and r != 0.0) or (method != 'none' and r == 0.0):
                                continue
                            for lr in lr_values:
                                for dr in dropout_rates:
                                    for bs in batch_sizes:
                                        cmd = [
                                            "python3", "alzheimers_prediction.py",
                                            "--optimizer", optimizer,
                                            "--momentum",    str(momentum),
                                            "--lr",          str(lr),
                                            "--epochs",      "1100",
                                            "--use_l1",      str(use_l1),
                                            "--use_l2",      str(use_l2),
                                            "--r",           str(r),
                                            "--dropout_rate",str(dr),
                                            "--b_size",      str(bs),           # batch size flag
                                            "--more_layers", "True",
                                            "--hidden_layers", hidden_str
                                        ]
                                        yield cmd

    def run_experiment(self, cmd, run_id):
        self._clear_gpu_memory()
        # append run_id flag
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
        total, success = 0, 0
        for run_id, cmd in enumerate(self.generate_commands(), start=1):
            total += 1
            if self.run_experiment(cmd, run_id):
                success += 1
            time.sleep(1)
        print(f"\nCompleted {success}/{total} experiments successfully")

if __name__ == "__main__":
    NeuronComboRunner().run_all()
