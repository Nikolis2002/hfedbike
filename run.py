# run_experiments.py

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
        for combo in itertools.product(sizes, repeat=3):
            yield list(combo)
        for combo in itertools.product(sizes, repeat=2):
            yield list(combo)

    def generate_commands(self):
        optimizers  = ['adam', 'nadam','SGD']
        methods     = ['l1', 'l2']  # only L1 and L2
        # regularization strengths for l1/l2
        r_values    = [1e-4, 1e-3, 1e-2]
        lr_values   = [1e-4,5e-4,1e-3]
        batch_size  = 32  # fixed batch size

        for optimizer in optimizers:
            # For SGD test two momentum values; for others momentum=0
            momentums = [0.2, 0.6] if optimizer == 'SGD' else [0.0]

            for momentum in momentums:
                for layers in self.generate_layer_combos():
                    hidden_str = ",".join(str(n) for n in layers)

                    for method in methods:
                        # L1 or L2: sweep r_values
                        for r in r_values:
                            for lr in lr_values:
                                cmd = [
                                    "python3", "pre_processing.py",
                                    "--optimizer",       optimizer,
                                    "--momentum",        str(momentum),
                                    "--lr",              str(lr),
                                    "--epochs",          "1100",
                                    "--b_size",          str(batch_size),
                                    "--more_layers",     "True",
                                    "--hidden_layers",   hidden_str
                                ]
                                # toggle L1/L2 flag exclusively
                                if method == 'l1':
                                    cmd += ["--use_l1", "True"]
                                else:  # l2
                                    cmd += ["--use_l2", "True"]
                                # add regularization factor
                                cmd += ["--r", str(r)]
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
        total = success = 0
        for run_id, cmd in enumerate(self.generate_commands(), start=1):
            total += 1
            if self.run_experiment(cmd, run_id):
                success += 1
            time.sleep(1)
        print(f"\nCompleted {success}/{total} experiments successfully")

if __name__ == "__main__":
    NeuronComboRunner().run_all()
